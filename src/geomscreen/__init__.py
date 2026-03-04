"""geomscreen.tasks

Opinionated helpers for building :class:`dplutils.pipeline.PipelineTask` objects.

Assumptions (explicit by design)
--------------------------------
- **ASE is installed.**
- **Geometry columns contain single-frame extxyz strings (or missing values).**
- **Action functions raise exceptions on real failures.** The task wrappers
  catch these exceptions per-row, log them, and mark the row as completed with
  an error status.

Reruns / resume
---------------
Each task (except filter tasks) creates a per-row status column
``<task_name>_status`` (and an error column ``<task_name>_error``). Reruns are
controlled by this status column:

- If ``status`` is already set (any non-empty value), the row is skipped.
- To retry a row, clear the status cell (and any outputs you want recomputed).

ASE action semantics
--------------------
``ase_task`` supports either a single callable action, or an *action sequence*
(a tuple of callables). In a sequence:

- All but the last step are *intermediate steps* that must return ``Atoms`` or
  ``None``.
- ``None`` from an intermediate step means "I mutated the input Atoms in-place".
- Returning ``Atoms`` means "use this Atoms for subsequent steps".
- The final step returns an :class:`ActionResult`.

Interpreting ``ActionResult``:

- **Single-output task** (``outcol`` is a single column)
    - return ``Atoms`` -> write extxyz
    - return ``None``  -> interpreted as "mutated in-place"; write extxyz of the
      mutated input
    - return scalar    -> store directly

- **Multi-output task** (``outcol`` is a tuple of columns)
    - the action must return a tuple of the same length
    - each item is stored in the corresponding output column
    - ``None`` *inside* the tuple means "completed, but no value for this output"
      (the row is still considered complete and will not be rerun)

Atoms are coerced to extxyz **whenever they appear in outputs**, regardless of
column name.
"""

import io
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from types import SimpleNamespace
from typing import Any, cast

import ase.io
import pandas as pd
from ase import Atoms
from pandas.api.typing.aliases import Scalar

from dplutils import observer
from dplutils.pipeline import PipelineTask

from fairchem.core.units.mlip_unit._batch_serve import setup_batch_predict_server
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from ray import serve

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

ColName = str
ColNames = ColName | tuple[ColName, ...]

GeomAction = Atoms | None

TableValue = Scalar
ActionSingleResult = GeomAction | TableValue

MultiResultItem = Atoms | TableValue | None
ActionMultiResult = tuple[MultiResultItem, ...]

ActionResult = ActionSingleResult | ActionMultiResult

ActionFunc = Callable[[Atoms], ActionResult]
ActionIntermediateStep = Callable[[Atoms], GeomAction]
ActionSeq = tuple[*tuple[ActionIntermediateStep, ...], ActionFunc]
Action = ActionFunc | ActionSeq

BatchActionFunc = Callable[[Atoms, Any], ActionResult]
BatchActionSetup = Callable[[Atoms, Any], GeomAction]
BatchActionSeq = tuple[BatchActionSetup, *tuple[ActionIntermediateStep, ...], ActionFunc]
BatchAction = BatchActionFunc | BatchActionSeq

# -----------------------------------------------------------------------------
# Logging (match bluephos default format)
# -----------------------------------------------------------------------------


def _setup_logging(name: str) -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    return logging.getLogger(name)


logger = _setup_logging(__name__)


_OMOL_DATASET_TO_TASKS = {
    "omol": [
        SimpleNamespace(property="energy"),
        SimpleNamespace(property="forces"),
    ]
}


class _ServePredictorProxy:
    """Small calculator-facing proxy for a FAIRChem Ray Serve app.

    This is intentionally narrow and omol-focused. The important part is that
    it does *not* fetch raw ``dataset_to_tasks`` from the Serve replica,
    because those Task objects can pull much heavier state through Ray than the
    calculator actually needs.

    For the current geomscreen example, FAIRChemCalculator only needs enough
    local metadata to accept ``task_name="omol"`` and expose the usual energy /
    forces properties. The real inference still happens entirely inside the
    Serve deployment.
    """

    def __init__(self, server_handle: Any):
        self.server_handle = server_handle
        self.dataset_to_tasks = _OMOL_DATASET_TO_TASKS
        self.inference_settings = self._get_predict_unit_attribute("inference_settings")
        self._atom_refs: dict | None | object = None

    def _get_predict_unit_attribute(self, attribute_name: str) -> Any:
        return self.server_handle.get_predict_unit_attribute.remote(attribute_name).result()

    @property
    def atom_refs(self) -> dict | None:
        if self._atom_refs is None:
            self._atom_refs = self._get_predict_unit_attribute("atom_refs")
        return cast(dict | None, self._atom_refs)

    def predict(self, data: Any, undo_element_references: bool = True) -> dict:
        return self.server_handle.predict.remote(data, undo_element_references).result()

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _as_cols(cols: ColNames) -> tuple[str, ...]:
    return (cols,) if isinstance(cols, str) else cols


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _ensure_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = None


def _is_unset(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip() == ""
    try:
        return bool(pd.isna(val))
    except Exception:
        return False


def _is_bad_input(val: object) -> bool:
    return _is_unset(val)


def _row_id(row: pd.Series) -> str:
    return str(row.name)


def _read_extxyz(text: str) -> Atoms:
    # index=0 enforces single-frame semantics.
    return cast(Atoms, ase.io.read(io.StringIO(text), format="extxyz", index=0))


def _write_extxyz(atoms: Atoms) -> str:
    buf = io.StringIO()
    ase.io.write(buf, atoms, format="extxyz")
    return buf.getvalue()


def _assign_failure(row: pd.Series, outcols: tuple[str, ...]) -> pd.Series:
    for c in outcols:
        row[c] = pd.NA
    return row


def _action_label(action: Action | BatchAction) -> str:
    f = action[-1] if isinstance(action, tuple) else action
    return getattr(f, "__name__", type(f).__name__)


def _run_action(action: Action, atoms: Atoms, info_fields: dict) -> tuple[ActionResult, Atoms]:
    """Run an Action (callable or ActionSeq) and return (result, current_atoms)."""

    if not isinstance(action, tuple):
        return action(atoms), atoms

    # tuple form: (step1, step2, ..., final)
    *steps, final = action

    for step in steps:
        # Guarantee all required info present in the atoms.info dict
        atoms.info.update(info_fields)
        out = step(atoms)
        if out is None:
            continue  # mutated in-place
        if isinstance(out, Atoms):
            atoms = out
            continue
        raise TypeError(
            "ActionIntermediateStep must return ase.Atoms or None; "
            f"got {type(out).__name__} from {getattr(step, '__name__', step)!r}"
        )
    # Remove the additional info fields, they're intended to be temporary
    for k in info_fields:
        atoms.info.pop(k, None)
    return final(atoms), atoms


def _run_batch_action(
    action: BatchAction,
    atoms: Atoms,
    predict_unit: Any,
    info_fields: dict,
) -> tuple[ActionResult, Atoms]:
    """Run a FAIRChem batched Action and return (result, current_atoms).

    `BatchAction` mirrors the `ase_task` Action API, but the first step (or the
    entire callable) additionally receives a calculator-facing predictor proxy.

    Cleanup: any `info_fields` injected into `atoms.info` are removed before
    returning.
    """

    # Callable form: action(atoms, predict_unit) -> ActionResult
    if not isinstance(action, tuple):
        atoms.info.update(info_fields)
        try:
            return action(atoms, predict_unit), atoms
        finally:
            for k in info_fields:
                atoms.info.pop(k, None)

    # Tuple form: (setup_step, *intermediate_steps, final)
    setup_step, *rest = action

    atoms.info.update(info_fields)
    try:
        out = setup_step(atoms, predict_unit)
        if out is not None and not isinstance(out, Atoms):
            raise TypeError(
                "BatchActionSetup must return ase.Atoms or None; "
                f"got {type(out).__name__} from {getattr(setup_step, '__name__', setup_step)!r}"
            )
        if isinstance(out, Atoms):
            atoms = out

        # The remaining steps are a standard geomscreen `Action`.
        return _run_action(tuple(rest), atoms, info_fields)
    finally:
        for k in info_fields:
            atoms.info.pop(k, None)


# -----------------------------------------------------------------------------
# Core row-wise apply functions
# -----------------------------------------------------------------------------


def _embed_apply(
    embed_function: Callable[..., Atoms],
    incols: tuple[str, ...],
    outcol: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Row-wise embedding task.

    - Inputs are taken from `incols` and passed positionally to `embed_function`.
    - Output `Atoms` is written to `outcol` as an extxyz string.
    """

    _require_columns(df, incols)
    _ensure_columns(df, (outcol,))

    task_name = outcol
    status_col = f"{task_name}_status"
    error_col = f"{task_name}_error"
    walltime_col = f"{task_name}_walltime"
    _ensure_columns(df, (status_col, error_col, walltime_col))

    def one_row(row: pd.Series) -> pd.Series:
        rid = _row_id(row)

        if not _is_unset(row.get(status_col)):
            logger.info(f"[{task_name}] skipping {rid}: status already set (rerun)")
            return row

        args = [row.get(c) for c in incols]
        if any(_is_bad_input(v) for v in args):
            logger.warning(f"[{task_name}] missing input for {rid}; marking failed")
            row[outcol] = pd.NA
            row[status_col] = "missing_input"
            row[error_col] = "missing_input"
            row[walltime_col] = None
            return row

        try:
            with observer.timer(task_name) as t:
                atoms = embed_function(*args)
            row[walltime_col] = t.accum

            if not isinstance(atoms, Atoms):
                raise TypeError(
                    f"embed_function must return ase.Atoms (got {type(atoms).__name__})"
                )

            row[outcol] = _write_extxyz(atoms)
            row[status_col] = "ok"
            row[error_col] = None
            return row

        except Exception as exc:
            logger.exception(f"[{task_name}] embed failed for {rid}: {type(exc).__name__}: {exc}")
            row[outcol] = pd.NA
            row[status_col] = "error"
            row[error_col] = f"{type(exc).__name__}: {exc}"
            row[walltime_col] = None
            return row

    return df.apply(one_row, axis=1)


def _ase_apply(
    action: Action,
    incol: str,
    outcols: tuple[str, ...],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Row-wise ASE action task."""

    _require_columns(df, (incol,))
    _ensure_columns(df, outcols)

    task_name = (outcols[0] if len(outcols) == 1 else "__".join(outcols)) or _action_label(action)
    status_col = f"{task_name}_status"
    error_col = f"{task_name}_error"
    walltime_col = f"{task_name}_walltime"
    _ensure_columns(df, (status_col, error_col, walltime_col))

    def one_row(row: pd.Series) -> pd.Series:
        rid = _row_id(row)

        if not _is_unset(row.get(status_col)):
            logger.info(f"[{task_name}] skipping {rid}: status already set (rerun)")
            return row

        raw_geom = row.get(incol)
        if _is_unset(raw_geom):
            logger.warning(f"[{task_name}] missing input geometry for {rid}; skipping")
            _assign_failure(row, outcols)
            row[status_col] = "missing_input"
            row[error_col] = "missing_input"
            row[walltime_col] = None
            return row

        try:
            atoms = _read_extxyz(str(raw_geom))
        except Exception as exc:
            logger.exception(
                f"[{task_name}] failed to parse extxyz for {rid}: {type(exc).__name__}: {exc}"
            )
            _assign_failure(row, outcols)
            row[status_col] = "error"
            row[error_col] = f"{type(exc).__name__}: {exc}"
            row[walltime_col] = None
            return row

        atoms.info.setdefault("name", rid)
        label = str(atoms.info.get("name", rid))

        try:
            with observer.timer(task_name) as t:
                # Save the names of the input and output columns to atoms.info
                colvals = {'_geomscreen_incol': incol, '_geomscreen_outcols': outcols}
                result, atoms_after = _run_action(action, atoms, colvals)
            row[walltime_col] = t.accum
        except Exception as exc:
            logger.exception(f"[{task_name}] action failed for {label}: {type(exc).__name__}: {exc}")
            _assign_failure(row, outcols)
            row[status_col] = "error"
            row[error_col] = f"{type(exc).__name__}: {exc}"
            row[walltime_col] = None
            return row

        # No outputs requested: still record completion.
        if len(outcols) == 0:
            row[status_col] = "ok"
            row[error_col] = None
            return row

        # Multi-output: tuple return must align to outcols.
        if isinstance(result, tuple):
            if len(result) != len(outcols):
                raise ValueError(
                    f"action returned {len(result)} outputs but outcol has {len(outcols)} columns"
                )

            for c, item in zip(outcols, result):
                if isinstance(item, Atoms):
                    row[c] = _write_extxyz(item)
                else:
                    row[c] = item if item is not None else pd.NA

            row[status_col] = "partial" if any(item is None for item in result) else "ok"
            row[error_col] = None
            return row

        # Single-output: must have exactly one outcol.
        if len(outcols) != 1:
            raise TypeError(
                f"action returned a single value but outcol has {len(outcols)} columns; "
                "return a tuple to fill multiple outcols"
            )

        out = outcols[0]
        if isinstance(result, Atoms):
            row[out] = _write_extxyz(result)
        elif result is None:
            row[out] = _write_extxyz(atoms_after)
        else:
            row[out] = result

        row[status_col] = "ok"
        row[error_col] = None
        return row

    return df.apply(one_row, axis=1)


def _fairchem_apply(
    action: BatchAction,
    incol: str,
    outcols: tuple[str, ...],
    server: str,
    client_max_workers: int | None,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run a FAIRChem-batched action over a dataframe chunk.

    This is analogous to `_ase_apply`, but executes rows concurrently using a
    local threadpool while forwarding inference to a *separately started*
    Ray Serve batching server.

    Notes
    -----
    - The Ray Serve app must already be running (see `start_fairchem_batch_server`).
    - This task should not request GPUs (use GPUs only for the Serve deployment).
    """

    _require_columns(df, (incol,))
    _ensure_columns(df, outcols)

    task_name = (outcols[0] if len(outcols) == 1 else "__".join(outcols)) or _action_label(action)
    status_col = f"{task_name}_status"
    error_col = f"{task_name}_error"
    walltime_col = f"{task_name}_walltime"
    _ensure_columns(df, (status_col, error_col, walltime_col))

    # Fast path: nothing to do.
    todo = df[status_col].apply(_is_unset)
    if not bool(todo.any()):
        return df

    # Mark missing-input rows (only among those not already completed).
    missing_geom = todo & df[incol].apply(_is_unset)
    if bool(missing_geom.any()):
        idx = df.index[missing_geom]
        if len(outcols) > 0:
            df.loc[idx, list(outcols)] = pd.NA
        df.loc[idx, status_col] = "missing_input"
        df.loc[idx, error_col] = "missing_input"
        df.loc[idx, walltime_col] = None

    # Rows that need actual computation.
    work = todo & ~df[incol].apply(_is_unset)
    if not bool(work.any()):
        return df

    # Connect to an already-running Serve app and build a tiny omol-focused
    # proxy for FAIRChemCalculator. Inference still runs through the Serve
    # replica; the local validation hook is intentionally a no-op.
    predict_unit = _ServePredictorProxy(serve.get_app_handle(server))

    max_workers = client_max_workers if client_max_workers is not None else min(cpu_count(), 16)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    colvals = {"_geomscreen_incol": incol, "_geomscreen_outcols": outcols}

    def run_one(idx_and_geom: tuple[object, object]) -> tuple[object, dict[str, object]]:
        idx, raw_geom = idx_and_geom
        rid = str(idx)

        try:
            atoms = _read_extxyz(str(raw_geom))
        except Exception as exc:
            logger.exception(
                f"[{task_name}] failed to parse extxyz for {rid}: {type(exc).__name__}: {exc}"
            )
            out: dict[str, object] = {c: pd.NA for c in outcols}
            out.update({status_col: "error", error_col: f"{type(exc).__name__}: {exc}", walltime_col: None})
            return idx, out

        atoms.info.setdefault("name", rid)
        label = str(atoms.info.get("name", rid))

        try:
            with observer.timer(task_name) as t:
                result, atoms_after = _run_batch_action(action, atoms, predict_unit, colvals)
            walltime = t.accum
        except Exception as exc:
            logger.exception(f"[{task_name}] action failed for {label}: {type(exc).__name__}: {exc}")
            out = {c: pd.NA for c in outcols}
            out.update({status_col: "error", error_col: f"{type(exc).__name__}: {exc}", walltime_col: None})
            return idx, out

        out: dict[str, object] = {walltime_col: walltime}

        # No outputs requested: still record completion.
        if len(outcols) == 0:
            out.update({status_col: "ok", error_col: None})
            return idx, out

        # Multi-output: tuple return must align to outcols.
        if isinstance(result, tuple):
            if len(result) != len(outcols):
                raise ValueError(
                    f"action returned {len(result)} outputs but outcol has {len(outcols)} columns"
                )

            for c, item in zip(outcols, result):
                if isinstance(item, Atoms):
                    out[c] = _write_extxyz(item)
                else:
                    out[c] = item if item is not None else pd.NA

            out[status_col] = "partial" if any(item is None for item in result) else "ok"
            out[error_col] = None
            return idx, out

        # Single-output: must have exactly one outcol.
        if len(outcols) != 1:
            raise TypeError(
                f"action returned a single value but outcol has {len(outcols)} columns; "
                "return a tuple to fill multiple outcols"
            )

        outcol = outcols[0]
        if isinstance(result, Atoms):
            out[outcol] = _write_extxyz(result)
        elif result is None:
            out[outcol] = _write_extxyz(atoms_after)
        else:
            out[outcol] = result

        out[status_col] = "ok"
        out[error_col] = None
        return idx, out

    items = [(idx, df.at[idx, incol]) for idx in df.index[work]]
    try:
        for idx, updates in executor.map(run_one, items):
            for col, val in updates.items():
                df.at[idx, col] = val
    finally:
        executor.shutdown(wait=True)

    return df


def _filter_apply(
    incols: tuple[str, ...],
    predicate: Callable[..., bool],
    filter_in: bool,
    df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(df, incols)

    def row_pred(row: pd.Series) -> bool:
        args = [row.get(c) for c in incols]
        if any(_is_bad_input(v) for v in args):
            return False
        return bool(predicate(*args))

    mask = df.apply(row_pred, axis=1)
    return df[mask] if filter_in else df[~mask]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def ase_task(
    action: Action,
    incol: ColName,
    outcol: ColNames,
    **task_kwargs: Any,
) -> PipelineTask:
    """Create a `PipelineTask` that runs an ASE action on a geometry column.

    This factory wraps one “row-wise” ASE computation behind the dplutils
    `PipelineTask` interface. It handles:

    - Reading input geometries from `extxyz` strings stored in `df[incol]`
    - Running an action on an `ase.Atoms` instance
    - Writing outputs back into one or more dataframe columns
    - Per-row logging, walltime measurement, and robust reruns/resume via a
      status column

    Parameters
    ----------
    action
        Either:
        - an `ActionFunc`: `Callable[[Atoms], ActionResult]`, or
        - an `ActionSeq`: a tuple of functions executed in order.

        **Action sequences**
        - Intermediate steps (all but the last) must return `Atoms` or `None`.
          - Returning `Atoms` replaces the working geometry for subsequent steps.
          - Returning `None` means “mutated the input `Atoms` in-place”.
        - The final step returns an `ActionResult`.

        **None semantics**
        - If `outcol` is a single column and the final action returns `None`,
          it is interpreted as “mutated in place”; the mutated geometry is written
          to `outcol` as extxyz.
        - If `outcol` is multiple columns (a tuple) and the final action returns
          a tuple, any `None` *inside that tuple* means “completed, but no value
          for this output”. The row is still marked complete and won’t rerun.

    incol
        Name of the input geometry column containing a single-frame extxyz string
        per row.
    outcol
        Output column name(s).
        - If a string: the action is treated as single-output.
        - If a tuple: the action must return a tuple of the same length.

        `Atoms` outputs are **always** serialized to extxyz strings whenever they
        appear, regardless of the column name.

        An empty tuple `outcol=()` is allowed: the action runs for side effects
        and the row is still marked complete via the status column.
    **task_kwargs
        Forwarded directly to `PipelineTask(...)`. Use this to configure runtime
        behavior, e.g.:

        - `num_cpus=...`, `num_gpus=...`
        - `batch_size=...`
        - `resources={...}`
        - `kwargs={...}`, `context_kwargs={...}` (if you want the executor to pass
          extra keyword args into the underlying task function)

    Outputs and reruns
    ------------------
    In addition to `outcol`, this task writes three bookkeeping columns based on
    the derived task name (typically `outcol` or `"__".join(outcol)`):

    - `<task>_status`: authoritative completion marker (`ok`, `partial`, `error`, …)
    - `<task>_error`: stringified exception on failure
    - `<task>_walltime`: elapsed time accumulated by `dplutils.observer.timer`

    A row is considered “already done” if `<task>_status` is non-empty.
    To force a rerun, clear that status cell (and any outputs you want recomputed).

    Returns
    -------
    PipelineTask
        A configured `PipelineTask` ready to be included in a `PipelineGraph`.

    Examples
    --------
    Geometry optimization (in-place optimizer returning `None`):

    >>> ase_task(optimize_geometry, "initial_geom", "optimized_geom")

    Setup + optimize via action sequence:

    >>> ase_task((triplet_setup, optimize_geometry), "initial_geom", "triplet_geom")

    Multi-output (e.g., optimized geometry and property):

    >>> def opt_and_energy(atoms): ...
    >>> ase_task(opt_and_energy, "geom_in", ("geom_out", "energy"))
    """
    outcols = _as_cols(outcol)
    name = (outcols[0] if len(outcols) == 1 else "__".join(outcols)) or _action_label(action)
    func = partial(_ase_apply, action, incol, outcols)
    return PipelineTask(name, func, **task_kwargs)

def start_fairchem_batch_server(
    predict_unit: MLIPPredictUnit,
    *,
    server: str = "predict-server",
    max_batch_size: int = 512,
    batch_wait_timeout_s: float = 0.1,
    split_oom_batch: bool = True,
    num_replicas: int = 1,
    serve_cpus: int | None = 8,
    serve_gpus: int | None = 1,
    route_prefix: str = "/predict",
):
    """Start a Ray Serve batching server for FAIRChem inference.

    This should be called once in the *driver* process, right after `ray.init()`
    and before the pipeline starts submitting tasks.

    Notes
    -----
    - The GPU(s) should be reserved by this Serve deployment (via `serve_gpus`).
      The individual `fairchem_task` pipeline tasks should use `num_gpus=0`.
    - `serve_cpus` is the CPU reservation per Serve replica (FAIRChem defaults to 8).
    """
    ray_actor_options: dict[str, float] = {}
    if serve_cpus is not None:
        ray_actor_options["num_cpus"] = float(serve_cpus)
    if serve_gpus is not None:
        ray_actor_options["num_gpus"] = float(serve_gpus)

    return setup_batch_predict_server(
        predict_unit=predict_unit,
        max_batch_size=max_batch_size,
        batch_wait_timeout_s=batch_wait_timeout_s,
        split_oom_batch=split_oom_batch,
        num_replicas=num_replicas,
        ray_actor_options=ray_actor_options,
        deployment_name=server,
        route_prefix=route_prefix,
    )


def fairchem_task(
    action: BatchAction,
    incol: ColName,
    outcol: ColNames,
    *,
    server: str = "predict-server",
    client_max_workers: int | None = None,
    **task_kwargs: Any,
) -> PipelineTask:
    """Create a `PipelineTask` that runs a FAIRChem-batched ASE workflow.

    Expected usage
    --------------
    1) Start a Ray Serve batching server once (driver-side) with
       :func:`start_fairchem_batch_server`.
    2) Use :func:`fairchem_task` in your pipeline. Each task:
       - connects to the existing Serve app (`server=...`)
       - builds a tiny omol-focused proxy around the Serve handle
       - avoids fetching raw FAIRChem `dataset_to_tasks` objects from Serve
       - uses a no-op local `validate_atoms_data(...)`
       - runs rows concurrently in a local threadpool to feed the batch server

    Parameters
    ----------
    action
        Either:

        - a callable of the form ``(atoms, predict_unit) -> ActionResult``, or
        - an action sequence (a tuple of callables).

        In a sequence, the first step is a FAIRChem-specific setup step:

        - ``setup(atoms, predict_unit) -> Atoms | None``

        It typically attaches a :class:`~fairchem.core.calculate.FAIRChemCalculator`
        built from the provided ``predict_unit``. Subsequent steps are standard
        geomscreen/ASE actions that accept only ``Atoms``.

    incol
        Input geometry column (single-frame extxyz strings).
    outcol
        Output column name(s). Semantics are identical to :func:`ase_task`.
    server
        Ray Serve application name (passed to `serve.get_app_handle(server)`).
    client_max_workers
        Number of threads used to run rows concurrently inside the worker process.
        If None, defaults to `min(os.cpu_count(), 16)`.
    **task_kwargs
        Forwarded to :class:`dplutils.pipeline.PipelineTask`.

    Notes
    -----
    `fairchem_task` tasks must not request GPUs. GPUs are reserved for the Serve
    deployment created by :func:`start_fairchem_batch_server`.

    The local predictor proxy is intentionally omol-specific and does not
    perform FAIRChem validation. If your workflow depends on model-specific
    `validate_atoms_data(...)` behavior, or on non-omol task metadata, set the
    required `atoms.info` fields yourself and revisit this proxy.
    """

    if float(task_kwargs.get("num_gpus", 0) or 0) != 0:
        raise ValueError(
            "fairchem_task pipeline tasks must use num_gpus=0. "
            "Reserve GPUs for the batch server started by start_fairchem_batch_server()."
        )
    task_kwargs["num_gpus"] = 0

    outcols = _as_cols(outcol)
    name = (outcols[0] if len(outcols) == 1 else "__".join(outcols)) or _action_label(action)
    func = partial(_fairchem_apply, action, incol, outcols, server, client_max_workers)
    return PipelineTask(name, func, **task_kwargs)


def filter_task(
    predicate: Callable[..., bool],
    incol: ColNames,
    filter_in: bool = True,
    **task_kwargs: Any,
) -> PipelineTask:
    """Create a `PipelineTask` that filters rows by a boolean predicate.

    This task *removes* rows from the dataframe chunk; it does not add columns.

    Parameters
    ----------
    predicate
        Callable returning `True`/`False`. It is called positionally with the
        values of `incol` in order. For example:

        - `incol="energy"` -> `predicate(row["energy"])`
        - `incol=("energy", "gap")` -> `predicate(row["energy"], row["gap"])`

        Missing values are treated as predicate=False:
        `None`, `NaN`/`pd.NA`-like values, and empty strings, all cause the
        predicate to be skipped and the row to be treated as not passing.
    incol
        Input column name(s) whose values are passed to `predicate`.
    filter_in
        If True, keep rows where `predicate(...)` is True.
        If False, keep rows where `predicate(...)` is False.
    **task_kwargs
        Forwarded to `PipelineTask(...)` (e.g., `batch_size`, `num_cpus`, etc.).

    Notes
    -----
    Exceptions from `predicate` are intentionally not caught. A crashing predicate
    is usually a configuration/programming error (wrong columns, wrong types, etc.)
    and should be surfaced quickly.

    Returns
    -------
    PipelineTask
        A filtering task suitable for inclusion in a `PipelineGraph`.

    Examples
    --------
    Keep rows with blue phosphorescence (can be simplified with threshold_task):

    >>> filter_task(lambda e: e > 2.6, "triplet_minus_ground")

    Or, if the energies have not been subtracted yet:

    >>> filter_task(lambda t, g: (t - g) > 2.6, ("E_triplet", "E_ground"))
    """
    incols = _as_cols(incol)
    base = "__".join(incols) if incols else "filter"
    name = f"{base}_{'in' if filter_in else 'out'}"
    return PipelineTask(name, partial(_filter_apply, incols, predicate, filter_in), **task_kwargs)


def threshold_task(
    incol: ColNames,
    threshold: float,
    keep_lower_than: bool,
    **task_kwargs: Any,
) -> PipelineTask:
    """Convenience wrapper around `filter_task` for numeric thresholds.

    This is defined *in terms of* `filter_task`: it builds a predicate
    `x < threshold`, then chooses whether to keep matching rows or the complement.

    Parameters
    ----------
    incol
        Exactly one input column. (If you pass multiple columns, this raises.)
    threshold
        Threshold value to compare against.
    keep_lower_than
        - True: keep rows where `value < threshold`
        - False: keep rows where `value >= threshold`
    **task_kwargs
        Forwarded to `PipelineTask(...)`.

    Returns
    -------
    PipelineTask
        A filtering task whose name includes the threshold.

    Notes
    -----
    The generated predicate uses the raw column value. If values are not
    comparable to `threshold`, the predicate will raise (and the pipeline will
    fail), which is usually what you want for misconfigured inputs.

    Examples
    --------
    Keep rows with blue phosphorescence:

    >>> threshold_task("triplet_minus_ground", 2.6, keep_lower_than=False)
    """
    incols = _as_cols(incol)
    if len(incols) != 1:
        raise ValueError("threshold_task expects exactly one input column")

    def pred(x: object) -> bool:
        return cast(TableValue, x) < threshold

    # Build via filter_task, then rename to a more informative name
    base_task = filter_task(pred, incol, filter_in=keep_lower_than, **task_kwargs)

    col = incols[0]
    op = "lt" if keep_lower_than else "ge"
    thr = str(threshold).replace("-", "m").replace(".", "p")
    return base_task(name=f"{col}_{op}_{thr}")


def embed_task(
    embed_function: Callable[..., Atoms],
    incol: ColNames,
    outcol: ColName,
    **task_kwargs: Any,
) -> PipelineTask:
    """Create a `PipelineTask` that constructs an ASE `Atoms` geometry from columns.

    This is meant for “embedding” or structure-construction steps (e.g., RDKit
    embedding, file loading, etc.). It:

    - Calls `embed_function(*row[incol...])`
    - Requires an `ase.Atoms` return value (raise exceptions for true failure)
    - Serializes the result into an extxyz string stored in `df[outcol]`
    - Adds per-row `<outcol>_status`, `<outcol>_error`, and `<outcol>_walltime`

    Parameters
    ----------
    embed_function
        Callable returning an `ase.Atoms`. If embedding fails, it should raise an
        exception. (This wrapper catches per-row exceptions, logs them, and marks
        the row failed.)
    incol
        One or more input columns whose values are passed positionally into
        `embed_function`.
    outcol
        Output geometry column that will store the extxyz string.
    **task_kwargs
        Forwarded to `PipelineTask(...)`.

    Reruns
    ------
    This task uses `<outcol>_status` as the authoritative “done” marker. If
    `<outcol>_status` is non-empty, that row is skipped on rerun. To retry a row,
    clear the status cell.

    Returns
    -------
    PipelineTask
        An embedding task suitable for inclusion in a `PipelineGraph`.

    Examples
    --------
    Retrieve a geometry from ASE's built-in molecule database:

    >>> from ase.build import molecule
    >>> embed_task(molecule, "formula", "initial_geom")

    """
    incols = _as_cols(incol)
    func = partial(_embed_apply, embed_function, incols, outcol)
    return PipelineTask(outcol, func, **task_kwargs)

__all__ = [
    "ase_task",
    "fairchem_task",
    "embed_task",
    "filter_task",
    "threshold_task",
]
