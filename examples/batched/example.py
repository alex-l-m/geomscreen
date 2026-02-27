from functools import partial

import pandas as pd
import ray
from fairchem.core import pretrained_mlip

from dplutils.cli import cli_run
from dplutils.pipeline import PipelineGraph
from dplutils.pipeline.ray import RayStreamGraphExecutor

from geomscreen import embed_task, fairchem_task, start_fairchem_batch_server
from helpers import (
    get_validator_predict_unit,
    setup,
    embed_smiles,
    optimize_geometry,
    energy,
)

ground_setup = partial(setup, 1)
triplet_setup = partial(setup, 3)

ray.init(address="local", num_cpus=24, num_gpus=1, include_dashboard=False)

# Start the FAIRChem Ray Serve batch server once (reserves the GPU).
gpu_predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
start_fairchem_batch_server(
    gpu_predict_unit,
    server="predict-server",
    serve_cpus=8,
    serve_gpus=1,
    max_batch_size=512,
    batch_wait_timeout_s=0.1,
)

graph = PipelineGraph(
    [
        embed_task(embed_smiles, "smiles", "initial_geom"),
        fairchem_task(
            (triplet_setup, optimize_geometry),
            "initial_geom",
            "triplet_geom",
            validator=get_validator_predict_unit,
            server="predict-server",
            num_cpus=1,
        ),
        fairchem_task(
            (ground_setup, energy),
            "triplet_geom",
            "ground_energy",
            validator=get_validator_predict_unit,
            server="predict-server",
            num_cpus=1,
        ),
        fairchem_task(
            (triplet_setup, energy),
            "triplet_geom",
            "triplet_energy",
            validator=get_validator_predict_unit,
            server="predict-server",
            num_cpus=1,
        ),
    ]
)

executor = RayStreamGraphExecutor(
    graph,
    generator=lambda: pd.read_csv("test_data/organic_phos_smiles_energy.csv", chunksize=200),
)

cli_run(executor)
