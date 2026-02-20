from functools import partial, lru_cache
import pandas as pd
import ray
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit
from fairchem.core.calculate import InferenceBatcher
from dplutils.pipeline import PipelineTask, PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import fairchem_task, embed_task

@lru_cache(maxsize=1)
def get_batcher() -> InferenceBatcher:
    from fairchem.core.calculate import InferenceBatcher

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    batcher = InferenceBatcher(predictor, concurrency_backend_options={'max_workers': 8})
    return batcher

def setup(multiplicity: int, atoms: Atoms, predictor: BatchServerPredictUnit) -> None:
    from fairchem.core.calculate import FAIRChemCalculator
    # Set charge and multiplicity
    atoms.info["charge"] = 0
    atoms.info["spin"] = multiplicity
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

def embed_smiles(smiles: str) -> Atoms:
    from rdkit.Chem.rdDistGeom import EmbedMolecule
    from rdkit.Chem.rdmolfiles import MolFromSmiles
    from rdkit.Chem.rdmolops import AddHs
    rdkit_mol = MolFromSmiles(smiles)
    rdkit_mol = AddHs(rdkit_mol)
    EmbedMolecule(rdkit_mol)
    pos = rdkit_mol.GetConformer().GetPositions()
    elem = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    ase_mol = Atoms(positions=pos, symbols=elem)
    return ase_mol

ground_setup = partial(setup, 1)
triplet_setup = partial(setup, 3)

def optimize_geometry(atoms: Atoms) -> None:
    opt = BFGS(atoms, logfile=None, trajectory=None)
    opt.run(fmax=0.02)

def energy(atoms: Atoms) -> float:
    energy = atoms.get_potential_energy()
    return float(energy)

ray.init()

graph = PipelineGraph([
    embed_task(embed_smiles, "smiles", "initial_geom"),
    fairchem_task((triplet_setup, optimize_geometry), "initial_geom", "triplet_geom", batcher=get_batcher, num_cpus=8),
    fairchem_task((ground_setup, energy), "triplet_geom", "ground_energy", batcher=get_batcher, num_cpus=8),
    fairchem_task((triplet_setup, energy), "triplet_geom", "triplet_energy", batcher=get_batcher, num_cpus=8),
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/organic_phos_smiles_energy.csv", chunksize=200),
)

cli_run(executor)
