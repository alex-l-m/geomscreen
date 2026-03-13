from functools import partial
import pandas as pd
import ray
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from dplutils.pipeline import PipelineTask, PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import ase_task, embed_task, ge_task, lt_task

def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(rank=df["gap"].rank(ascending=False))
rank_task = PipelineTask("rank", add_rank)

def add_difference(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(gap=df["triplet_energy"] - df["ground_energy"])
difference_task = PipelineTask("gap", add_difference)

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

def setup(multiplicity: int, atoms: Atoms) -> None:
    atoms.calc = TBLite(multiplicity=multiplicity)

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
    ase_task((triplet_setup, optimize_geometry), "initial_geom", "triplet_geom"),
    ase_task((ground_setup, energy), "triplet_geom", "ground_energy"),
    ase_task((triplet_setup, energy), "triplet_geom", "triplet_energy"),
    difference_task,
    rank_task,
    lt_task("rank", 3.5)
    ])

graph.add_edge(rank_task, ge_task("rank", 3.5))

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/organic_phos_smiles_energy.csv", chunksize=200),
)

cli_run(executor)
