from functools import partial
import pandas as pd
import ray
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from dplutils.pipeline import PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import ase_task, embed_task

def read_mol2(inpath: str) -> Atoms:
    from rdkit.Chem.rdmolfiles import MolFromMol2File
    rdkit_mol = MolFromMol2File(inpath)
    assert rdkit_mol is not None
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
    embed_task(read_mol2, 'structure_path', 'initial_geom'),
    ase_task((triplet_setup, optimize_geometry), 'initial_geom', 'triplet_geom'),
    ase_task((ground_setup, energy), 'triplet_geom', 'ground_energy'),
    ase_task((triplet_setup, energy), 'triplet_geom', 'triplet_energy')
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/irppy3.csv", chunksize=200),
)

cli_run(executor)
