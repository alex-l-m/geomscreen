import pandas as pd
import ray
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from dplutils.pipeline import PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import ase_task, embed_task

def rdkit_embed(smiles: str) -> Atoms:
    from rdkit.Chem.rdmolfiles import MolFromSmiles
    from rdkit.Chem.rdmolops import AddHs
    from rdkit.Chem.rdDistGeom import EmbedMolecule
    rdkit_mol_nohs = MolFromSmiles(smiles)
    assert rdkit_mol_nohs is not None
    rdkit_mol = AddHs(rdkit_mol_nohs)
    assert EmbedMolecule(rdkit_mol) == 0
    pos = rdkit_mol.GetConformer().GetPositions()
    elem = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    ase_mol = Atoms(positions=pos, symbols=elem)
    return ase_mol

def optimize_geometry(atoms: Atoms) -> None:
    atoms.calc = TBLite()
    opt = BFGS(atoms, logfile=None, trajectory=None)
    opt.run(fmax=0.02)

def energy(atoms: Atoms) -> float:
    atoms.calc = TBLite()
    energy = atoms.get_potential_energy()
    return float(energy)

ray.init()

graph = PipelineGraph([
    embed_task(rdkit_embed, "smiles", "initial_geom"),
    ase_task(optimize_geometry, "initial_geom", "optimized_geom"),
    ase_task(energy, "optimized_geom", "energy")
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/cycloalkanes.csv", chunksize=200),
)

cli_run(executor)
