from functools import lru_cache
from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit
from fairchem.core.calculate import InferenceBatcher

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

def optimize_geometry(atoms: Atoms) -> None:
    opt = BFGS(atoms, logfile=None, trajectory=None)
    opt.run(fmax=0.02)

def energy(atoms: Atoms) -> float:
    energy = atoms.get_potential_energy()
    return float(energy)
