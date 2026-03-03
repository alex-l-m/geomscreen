from ase import Atoms
from ase.optimize import BFGS
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs


def setup(multiplicity: int, atoms: Atoms, predictor) -> None:
    from fairchem.core.calculate import FAIRChemCalculator

    # geomscreen's local predictor proxy does not populate OMOL defaults.
    # Set charge and multiplicity before attaching the calculator.
    atoms.info["charge"] = 0
    atoms.info["spin"] = multiplicity
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")


def embed_smiles(smiles: str) -> Atoms:
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
