

Library for virtual screening of molecules using geometry-based models.

Models are run using Atomic Simulation Environment.

Screening pipelines are constructed and run using dplutils.

Basically, this is a library of dplutils tasks, including tasks for
running models with ASE and tasks for filtering the results.

# Example: Strain Energy

Beginning with a simple example that requires embedding a SMILES string
as a geometry, optimizing the geometry, and computing an energy.

Input table `test_data/cycloalkanes.csv`:

    mol_id,n_c,smiles
    cyclohexane,6,C1CCCCC1
    cyclopentane,5,C1CCCC1
    cyclobutane,4,C1CCC1
    cyclopropane,3,C1CC1

Computing the energies of each:

``` python
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
```

The result is a table with an `energy` column:

``` python
import pandas as pd
from glob import glob
results = pd.concat(
    (pd.read_parquet(infile) for infile in glob("*.parquet")),
    ignore_index=True)
print(results)
```

             mol_id  n_c    smiles  \
    0   cyclohexane    6  C1CCCCC1   
    1  cyclopentane    5   C1CCCC1   
    2   cyclobutane    4    C1CCC1   
    3  cyclopropane    3     C1CC1   

                                            initial_geom initial_geom_status  \
    0  18\nProperties=species:S:1:pos:R:3 pbc="F F F"...                  ok   
    1  15\nProperties=species:S:1:pos:R:3 pbc="F F F"...                  ok   
    2  12\nProperties=species:S:1:pos:R:3 pbc="F F F"...                  ok   
    3  9\nProperties=species:S:1:pos:R:3 pbc="F F F"\...                  ok   

      initial_geom_error  initial_geom_walltime  \
    0               None             102.585846   
    1               None               1.303472   
    2               None               0.876995   
    3               None               0.714976   

                                          optimized_geom optimized_geom_status  \
    0  18\nProperties=species:S:1:pos:R:3:energies:R:...                    ok   
    1  15\nProperties=species:S:1:pos:R:3:energies:R:...                    ok   
    2  12\nProperties=species:S:1:pos:R:3:energies:R:...                    ok   
    3  9\nProperties=species:S:1:pos:R:3:energies:R:1...                    ok   

      optimized_geom_error  optimized_geom_walltime      energy energy_status  \
    0                 None               139.447088 -516.376359            ok   
    1                 None                95.621541 -430.329276            ok   
    2                 None                86.991137 -343.507726            ok   
    3                 None                28.659565 -257.098332            ok   

      energy_error  energy_walltime  
    0         None         5.633754  
    1         None         4.007234  
    2         None         2.966991  
    3         None         2.142956  

The energy per carbon shows the trend in strain energy:

``` python
results['energy_per_c'] = results['energy'] / results['n_c']
print(results[['mol_id', 'energy_per_c']])
```

             mol_id  energy_per_c
    0   cyclohexane    -86.062727
    1  cyclopentane    -86.065855
    2   cyclobutane    -85.876932
    3  cyclopropane    -85.699444

# Example: Reaction energy

This example shows what to do when the ASE calculator needs to be
customized to the molecule. We will calculate the combustion energy of
methane, which requires taking into account the triplet state of oxygen.

Input table `test_data/combustion_reactants.csv`:

    mol_id,coef,multiplicity,smiles
    methane,-1,1,C
    oxygen,-2,3,O=O
    carbon_dioxide,1,1,O=C=O
    water,2,1,O

Calculating energies:

``` python
import pandas as pd
import ray
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from dplutils.pipeline import PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import ase_task, embed_task

def embed_with_multiplicity(smiles: str, multiplicity: int) -> Atoms:
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
    ase_mol.info["multiplicity"] = multiplicity
    return ase_mol

def optimize_geometry(atoms: Atoms) -> None:
    multiplicity = atoms.info["multiplicity"]
    atoms.calc = TBLite(multiplicity=multiplicity)
    opt = BFGS(atoms, logfile=None, trajectory=None)
    opt.run(fmax=0.02)

def energy(atoms: Atoms) -> float:
    multiplicity = atoms.info["multiplicity"]
    atoms.calc = TBLite(multiplicity=multiplicity)
    energy = atoms.get_potential_energy()
    return float(energy)

ray.init()

graph = PipelineGraph([
    embed_task(embed_with_multiplicity, ("smiles", "multiplicity"), "initial_geom"),
    ase_task(optimize_geometry, "initial_geom", "optimized_geom"),
    ase_task(energy, "optimized_geom", "energy")
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/combustion_reactants.csv", chunksize=200),
)

cli_run(executor)
```

We see here that embedding can take a tuple of columns, rather than a
single column, and store the information as part of the ASE Atoms
object.

Then, calculating the reaction energy:

``` python
import pandas as pd
from glob import glob
results = pd.concat(
    (pd.read_parquet(infile) for infile in glob("*.parquet")),
    ignore_index=True)
rxn_energy = (results['coef'] * results['energy']).sum()
print(rxn_energy)
```

    -12.682898764726929

This should be the combustion energy of methane.

# Example: Phosphorescence emission energy

Now we will consider an example where it is convenient to split the ASE
action into two parts. We will do energy calculations on the same
molecule, with multiplicities 1 and 3. It is therefore convenient to
define a “setup function” that sets the calculator, and a separate
“action function” that calculates the energy.

We will use [an example
molecule](www.ccdc.cam.ac.uk/structures/Search?Ccdcid=KAVJOH01)
displaying phosphorescence, and use the triplet energy above the ground
state energy to estimate the phosphorescence emission energy. For
pipelines that read from files, usually it is convenient to begin with a
table containing paths, as awkward as it seems when we’re running on a
single example.

    mol_id,structure_path
    irppy3,test_data/KAVJOH01.mol2

Then, the pipeline:

``` python
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
    embed_task(read_mol2, "structure_path", "initial_geom"),
    ase_task((triplet_setup, optimize_geometry), "initial_geom", "triplet_geom"),
    ase_task((ground_setup, energy), "triplet_geom", "ground_energy"),
    ase_task((triplet_setup, energy), "triplet_geom", "triplet_energy")
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/irppy3.csv", chunksize=200),
)

cli_run(executor)
```

This is calculating a vertical excitation energy; that is, both energies
are calculated in the triplet geometry.

Then, taking the difference to obtain an estimate of the phosphorescence
energy:

``` python
import pandas as pd
from glob import glob
results = pd.concat(
    (pd.read_parquet(infile) for infile in glob("*.parquet")),
    ignore_index=True)
print(results['triplet_energy'] - results['ground_energy'])
```

    0    0.872711
    dtype: float64

# Example: Filtering

This next example estimates phosphorescence emission energies for the
organic phosphorescent molecules from [Yuan et al
2010](https://doi.org/10.1021/jp909388y) and filters to make a table of
the three bluest.

``` python
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
```

Looking at the output:

``` python
kept = pd.concat(pd.read_parquet(inpath) for inpath in glob("task=rank_lt_3p5/*.parquet"))
print(kept[["mol_id", "gap", "rank"]])
```

      mol_id       gap  rank
    0     bp  2.652823   3.0
    1   dfbp  2.663454   2.0
    2    mbb  3.009887   1.0

In this case we also saved the molecules that were filtered out:

``` python
discarded = pd.concat(pd.read_parquet(inpath) for inpath in glob("task=rank_ge_3p5/*.parquet"))
print(discarded[["mol_id", "gap", "rank"]])
```

      mol_id       gap  rank
    0   dcbp  2.580694   4.0
    1   dbbp  2.539513   6.0
    2    bbp  2.576435   5.0
    3    abp  2.209851   8.0
    4  dbbp2  2.483134   7.0

This strategy is useful if we want to do another, more expensive
calculation on the top ranked molecules, but we also want to be able to
look at the results of the fast calculation that was used for filtering.

# Example: Batching

The style we have used so far is inefficient when using a GPU. The
FAIRChem library provides an interface to ASE that enables batching for
more efficient GPU usage. To use it, we need a separate task. Running
the same organic phosphors, but batching with FAIRChem.

With these helpers:

``` python
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
```

Running this code:

``` python
import os
from functools import partial

import pandas as pd
import ray
from fairchem.core import pretrained_mlip

from dplutils.cli import cli_run
from dplutils.pipeline import PipelineGraph
from dplutils.pipeline.ray import RayStreamGraphExecutor

from geomscreen import embed_task, fairchem_task, start_fairchem_batch_server
from helpers import (
    setup,
    embed_smiles,
    optimize_geometry,
    energy,
)

# Ensure 0-GPU Ray tasks still see CUDA
# This will be the default behavior in future ray versions
# Even though the dplutils task doesn't use a GPU to run the model, it still
# needs the GPU available to retrieve the results
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

ground_setup = partial(setup, 1)
triplet_setup = partial(setup, 3)

ray.init(address="local", num_cpus=24, num_gpus=1, include_dashboard=False)

# Start the FAIRChem Ray Serve batch server once (reserves the GPU).
gpu_predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
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
            server="predict-server",
            num_cpus=1,
        ),
        fairchem_task(
            (ground_setup, energy),
            "triplet_geom",
            "ground_energy",
            server="predict-server",
            num_cpus=1,
        ),
        fairchem_task(
            (triplet_setup, energy),
            "triplet_geom",
            "triplet_energy",
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
```

``` python
import pandas as pd
from glob import glob
results = pd.concat(
    (pd.read_parquet(infile, columns=["mol_id", "ground_energy", "triplet_energy"])
    for infile in glob("*.parquet")),
    ignore_index=True)
results["gap"] = results["triplet_energy"] - results["ground_energy"]
gap_only = results[["mol_id", "gap"]]
print(gap_only)
```

      mol_id       gap
    0     bp  2.494904
    1   dfbp  2.525963
    2   dcbp  2.514308
    3   dbbp  2.566979
    4    bbp  2.558657
    5    abp  1.136663
    6    mbb  3.042616
    7  dbbp2  2.409539
