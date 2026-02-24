from functools import partial
import pandas as pd
import ray
from dplutils.pipeline import PipelineGraph
from dplutils.cli import cli_run
from dplutils.pipeline.ray import RayStreamGraphExecutor
from geomscreen import fairchem_task, embed_task
from helpers import get_batcher, setup, embed_smiles, optimize_geometry, energy

ground_setup = partial(setup, 1)
triplet_setup = partial(setup, 3)
ray.init(address="local", num_cpus=24, num_gpus=1, include_dashboard=False)

graph = PipelineGraph([
    embed_task(embed_smiles, "smiles", "initial_geom"),
    fairchem_task((triplet_setup, optimize_geometry), "initial_geom", "triplet_geom", batcher=get_batcher, num_cpus=8, num_gpus=1),
    fairchem_task((ground_setup, energy), "triplet_geom", "ground_energy", batcher=get_batcher, num_cpus=8, num_gpus=1),
    fairchem_task((triplet_setup, energy), "triplet_geom", "triplet_energy", batcher=get_batcher, num_cpus=8, num_gpus=1),
    ])

executor = RayStreamGraphExecutor(graph,
        generator=lambda: pd.read_csv("test_data/organic_phos_smiles_energy.csv", chunksize=200),
)

cli_run(executor)
