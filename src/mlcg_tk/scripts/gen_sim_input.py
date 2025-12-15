import os.path as osp
import sys

from mlcg_tk.input_generator.raw_dataset import SampleCollection, RawDataset, SimInput
from mlcg_tk.input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from mlcg_tk.input_generator.raw_data_loader import DatasetLoader, SimInput_loader
from mlcg_tk.input_generator.prior_gen import Bonds, PriorBuilder
from mlcg_tk.input_generator.utils import get_output_tag
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
import pickle as pck

import numpy as np

from mlcg.data import AtomicData
import torch
from copy import deepcopy


def process_sim_input(
    dataset_name: str,
    raw_data_dir: str,
    save_dir: str,
    tag: str,
    pdb_fns: List[str],
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues: List[str],
    copies: int,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    mass_scale: Optional[float] = 418.4,
):
    """
    Generates input AtomicData objects for coarse-grained simulations

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    raw_data_dir : str
        Path to location of input structures
    save_dir : str
        Path to directory in which output will be saved
    tag : str
        Label given to all output files produced from dataset
    pdb_fns : str
        List of pdb filenames from which input will be generated
    cg_atoms : List[str]
        List of atom names to preserve in coarse-grained resolution
    embedding_map : CGEmbeddingMap
        Mapping object
    embedding_func : Callable
        Function which will be used to apply CG mapping
    skip_residues : List[str]
        List of residues to skip, can be None
    copies : int
        Copies that will be produced of each structure listing in pdb_fns
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters
    mass_scale : str
        Optional scaling factor applied to atomic masses
    """
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []

    dataset = SimInput(dataset_name, tag, pdb_fns)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = SimInput_loader()
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            name=samples.name, raw_data_dir=raw_data_dir
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        cg_trajs = samples.input_traj.atom_slice(samples.cg_atom_indices)
        cg_masses = (
            np.array([atom.element.mass for atom in cg_trajs[0].topology.atoms])
            / mass_scale
        )
        prior_nls = samples.get_prior_nls(
            prior_builders=prior_builders,
            save_nls=False,
            save_dir=save_dir,
            prior_tag=prior_tag,
        )
        cg_types = samples.cg_dataframe["type"].to_list()
        for i in range(cg_trajs.n_frames):
            cg_traj = cg_trajs[i]
            cg_coords = cg_traj.xyz * 10
            for i in range(copies):
                cg_coord_list.append(cg_coords)
                cg_type_list.append(cg_types)
                cg_mass_list.append(cg_masses)
                cg_nls_list.append(prior_nls)

    data_list = []
    for coords, types, masses, nls in zip(
        cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list
    ):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[0]),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
        )
        data.neighbor_list = deepcopy(nls)
        data_list.append(data)

    torch.save(
        data_list,
        f"{save_dir}{get_output_tag([dataset_name, tag], placement='before')}configurations.pt",
    )

def main():
    print("Start gen_sim_input.py: {}".format(ctime()))
    CLI([process_sim_input])
    print("Finish gen_sim_input.py: {}".format(ctime()))



if __name__ == "__main__":
    main()