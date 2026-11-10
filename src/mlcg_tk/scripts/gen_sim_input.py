import os.path as osp
import sys

from mlcg_tk.input_generator.raw_dataset import SampleCollection, RawDataset, SimInput
from mlcg_tk.input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from mlcg_tk.input_generator.raw_data_loader import DatasetLoader, SimInput_loader
from mlcg_tk.input_generator.prior_gen import Bonds, PriorBuilder
from mlcg_tk.input_generator.utils import get_output_tag, build_exclusion_list
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional, Type
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
    load_nls_from_file: Optional[str] = None,
    mass_scale: Optional[float] = 418.4,
    collection_cls: Type[SampleCollection] = SampleCollection,
    smpl_loader: Type[DatasetLoader] = SimInput_loader,
    excluded_order: Optional[int] = None,
    symmetrize_excluded_pairs: bool = False,
    excluded_prior_nls: Optional[List[str]] = None,
):
    r"""
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
    load_nls_from_file : Optional[str]
        If given, prior nls will be loaded from this file instead of computed. 
    mass_scale : str
        Optional scaling factor applied to atomic masses
    collection_cls : Type[SampleCollection]
        Class type for sample collection
    smpl_loader : Type[DatasetLoader]
        Loader class for dataset
    excluded_order : int
        Maximum graph separation of the excluded pairs:
        2 = bonds
        3 = bonds and angles
        4 = bonds, angles, and dihedrals
    symmetrize_excluded_pairs : bool
        If true, symmetric pairs will be added to excluded pairs list
    excluded_prior_nls : Optional[List[str]]
        List of names of prior neighbour lists to be used to compute excluded pairs. If None, 
        all pairs up to excluded_order bonds apart are computed from the topology and excluded. 
        The highest order of the selected neighbour lists must equal excluded_order.
    """
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []
    cg_excl_list = []

    dataset = SimInput(dataset_name, tag, pdb_fns, collection_cls=collection_cls)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = smpl_loader()
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
        if load_nls_from_file is not None:
            if osp.isfile(load_nls_from_file):
                with open(load_nls_from_file, "rb") as f:
                    prior_nls = pck.load(f)
            else:
                raise FileNotFoundError(f"No prior neighbour lists found at {load_nls_from_file}")
        else:
            prior_nls = samples.get_prior_nls(
                prior_builders=prior_builders,
                save_nls=False,
                save_dir=save_dir,
                prior_tag=prior_tag,
            )
        cg_types = samples.cg_dataframe["type"].to_list()
        excluded_pairs = None
        if excluded_order is not None and excluded_order > 1:
            if excluded_prior_nls is not None:
                missing = [n for n in excluded_prior_nls if n not in prior_nls]
                if missing:
                    raise KeyError(f"{missing} not in prior nls {list(prior_nls.keys())}")
                pairs = build_exclusion_list(
                    excluded_order=excluded_order,
                    symmetrize=symmetrize_excluded_pairs,
                    prior_nls=[prior_nls[n] for n in excluded_prior_nls],
                )
            else:
                pairs = build_exclusion_list(
                    excluded_order=excluded_order,
                    symmetrize=symmetrize_excluded_pairs,
                    topology=cg_trajs.topology,
                )
            excluded_pairs = {
                "excluded_order": excluded_order,
                "index_mapping": torch.as_tensor(pairs, dtype=torch.long),
            }
        for i in range(cg_trajs.n_frames):
            cg_traj = cg_trajs[i]
            cg_coords = cg_traj.xyz * 10
            for i in range(copies):
                cg_coord_list.append(cg_coords)
                cg_type_list.append(cg_types)
                cg_mass_list.append(cg_masses)
                cg_nls_list.append(prior_nls)
                cg_excl_list.append(excluded_pairs)

        samples.save_cg_output(save_dir, save_coord_force=False, save_cg_maps=False)

    data_list = []
    for coords, types, masses, nls, excl in zip(
        cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list, cg_excl_list,
    ):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[0]),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
        )
        data.neighbor_list = deepcopy(nls)
        if excl is not None:
            data.excluded_pairs = deepcopy(excl)
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
