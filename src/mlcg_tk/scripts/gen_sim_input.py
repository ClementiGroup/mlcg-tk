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

from typing import Dict, List, Union, Callable, Optional, Type
from jsonargparse import CLI
import pickle as pck

import numpy as np
import mdtraj as md

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
    collection_cls: Type[SampleCollection] = SampleCollection,
    smpl_loader: Type[DatasetLoader] = SimInput_loader,
    cg_virtual_atoms: Optional[List[Dict]] = None,
    pbc: Optional[List[bool]] = None,
    cell1: Optional[Union[List[float], np.ndarray]] = None,
    cell2: Optional[Union[List[float], np.ndarray]] = None,
    cell3: Optional[Union[List[float], np.ndarray]] = None,
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
    collection_cls : Type[SampleCollection]
        SampleCollection class to use for dataset
    smpl_loader : Type[DatasetLoader]
        DatasetLoader class to use for loading raw data
    cg_virtual_atoms : List[Dict]
        Dictionary mapping CG bead indices to lists of atomistic atom indices.
        These atomistic atoms will be weighted equally (1/N each) in the CG mapping.
        Format: cg_bead_index: [atom1_index, atom2_index, ...]
    pbc: Optional[List[bool]]
        Periodic boundary conditions (3 booleans for x, y, z directions)
    cell1: Optional[Union[List[float], np.ndarray]]
        First cell vector (3 floats)
    cell2: Optional[Union[List[float], np.ndarray]]
        Second cell vector (3 floats)
    cell3: Optional[Union[List[float], np.ndarray]]
        Third cell vector (3 floats)
    """
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []
    cg_pbc_list = []
    cg_cell_list = []

    use_pbc = pbc is not None
    use_cell = all(cell is not None for cell in [cell1, cell2, cell3])
    
    if use_pbc:
        if len(pbc) != 3:
            raise ValueError("pbc must be a list/array of 3 booleans")
    
    if use_cell:
        if len(cell1) != 3:
            raise ValueError("cell1 must be a list/array of 3 floats corresponding to v1")
        if len(cell2) != 3:
            raise ValueError("cell2 must be a list/array of 3 floats corresponding to v2")
        if len(cell3) != 3:
            raise ValueError("cell3 must be a list/array of 3 floats corresponding to v3")
    
    # Consistency between PBC and cell
    if use_pbc and not use_cell:
        raise ValueError("cell vectors must be provided if pbc is specified")
    if use_cell and not use_pbc:
        raise ValueError("pbc must be provided if cell vectors are specified")
    
    if use_cell:
        cell = np.stack((cell1, cell2, cell3), axis=1, dtype=np.float32)
        print(f"Using cell: {cell}")
    else:
        cell = None
        print("No cell information provided - using non-periodic system")
    
    dataset = SimInput(dataset_name, tag, pdb_fns, collection_cls=collection_cls)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = smpl_loader()
        samples.input_traj, samples.top_dataframe = sample_loader.get_input_top(
            name=samples.name, raw_data_dir=raw_data_dir
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
            virtual_atoms=cg_virtual_atoms,
        )
        if cg_virtual_atoms:
            cg_map = samples.cg_map
            np_trajs = np.einsum('nmi,km->nki', samples.input_traj.xyz, cg_map)
            tmp_topology = samples.input_traj.atom_slice(samples.cg_atom_indices).topology
            cg_trajs = md.Trajectory(np_trajs, tmp_topology)
        else:
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
    if not use_pbc:
        # Non-periodic case
        for coords, types, masses, nls in zip(
            cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list,
        ):
            data = AtomicData.from_points(
                pos=torch.tensor(coords[0]),
                atom_types=torch.tensor(types),
                masses=torch.tensor(masses),
            )
            data.neighbor_list = deepcopy(nls)
            data_list.append(data)
    else:
        # Periodic case
        for coords, types, masses, nls, pbcs, cells in zip(
            cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list, cg_pbc_list, cg_cell_list
        ):
            data = AtomicData.from_points(
                pos=torch.tensor(coords[0]),
                atom_types=torch.tensor(types),
                masses=torch.tensor(masses),
                pbc=torch.tensor(pbcs),
                cell=torch.tensor(cells)
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