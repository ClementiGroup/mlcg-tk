import os.path as osp
import sys

from mlcg_tk.input_generator.raw_dataset import SampleCollection, RawDataset
from mlcg_tk.input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from mlcg_tk.input_generator.raw_data_loader import DatasetLoader
from mlcg_tk.input_generator.prior_gen import Bonds, PriorBuilder
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
import pickle as pck


def process_raw_dataset(
    dataset_name: str,
    names: List[str],
    sample_loader: DatasetLoader,
    raw_data_dir: str,
    tag: str,
    pdb_template_fn: str,
    save_dir: str,
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues: List[str],
    cg_mapping_strategy: str,
    stride: int = 1,
    force_stride: int = 100,
    filter_cis: Optional[bool] = False,
    batch_size: Optional[int] = None,
    mol_num_batches: Optional[int] = 1,
    atoms_batch_size: Optional[int] = None,
):
    """
    Applies coarse-grained mapping to coordinates and forces using input sample
    topology and specified mapping strategies

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    sample_loader : DatasetLoader
        Loader object defined for specific dataset
    raw_data_dir : str
        Path to coordinate and force files
    tag : str
        Label given to all output files produced from dataset
    pdb_template_fn : str
        Template file location of atomistic structure to be used for topology
    save_dir : str
        Path to directory in which output will be saved
    cg_atoms : List[str]
        List of atom names to preserve in coarse-grained resolution
    embedding_map : CGEmbeddingMap
        Mapping object
    embedding_func : Callable
        Function which will be used to apply CG mapping
    skip_residues : List[str]
        List of residues to skip, can be None
    cg_mapping_strategy : str
        Strategy to use for coordinate and force mappings;
        currently only "slice_aggregate" and "slice_optimize" are implemented
    stride : int
        Interval by which to stride loaded data
    force_stride : int
        stride for inferring the force maps in aggforce
    filter_cis : bool
        if True, frames with cis-configurations will be filtered out from the dataset
    batch_size : int
        Optional size in which performing batches of AA mapping to CG, to avoid
        memory overhead in large AA dataset
    mol_num_batches : int
        If greater than 1, will save each molecule data into the specified number of batches
        that will be treated as different samples
    atoms_batch_size : int, optional
        Optional batch size for processing atoms in large molecules (default: None). If specified, constraints among atoms for coordinate and
        force mappings (as defined by `cg_mapping_strategy`) will be computed in batches of this size. To significantly improve
        computational efficiency, it is assumed that structures have ordered residues. If `atoms_batch_size` exceeds the total number of atoms
        in the molecule, all atoms will be processed at once (default behavior).

    """
    dataset = RawDataset(dataset_name, names, tag, n_batches=mol_num_batches)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.mol_name, pdb_template_fn
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        aa_coords, aa_forces = sample_loader.load_coords_forces(
            raw_data_dir,
            samples.mol_name,
            stride=stride,
            batch=samples.batch,
            n_batches=samples.n_batches,
        )

        if samples.n_batches > 1 and samples.batch > 1:
            # this ensures that we are using the same force map across batches
            mapping = samples.load_cg_force_map(save_dir)
        else:
            mapping = cg_mapping_strategy

        cg_coords, cg_forces = samples.process_coords_forces(
            aa_coords,
            aa_forces,
            topology=samples.input_traj.top,
            mapping=mapping,
            force_stride=force_stride,
            batch_size=batch_size,
            filter_cis=filter_cis,
            atoms_batch_size=atoms_batch_size,
        )

        samples.save_cg_output(save_dir, save_coord_force=True, save_cg_maps=True)
        # the sample object will retain the output so it makes sense to delete them
        del samples.cg_coords
        del samples.cg_forces
        del samples.cg_map
        del samples.force_map


def build_neighborlists(
    dataset_name: str,
    names: List[str],
    sample_loader: DatasetLoader,
    tag: str,
    pdb_template_fn: str,
    save_dir: str,
    cg_atoms: List[str],
    embedding_map: CGEmbeddingMap,
    embedding_func: Callable,
    skip_residues: List[str],
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    raw_data_dir: Union[str, None] = None,
    cg_mapping_strategy: Union[str, None] = None,
    stride: int = 1,
    force_stride: int = 100,
    filter_cis: bool = False,
    batch_size: Optional[int] = None,
    mol_num_batches: Optional[int] = 1,
    atoms_batch_size: Optional[int] = None,
):
    """
    Generates neighbour lists for all samples in dataset using prior term information

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    sample_loader : DatasetLoader
        Loader object defined for specific dataset
    tag : str
        Label given to all output files produced from dataset
    pdb_template_fn : str
        Template file location of atomistic structure to be used for topology
    save_dir : str
        Path to directory in which output will be saved
    cg_atoms : List[str]
        List of atom names to preserve in coarse-grained resolution
    embedding_map : CGEmbeddingMap
        Mapping object
    embedding_func : Callable
        Function which will be used to apply CG mapping
    skip_residues : List[str]
        List of residues to skip, can be None
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters

    stride : int
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    force_stride : int
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    filter_cis : bool
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    batch_size : bool
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    mol_num_batches : int
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    atoms_batch_size : int
        unused in this function
        present to allow the use of the same .yaml config for process_raw_dataset and build_neighborlists
    """
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(dataset, f"Building NL for {dataset_name} dataset..."):
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.name, pdb_template_fn
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        prior_nls = samples.get_prior_nls(
            prior_builders, save_nls=True, save_dir=save_dir, prior_tag=prior_tag
        )

def main():
    print("Start gen_input_data.py: {}".format(ctime()))
    CLI([process_raw_dataset, build_neighborlists])
    print("Finish gen_input_data.py: {}".format(ctime()))


if __name__ == "__main__":
    main()