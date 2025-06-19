import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader
from input_generator.prior_gen import Bonds, PriorBuilder
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
    Processes a raw dataset by applying coarse-grained (CG) mapping to atomic coordinates and forces,
    using the provided topology and mapping strategies.

    Parameters
    ----------
    dataset_name : str
        Identifier for the dataset.
    names : List[str]
        List of molecule or sample names to process.
    sample_loader : DatasetLoader
        Loader object for retrieving trajectories and data.
    raw_data_dir : str
        Directory containing raw coordinate and force files.
    tag : str
        Label to append to all output files generated from this dataset.
    pdb_template_fn : str
        Path to the template PDB file for atomistic topology.
    save_dir : str
        Directory where processed outputs will be saved.
    cg_atoms : List[str]
        Atom names to retain in the coarse-grained representation.
    embedding_map : CGEmbeddingMap
        Object defining the mapping from atomistic to CG representations.
    embedding_func : Callable
        Function to apply the CG mapping.
    skip_residues : List[str]
        Residues to exclude from processing (can be empty).
    cg_mapping_strategy : str
        Strategy for mapping coordinates and forces (e.g., "slice_aggregate", "slice_optimize").
    stride : int, optional
        Interval for subsampling loaded data (default: 1).
    force_stride : int, optional
        Interval for subsampling forces during mapping (default: 100).
    filter_cis : Optional[bool], optional
        If True, filters out frames with cis-configurations (default: False).
    batch_size : Optional[int], optional
        Number of frames to process in each batch to reduce memory usage (default: None).
    mol_num_batches : Optional[int], optional
        Number of batches to split each molecule's data into (default: 1).
    atoms_batch_size : Optional[int], optional
        Batch size for atoms when processing large molecules (default: None).

    Returns
    -------
    None
        Saves processed CG data and mapping files to the specified directory.
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
    atoms_batch_size : Optional[int], optional
        Batch size for atoms when processing large molecules (default: None).
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


if __name__ == "__main__":
    print("Start gen_input_data.py: {}".format(ctime()))

    CLI([process_raw_dataset, build_neighborlists])

    print("Finish gen_input_data.py: {}".format(ctime()))
