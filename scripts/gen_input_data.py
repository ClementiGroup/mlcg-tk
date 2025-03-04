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
    batch_size: Optional[int] = None
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
    """
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.name, pdb_template_fn
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,
        )

        aa_coords, aa_forces = sample_loader.load_coords_forces(
            raw_data_dir, samples.name, stride=stride
        )

        cg_coords, cg_forces = samples.process_coords_forces(
            aa_coords, 
            aa_forces,
            topology=samples.input_traj.top,
            mapping=cg_mapping_strategy, 
            force_stride=force_stride,
            batch_size=batch_size,
            filter_cis=filter_cis
        )

        samples.save_cg_output(save_dir, save_coord_force=True, save_cg_maps=True)
        # the sample object will retain the output so it makes sense to delete them 
        del samples.cg_coords
        del samples.cg_forces
        


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
    filter_cis: bool = False,
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
