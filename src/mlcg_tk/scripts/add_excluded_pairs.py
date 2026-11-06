import os.path as osp

from mlcg_tk.input_generator.raw_dataset import RawDataset
from mlcg_tk.input_generator.utils import get_output_tag, build_exclusion_list

from tqdm import tqdm
from time import ctime
from typing import List
from jsonargparse import CLI
import h5py
import warnings


def add_excluded_pairs(
    dataset_name: str,
    names: List[str],
    dataset_tag: str,
    force_tag: str,
    training_data_dir: str,
    save_dir: str,
    excluded_order: int,
    mol_num_batches: int = 1,
    keep_batches: bool = False,
    overwrite: bool = False,
    symmetrize_excluded_pairs: bool = False,
):
    """
    Build excluded pairs list from topology and store them in existing h5 molecule groups

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    dataset_tag : str
        Label given to all output files produced from dataset
    force_tag : str
        Label given to produced delta forces and saved packaged data
    training_data_dir : str
        Path to directory from which input will be loaded
    save_dir : str
        Path to directory to which h5 file has been saved
    excluded_order : int
        Maximum graph separation of the excluded pairs:
        2 = bonds
        3 = bonds and angles
        4 = bonds, angles, and dihedrals
    mol_num_batches : int
        If greater than 1, will load each molecule data from the specified number of batches
        that will be treated as different samples.
    keep_batches : bool
        If set to True, batches will be put as individual molecules in the h5 dataset; otherwise, 
        if batches exist, they will be accumulated into one single molecule.
    overwrite : bool
        If True, replace existing exclusions.
    symmetrize_excluded_pairs: bool
        If true, symmetric pairs will be added to excluded pairs list.
    """
    output_tag = get_output_tag([dataset_name, force_tag], placement="after")
    h5_file = osp.join(save_dir, f"{output_tag[1:]}.h5")

    with h5py.File(h5_file, "a") as f:
        existing = f.attrs.get("excluded_order")
        if existing is not None and not overwrite:
            raise ValueError(
                f"{h5_file} already has excluded_order={existing}; use overwrite=True"
            )
        metaset = f[dataset_name]
        if keep_batches and mol_num_batches > 1:
            dataset = RawDataset(dataset_name, names, dataset_tag, n_batches=mol_num_batches)
        else:
            dataset = RawDataset(dataset_name, names, dataset_tag)
        merge_batches = mol_num_batches > 1 and not keep_batches
        for samples in tqdm(dataset, f"Adding excluded pairs to {dataset_name} dataset..."):
            mol = samples.mol_name if merge_batches else samples.name
            name = f"{samples.tag}{mol}"
            if name not in metaset:
                warnings.warn(f"{name} not in {h5_file}; skipping")
                continue
            hdf_group = metaset[name]
            mol_save_templ = osp.join(
                training_data_dir,
                get_output_tag([samples.tag, samples.mol_name], placement="before"),
            )
            cg_structure = samples.load_cg_structure(mol_save_templ)

            excluded_pairs = build_exclusion_list(
                topology=cg_structure.topology,
                order=excluded_order,
                symmetrize=symmetrize_excluded_pairs,
            )
            hdf_group.attrs["excluded_pairs"] = excluded_pairs
        f.attrs["excluded_order"] = excluded_order


def main():
    print("Start add_excluded_pairs.py: {}".format(ctime()))

    CLI([add_excluded_pairs], as_positional=False)

    print("Finish add_excluded_pairs.py: {}".format(ctime()))


if __name__ == "__main__":
    main()
