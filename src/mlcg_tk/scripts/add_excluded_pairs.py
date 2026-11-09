import os.path as osp

from mlcg_tk.input_generator.raw_dataset import RawDataset
from mlcg_tk.input_generator.utils import get_output_tag, build_exclusion_list

from tqdm import tqdm
from time import ctime
from typing import List, Optional
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
    excluded_prior_nls: Optional[List[str]] = None,
    prior_tag: Optional[str] = None,
):
    """
    Build excluded pairs list and store them in existing h5 molecule groups

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
    symmetrize_excluded_pairs : bool
        If true, symmetric pairs will be added to excluded pairs list.
    excluded_prior_nls : Optional[List[str]]
        List of names of prior neighbour lists to be used to compute excluded pairs. If None, 
        all pairs up to excluded_order bonds apart are computed from the topology and excluded. 
        The highest order of the selected neighbour lists must equal excluded_order.
    prior_tag : Optional[str]
        Tag of the prior neighbour lists to load (required if excluded_prior_nls is set)
    """
    output_tag = get_output_tag([dataset_name, force_tag], placement="after")
    h5_file = osp.join(save_dir, f"{output_tag[1:]}.h5")
    if excluded_prior_nls is not None and prior_tag is None:
        raise ValueError("prior_tag must be provided when excluded_prior_nls is set")

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
            cg_structure = None
            prior_nls = None
            if excluded_prior_nls is not None:
                full_prior_nls = samples.load_prior_nls(
                        mol_save_templ,
                        get_output_tag(prior_tag, placement="after")
                )
                missing = [n for n in excluded_prior_nls if n not in full_prior_nls]
                if len(missing) != 0:
                    raise KeyError(f"{missing} not in prior nls {list(full_prior_nls.keys())}")
                prior_nls = [full_prior_nls[n] for n in excluded_prior_nls]
            else:
                cg_structure = samples.load_cg_structure(mol_save_templ)

            excluded_pairs = build_exclusion_list(
                excluded_order=excluded_order,
                symmetrize=symmetrize_excluded_pairs,
                topology=cg_structure.topology if cg_structure is not None else None,
                prior_nls=prior_nls,
            )
            hdf_group.attrs["excluded_pairs"] = excluded_pairs
        f.attrs["excluded_order"] = excluded_order


def main():
    print("Start add_excluded_pairs.py: {}".format(ctime()))

    CLI([add_excluded_pairs], as_positional=False)

    print("Finish add_excluded_pairs.py: {}".format(ctime()))


if __name__ == "__main__":
    main()
