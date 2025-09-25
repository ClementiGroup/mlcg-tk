from typing import Final, List, Optional
import numpy as np
import numpy.random as r
import h5py
from tqdm import tqdm
import os
from jsonargparse import CLI
from copy import deepcopy
from time import ctime

from mlcg.utils import load_yaml, dump_yaml

rng: Final = r.default_rng()

META_CG_EMBEDS_KEY: Final = "cg_embeds"
META_CG_NFRAMES_KEY: Final = "N_frames"


def longest_common_substring(s1, s2):
    # Create a 2D array to store lengths of longest common suffixes
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest_len = 0
    end_pos = 0  # To store the end position of the longest common substring in s1

    # Build the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_len:
                    longest_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0

    # Extract the longest common substring
    return s1[end_pos - longest_len : end_pos]


def longest_common_substring_multiple(strings):
    # Start by assuming the entire first string as a candidate
    common_substring = strings[0]

    # Find the longest common substring for each subsequent string
    for s in strings[1:]:
        common_substring = longest_common_substring(common_substring, s)
        if not common_substring:
            break  # No common substring found, exit early

    return common_substring


def add_noise_decoy_molecule(
    source_molecule: h5py.Group,
    location: h5py.Group,
    scale: float,
    name: str,
    coords_name: str = "cg_coords",
    forces_name: str = "cg_delta_forces",
    stride: int = 50,
    copy_metadata: bool = True,
) -> h5py.Group:
    """Create a zero-force decoy molecule from a real molecule in an h5.

    Function written by Aleksander Durumeric.

    The molecule is created by extracting coordinats from an existing molecule, adding
    Gaussian noise to them, and then storing them (along with 0-forces) under a new
    molecule entry. Entries are added for attrs if the corresponding metadata option
    is set.

    Arguments:
    ---------
    source_molecule:
        hdf5.Group that contains coordinate entries under "cg_coords". These coordinates
        are strided by `stride` and then noised to create the noise molecule coordinates.
    location:
        hdf5.Group under which a new group will be created for the added molecule. The
        new molecule is itself a group with name `name`.
    scale:
        Standard deviation of the added 0-mean Gaussian noise.
    name:
        Name of hdf5.Group added. See `location`.
    coords_name:
        Name of dataset that holds the coordinates. Used to query source_molecule and
        to name the corresponding entry in the new molecule.
    forces_name:
        Name of dataset that holds the forces. Only used to to name the corresponding
        entry in the new molecule.
    stride:
        Amount by which to stride the source data when creating the new molecule. For
        example `50` would create a noise molecule with approximatly 1/50th of the frames.
    copy_metadata:
        If true, the attrs under META_CG_EMBEDS_KEY is copied to the created
        molecule and META_CG_NFRAMES_KEY is used to store the number of frames.
        This typically should be set to True.

    Notes:
    -----
    This method saves coordinates and forces as float32, no matter what precision they
    originate as.
    """

    # obtain source coords (for h5s [::x] does a copy, not a view, do not do this if
    # operating on arrays or tensors for source data
    coords = source_molecule[coords_name][::stride]
    if len(coords.shape) != 3:
        raise ValueError("Unexpected coordinate shape.")
    # create noised coords (in place)
    coords += rng.normal(loc=0.0, scale=scale, size=coords.shape).astype(np.float32)
    # create zero forces
    forces = np.zeros_like(coords)

    # create new molecule, store data
    new_molecule_group = location.create_group(name)
    new_molecule_group[coords_name] = coords
    new_molecule_group[forces_name] = forces

    # place metadata
    if copy_metadata:
        new_molecule_group.attrs[META_CG_EMBEDS_KEY] = source_molecule.attrs[
            META_CG_EMBEDS_KEY
        ][:]
        new_molecule_group.attrs[META_CG_NFRAMES_KEY] = len(coords)

    return new_molecule_group


def add_decoy(
    h5_files: List[str],
    datasets: List[str],
    mol_name_prefix: str,
    scale: float,
    stride: Optional[int] = 50,
    append: Optional[bool] = True,
) -> None:
    """
    Adds decoy molecules with Gaussian noise to the specified HDF5 datasets, optionally combines them into a single HDF5 file.

    This function processes multiple HDF5 files, and for each file, it generates decoy molecules by adding Gaussian noise to
    the coordinates of the molecules in the specified dataset. The decoys are stored as separate molecules in the same datasets, with the option
    to append them to the existing h5s or to copy the existing h5s into new ones before appending the decoy molecules.
    The decoy molecules are given a name based on the provided prefix.

    After decoys are added to all specified files, the function can optionally combine the provided h5 files
    into a single combined HDF5 file, using external links to the original files. The name of the combined file can either
    be provided or generated automatically based on the input files.

    Arguments:
    ----------
    h5_files: List[str]
        A list of paths to the HDF5 files where decoy molecules should be added. Each file should contain one single dataset specified
        in the `datasets` argument.

    datasets: List[str]
        A list of dataset names within each corresponding HDF5 file. These datasets should contain molecules where
        decoys will be added.

    mol_name_prefix: str
        The prefix to be added to the name of each decoy molecule. Each decoy molecule will be named by appending its
        original molecule name to this prefix.

    scale: float
        The standard deviation of the Gaussian noise to be added to the coordinates of the beads.

    stride: Optional[int], default=50
        The stride value to be used when selecting frames from the original molecules. For example, a stride of 50 will
        select every 50th frame from the original molecule dataset. If not specified, defaults to 50.

    append: Optional[bool], default=True
        If `True`, decoy molecules will be added to the datasets in the existing HDF5 files. If `False`, the HDF5 file
        will be copied before appending the decoy molecules in the new HDF5 file.

    Notes:
    -----
    - The `scale` value controls the level of noise added to the decoy molecules. A larger value will result in more distorted configurations.
    - The `stride` argument helps reduce the number of frames included in the decoy molecule by selecting a subset based on
      the specified interval. The higher the stride the less decoys are present in the dataset.
    """
    assert len(h5_files) == len(
        datasets
    ), "this function currently cannot handle more than one dataset per h5 file"
    decoy_h5_files = []
    for i, h5_file in enumerate(h5_files):
        if not append:  # copy h5 datasets
            os.system(
                f"cp {os.path.dirname(h5_file)} {os.path.join(os.path.dirname(h5_file), f'DECOY_{os.path.basename(h5_dataset)}')}"
            )
            decoy_h5_file = os.path.join(
                os.path.dirname(h5_file), f"DECOY_{os.path.basename(h5_file)}"
            )
        else:
            decoy_h5_file = h5_file
        decoy_h5_files.append(decoy_h5_file)

        with h5py.File(decoy_h5_file, "a") as f:
            for mol in tqdm(f[datasets[i]].keys(), desc=f"Adding {datasets[i]} decoys"):
                add_noise_decoy_molecule(
                    source_molecule=f[datasets[i]][mol],
                    location=f[datasets[i]],
                    scale=5.0,
                    name=f"{mol_name_prefix}_{mol}",
                )


def update_partition_file(
    partition_file: str,
    decoy_h5_files: List[str],
    mol_name_prefixes: List[str],
    partition_name: str,
) -> None:
    """
    Updates a partition file by adding new molecules with specified prefixes to the "molecules" list of existing datasets.

    This function loads a YAML partition file, updates it by appending molecule names with specified prefixes, and then
    saves the updated partition back to a new YAML file. It is typically used in scenarios where new decoy molecules,
    generated with a prefix (such as a decoy identifier), need to be added to the partition file.

    Arguments:
    ----------
    partition_file: str
        Path to the original partition YAML file that contains metadata about the datasets and molecules. The function will
        read this file and modify the "molecules" list within the "metasets" section of the partition.

    decoy_h5_files: List[str]
        A list of paths to HDF5 files that contain the decoy molecules. Each file corresponds to a dataset in the partition file.
        The number and ordering of files should match the number and ordering of datasets in the partition file.

    mol_name_prefixes: List[str]
        A list of prefixes to be added to the names of existing molecules in the partition file. Each prefix will be prepended
        to the names of the molecules in the partition's "molecules" list, essentially creating new "decoy" molecules with
        the prefixed names.

    partition_name: str
        The path where the updated partition YAML file will be saved. This will overwrite the existing file at that location.

    Notes:
    -----
    - The function assumes that the partition YAML file follows a specific structure, where molecule names are listed under
      the "train" section in the "metasets" -> "molecules" key.
    - The function does not add decoy molecules to the validation dataset provided in the partition file

    Example:
    --------
    If the partition file contains the following:
    ```yaml
    train:
      metasets:
        my_dataset:
          molecules:
            - mol1
            - mol2
    ```
    and mol_name_prefixes = ["DECOY_5", "DECOY_05"]
    The updated partition file will contain:
    ```yaml
    train:
      metasets:
        my_dataset:
          molecules:
            - mol1
            - mol2
            - DECOY_5_mol1
            - DECOY_5_mol2
            - DECOY_05_mol1
            - DECOY_05_mol2
    ```
    """
    partition = load_yaml(partition_file)

    assert len(decoy_h5_files) == len(
        partition["train"]["metasets"].keys()
    ), "Currently the number of decoy HDF5 files must match the number of datasets in the partition file."

    for dataset, decoy_h5_file in zip(
        partition["train"]["metasets"].keys(), decoy_h5_files
    ):
        mols = deepcopy(partition["train"]["metasets"][dataset]["molecules"])
        detailed_indices = partition["train"]["metasets"][dataset]["detailed_indices"]

        with h5py.File(decoy_h5_file, "r") as f:
            if dataset not in f:
                raise ValueError(
                    f"Dataset {dataset} not found in HDF5 file {decoy_h5_file}."
                )

        for mol in mols:
            for prefix in mol_name_prefixes:
                partition["train"]["metasets"][dataset]["molecules"].append(
                    f"{prefix}_{mol}"
                )
                decoy_mol = f"{prefix}_{mol}"
                with h5py.File(decoy_h5_file, "r") as f:
                    if decoy_mol not in f[dataset]:
                        raise ValueError(
                            f"Molecule {decoy_mol} not found in dataset {dataset} of HDF5 file {decoy_h5_file}."
                        )
                    if "cg_coords" not in f[dataset][decoy_mol]:
                        raise ValueError(
                            f"'cg_coords' dataset not found for molecule {decoy_mol} in dataset {dataset} of HDF5 file {decoy_h5_file}."
                        )

                    idx_decoy = np.arange(f[dataset][decoy_mol]["cg_coords"].shape[0])

                original_path = detailed_indices.get(mol, "")
                prior_tag = (
                    os.path.basename(original_path)
                    .replace(".npy", "")
                    .replace("train_idx_", "")
                    .replace(f"{mol}_", "")
                )

                if original_path:
                    dir_path = os.path.dirname(original_path)
                    new_filename = f"train_idx_{decoy_mol}_{prior_tag}.npy"
                    new_path = os.path.join(dir_path, new_filename)

                    np.save(new_path, idx_decoy)

                    detailed_indices[decoy_mol] = new_path

    dump_yaml(partition_name, partition)


if __name__ == "__main__":
    print("Start add_decoys.py: {}".format(ctime()))

    CLI([add_decoy, update_partition_file])

    print("Finish add_decoys.py: {}".format(ctime()))
