import os.path as osp
import sys

import mdtraj as md
import numpy as np
import torch

from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import FORCE_KEY
from mlcg.nn import SumOut

from mlcg_tk.input_generator.raw_dataset import *
from mlcg_tk.input_generator.utils import get_output_tag

from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI


def remove_baseline_forces_collated(
    collated_data: AtomicData, model: SumOut
) -> AtomicData:
    """Compute the forces on the input :obj:`collated_data` with the :obj:`models`
    and remove them from the reference forces contained in :obj:`data_list`.
    The computation of the forces is done on the whole :obj:`data_list` at once
    so it should not be too large.

    Parameters
    ----------
    collated_data:
        Collated list of AtomicData instances that contain the full
        reference forces
    models:
        SumOut object containing models that compute prior/baseline
        forces

    Returns
    -------
    List[AtomicData]:
        Uncollated list of AtomicData instances, where the value of the
        'forces' field is now the delta forces (original forces minus
        the baseline/prior forces). An additional field 'baseline' forces
        is added, whose value is equal to the baseline/prior forces
    """

    model.eval()
    collated_data = model(collated_data)
    baseline_forces = collated_data.out[FORCE_KEY].detach()
    collated_data.forces -= baseline_forces
    collated_data.baseline_forces = baseline_forces
    return collated_data


def produce_delta_forces(
    dataset_name: str,
    names: List[str],
    tag: str,
    save_dir: str,
    prior_tag: str,
    prior_fn: str,
    device: str,
    batch_size: int,
    force_tag: Optional[str] = None,
    mol_num_batches: Optional[int] = 1,
):
    """
    Removes prior energy terms from input forces to produce delta force input
    for training

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    tag : str
        Label given to all output files produced from dataset
    save_dir : str
        Path to directory from which input will be loaded and to which output will be saved
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_fn : str
        Path to filename in which prior model is saved
    device: str
        Device on which to run delta force calculations
    batch_size : int
        Number of frames to take per batch
    force_tag: str
        Optional tag to identify input for a particular run of delta force calculation
    mol_num_batches : int
        If greater than 1, will load each molecule data from the specified number of batches
        that were be treated as different samples
    """

    # prior_model = torch.load(open(prior_fn, "rb")).models.to(device)
    prior_model = torch.load(open(prior_fn, "rb")).to(device)
    dataset = RawDataset(dataset_name, names, tag, n_batches=mol_num_batches)
    for samples in tqdm(
        dataset, f"Processing delta forces for {dataset_name} dataset..."
    ):
        if not samples.has_saved_cg_output(save_dir, prior_tag):
            continue
        coords, forces, embeds, pdb, prior_nls = samples.load_cg_output(
            save_dir=save_dir, prior_tag=prior_tag
        )

        num_frames = coords.shape[0]
        delta_forces = []
        aux_data_list = [
            AtomicData.from_points(
                pos=torch.tensor(coords[i]),
                forces=torch.tensor(forces[i]),
                atom_types=torch.tensor(embeds),
                masses=None,
                neighborlist=prior_nls,
            )
            for i in range(batch_size)
        ]
        collated_data, _, _ = collate(aux_data_list[0].__class__, aux_data_list)
        collated_data = collated_data.to(device)
        slices = range(0, num_frames, batch_size)
        n_chunks = len(slices) - 1
        for k in range(n_chunks):
            current_frames = slice(slices[k], slices[k + 1])

            collated_data.pos = torch.tensor(
                coords[current_frames, :, :].reshape(-1, 3),
                device=device,
            )
            collated_data.forces = torch.tensor(
                forces[current_frames, :, :].reshape(-1, 3),
                device=device,
            )
            _ = remove_baseline_forces_collated(
                collated_data,
                prior_model,
            )
            delta_force = (
                collated_data.forces.detach()
                .cpu()
                .reshape(slices[k + 1] - slices[k], -1, 3)
            )
            delta_forces.append(delta_force.numpy())
        if slices[-1] < num_frames:
            # final piece
            last_batch_size = num_frames - slices[-1]
            collated_data, _, _ = collate(
                aux_data_list[0].__class__, aux_data_list[:last_batch_size]
            )
            collated_data = collated_data.to(device)
            collated_data.pos = torch.tensor(
                coords[slices[-1] :, :, :].reshape(-1, 3),
                device=device,
            )
            collated_data.forces = torch.tensor(
                forces[slices[-1] :, :, :].reshape(-1, 3),
                device=device,
            )
            _ = remove_baseline_forces_collated(
                collated_data,
                prior_model,
            )
            delta_force = (
                collated_data.forces.detach().cpu().reshape(last_batch_size, -1, 3)
            )
            delta_forces.append(delta_force.numpy())

        fnout = os.path.join(
            save_dir,
            f"{get_output_tag([tag, samples.name, prior_tag, force_tag], placement='before')}delta_forces.npy",
        )

        np.save(
            fnout,
            np.concatenate(delta_forces, axis=0).reshape(*coords.shape),
        )


def main():
    print("Start produce_delta_forces.py: {}".format(ctime()))
    CLI([produce_delta_forces])
    print("Finish produce_delta_forces.py: {}".format(ctime()))


if __name__ == "__main__":
    main()    