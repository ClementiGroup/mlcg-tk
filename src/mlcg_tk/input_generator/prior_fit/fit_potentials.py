import torch
from typing import Dict, Callable, Optional
import numpy as np
from scipy.integrate import trapezoid
from ..prior_gen import PriorBuilder
from ..embedding_maps import CGEmbeddingMap
from copy import deepcopy


def fit_potentials(
    nl_name: str,
    prior_builder: PriorBuilder,
    embedding_map: Optional[CGEmbeddingMap],
    temperature: float,
):
    """
    Fits energy function to atom type-specific statistics defined
    for a group of atoms in a neighbour list.

    Assumes that the resulting prior energies will be in [kcal/mol]!

    Parameters
    ----------
    nl_name:
        Neighbour list label
    prior_builder:
        PriorBuilder object containing histogram data
    embedding_map [Optional]:
        Instance of CGEmbeddingMap defining coarse-grained mapping;
        required to alter GLY statistics if defined in PriorBuilder.nl_builder.
    temperature:
        Temperature of the simulation data data (default=300K)

    Returns
    -------
    prior_models:
        nn.ModuleDict of prior models
    all_stats:
        dictionary of statistics dictionaries for each prior fit

    Returns
    -------
    model :ref:`mlcg.nn.GradientsOut` module containing gathered
    statistics and estimated energy parameters based on the `TargetPrior`.
    The following key/value pairs are common across all `TargetPrior`s:

    .. code-block:: python

        (*specific_types) : {

            ...

            "p" : torch.Tensor of shape [n_bins], containing the normalized bin counts
                of the of the 1-D feature corresponding to the atom_type group
                (*specific_types) = (specific_types[0], specific_types[1], ...)
            "p_bin": : torch.Tensor of shape [n_bins] containing the bin center values
            "V" : torch.tensor of shape [n_bins], containing the emperically estimated
                free energy curve according to a direct Boltzmann inversion of the
                normalized probability distribution for the feature.
            "V_bin" : torch_tensor of shape [n_bins], containing the bin center values
        }

    where `...` indicates other sub-key/value pairs apart from those enumerated above,
    which may appear depending on the chosen `TargetPrior`. For example,
    if `TargetPrior` is `HarmonicBonds`, there will also be keys/values associated with
    estimated bond constants and means.
    """
    histograms = prior_builder.histograms[nl_name]
    bin_centers = prior_builder.histograms.bin_centers
    prior_fit_fn = prior_builder.prior_fit_fn

    target_fit_kwargs = prior_builder.nl_builder.get_fit_kwargs(nl_name)

    kB = 0.0019872041  # kcal/(mol⋅K)
    beta = 1 / (temperature * kB)

    statistics = {}
    for kf in list(histograms.keys()):
        hist = torch.tensor(histograms[kf])

        mask = hist > 0
        bin_centers_nz = bin_centers[mask]
        ncounts_nz = hist[mask]
        dG_nz = -torch.log(ncounts_nz) / beta

        params = prior_fit_fn(
            bin_centers_nz=bin_centers_nz,
            dG_nz=dG_nz,
            ncounts_nz=ncounts_nz,
            **target_fit_kwargs
        )

        statistics[kf] = params

        statistics[kf]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kf]["p_bin"] = bin_centers
        statistics[kf]["V"] = dG_nz
        statistics[kf]["V_bin"] = bin_centers_nz

    if getattr(prior_builder.nl_builder, "replace_gly_ca_stats", False):
        statistics = replace_gly_stats(
            statistics, gly_bead=embedding_map["GLY"], ca_bead=embedding_map["CA"]
        )

    prior_model = prior_builder.get_prior_model(
        statistics, nl_name, targets="forces", **target_fit_kwargs
    )

    return prior_model


def replace_gly_stats(statistics, gly_bead, ca_bead):
    """
    Helper method for replacing poor GLY statistics for dihedral NL with statistics
    associated with general CA beads.
    """
    gly_atom_groups = [group for group in list(statistics.keys()) if gly_bead in group]
    for group in gly_atom_groups:
        gly_idx = group.index(gly_bead)
        ca_group = list(deepcopy(group))
        ca_group[gly_idx] = ca_bead
        try:
            statistics[group] = statistics[tuple(ca_group)]
        except KeyError:
            ca_group = (ca_group[0], ca_group[2], ca_group[1], ca_group[3])
    return statistics
