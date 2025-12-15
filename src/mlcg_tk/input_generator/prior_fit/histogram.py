import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Optional
from collections import defaultdict
import numpy as np
from copy import deepcopy
import torchist

from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import _Prior
from mlcg.geometry._symmetrize import _symmetrise_map, _flip_map
from mlcg.utils import tensor2tuple


plt.rcParams["figure.max_open_warning"] = 50


class HistogramsNL:
    """
    Accumulates and stores statistics for a given feature associated with
    specific atom groups (from defined neighbour lists).

    Attributes
    ----------
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin:
        Lower bound of bin edges
    bmax:
        Upper bound of bin edges
    """

    def __init__(
        self,
        n_bins: int,
        bmin: float,
        bmax: float,
    ) -> None:
        """
        Bin centers are set automatically from n_bins, bmin, and bmax.
        """
        self.n_bins = n_bins
        self.bmin = bmin
        self.bmax = bmax
        self.bin_centers = _get_bin_centers(n_bins, bmin, bmax)
        self.data = defaultdict(
            lambda: defaultdict(lambda: np.zeros(n_bins, dtype=np.float64))
        )

    def accumulate_statistics(
        self,
        nl_name: str,
        values: torch.Tensor,
        key_dict: dict,
        weights: Optional[torch.Tensor],
    ) -> None:
        """
        Accumulates statistics from computed features.

        Parameters
        ----------
        nl_name:
            Neighbour list tag
        values:
            Tensor of computed values to be binned
        atom_types:
            Tensor of embedding types associated with CG beads
        mapping:
            Tensor of atom groups for which values have been computed
        """
        hists = compute_hist_with_rep(
            values,
            key_dict,
            self.n_bins,
            self.bmin,
            self.bmax,
            weights,
        )
        for k, hist in hists.items():
            self.data[nl_name][k] += hist

    def __getitem__(self, nl_name: str):
        """
        Returns histograms associated with neighbour list label
        """
        return deepcopy(self.data[nl_name])

    def plot_histograms(self, key_map=None):
        """
        Plots distributions of binned features for data
        """
        figs = []
        for nl_name, hists in self.data.items():
            fig = plt.figure(figsize=(10, 6))
            ax = plt.gca()
            ax.set_title(f"histograms for NL:'{nl_name}'")
            if key_map is None:
                keymap = {k: str(k) for k in hists}
            else:
                keymap = {ks: list([key_map[k] for k in ks]) for ks in hists}

            for key, hist in hists.items():
                norm = np.abs(hist).max()
                ax.plot(self.bin_centers, hist / norm, label=f"{keymap[key]}")
            ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), ncols=len(hists) // 20 + 1
            )
            figs.append((nl_name, fig))

        return figs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = {
            nl_name: {key: hist for key, hist in hists.items()}
            for nl_name, hists in self.data.items()
        }
        return state

    def __setstate__(self, newstate):
        n_bins = newstate["n_bins"]
        data = defaultdict(
            lambda: defaultdict(lambda: np.zeros(n_bins, dtype=np.float64))
        )
        for nl_name, hists in newstate["data"].items():
            for key, hist in hists.items():
                data[nl_name][key] = hist
        newstate["data"] = data
        self.__dict__.update(newstate)


def _get_all_unique_keys(unique_types: torch.Tensor, order: int) -> torch.Tensor:
    """Helper function for returning all unique, symmetrised atom type keys

    Parameters
    ----------
    unique_types:
        Tensor of unique atom types of shape (order, n_unique_atom_types)
    order:
        The order of the interaction type

    Returns
    -------
    torch.Tensor:
       Tensor of unique atom types, symmetrised
    """
    # get all combinations of size order between the elements of unique_types
    keys = torch.cartesian_prod(*[unique_types for ii in range(order)]).t()
    # symmetrize the keys and keep only unique entries
    sym_keys = _symmetrise_map[order](keys)
    unique_sym_keys = torch.unique(sym_keys, dim=1)
    return unique_sym_keys


def _get_bin_centers(nbins: int, b_min: float, b_max: float) -> torch.Tensor:
    """Returns bin centers for histograms.

    Parameters
    ----------
    feature:
        1-D input values of a feature.
    nbins:
        Number of bins in the histogram
    b_min
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    b_max
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    torch.Tensor:
        torch tensor containing the locaations of the bin centers
    """

    if b_min >= b_max:
        raise ValueError("b_min must be less than b_max.")

    bin_centers = torch.zeros((nbins,), dtype=torch.float64)

    delta = (b_max - b_min) / nbins
    bin_centers = (
        b_min + 0.5 * delta + torch.arange(0, nbins, dtype=torch.float64) * delta
    )
    return bin_centers


def compute_hist_with_keys(
    values: torch.Tensor,
    key_dict: dict,
    nbins: int,
    bmin: float,
    bmax: float,
    weights: Optional[torch.Tensor],
) -> Dict:
    """Compute histograms using precomputed unique keys for this nl_name."""

    order = key_dict["order"]
    unique_keys_in_data = key_dict["unique_keys_in_data"]
    inverse_indices = key_dict["inverse_indices"]

    histograms = {}

    if unique_keys_in_data.numel() == 0:
        return histograms
    
    n_unique_keys = unique_keys_in_data.shape[1]

    bins = torch.linspace(
        bmin, bmax, steps=nbins + 1, dtype=values.dtype, device=values.device
    )

    for idx in range(n_unique_keys):
        mask = inverse_indices == idx
#        print(f"Shape of values: {values.shape}, shape of mask: {mask.shape}")
#        print(f"Mask first 10 values: {mask[:10]}")
        if not mask.any():
            continue

        val = values[mask]
        if isinstance(weights, torch.Tensor):
            n_atomgroups = int(val.shape[0] / weights.shape[0])
            # hist, _ = torch.histogram(
            #    val, bins=bins, weight=weights.tile((n_atomgroups,))
            # )
            hist = torchist.histogram(
                val, edges=bins, weight=weights.tile((n_atomgroups,))
            )
        else:
            # hist, _ = torch.histogram(val, bins=bins)
            hist = torchist.histogram(val, edges=bins)

        unique_key = unique_keys_in_data[:, idx]
        kk = tensor2tuple(unique_key)
        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kk] = hist.cpu().numpy()
        histograms[kf] = deepcopy(hist.cpu().numpy())

    return histograms

def compute_hist_with_rep(
    values: torch.Tensor,
    key_dict: dict,
    nbins: int,
    bmin: float,
    bmax: float,
    weights: Optional[torch.Tensor],
) -> Dict:
    """
    Compute histograms using precomputed unique keys for this nl_name.
    
    Parameters
    ----------
    values : torch.Tensor
        Computed feature values for the batch
    key_dict : dict
        Dictionary with unique keys from single frame
    nbins : int
        Number of histogram bins
    bmin : float
        Minimum bin value
    bmax : float
        Maximum bin value
    weights : Optional[torch.Tensor]
        Optional weights for histogram computation
    batch_size : int
        Number of structures in the batch
    """
    order = key_dict["order"]
    unique_keys_in_data = key_dict["unique_keys_in_data"]
    
    # Expand inverse indices for the batch
    inverse_indices_template = key_dict["inverse_indices"]
    if inverse_indices_template.numel() == 0:
        return {}
    else:
        repeat_factor = values.shape[0] // inverse_indices_template.shape[0]
        inverse_indices = inverse_indices_template.repeat(repeat_factor)
    
    histograms = {}
    if unique_keys_in_data.numel() == 0:
        return histograms
    
    n_unique_keys = unique_keys_in_data.shape[1]

    bins = torch.linspace(
        bmin, bmax, steps=nbins + 1, dtype=values.dtype, device=values.device
    )

    for idx in range(n_unique_keys):
        mask = inverse_indices == idx
        
        if not mask.any():
            continue

        val = values[mask]
        
        if isinstance(weights, torch.Tensor):
            # Weights are per structure, need to tile for all interactions
            n_atomgroups = int(val.shape[0] / weights.shape[0])
            hist = torchist.histogram(
                val, edges=bins, weight=weights.tile((n_atomgroups,))
            )
        else:
            hist = torchist.histogram(val, edges=bins)

        unique_key = unique_keys_in_data[:, idx]
        kk = tensor2tuple(unique_key)
        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kk] = hist.cpu().numpy()
        histograms[kf] = deepcopy(hist.cpu().numpy())

    return histograms

def compute_hist(
    values: torch.Tensor,
    atom_types: torch.Tensor,
    mapping: torch.Tensor,
    nbins: int,
    bmin: float,
    bmax: float,
    weights: Optional[torch.Tensor],
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.


    """
    unique_types = torch.unique(atom_types)
    order = mapping.shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    interaction_types = torch.vstack([atom_types[mapping[ii]] for ii in range(order)])

    interaction_types = _symmetrise_map[order](interaction_types)

    histograms = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [interaction_types[ii, :] == unique_key[ii] for ii in range(order)]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue
        bins = (
            torch.linspace(bmin, bmax, steps=nbins + 1).type(val.dtype).to(val.device)
        )
        if isinstance(weights, torch.Tensor):
            n_atomgroups = int(val.shape[0] / weights.shape[0])
            hist = torchist.histogram(
                val, edges=bins, weight=weights.tile((n_atomgroups,))
            )
        else:
            hist = torchist.histogram(val, edges=bins)
        kk = tensor2tuple(unique_key)
        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kk] = hist.cpu().numpy()
        histograms[kf] = deepcopy(hist.cpu().numpy())

    return histograms


def compute_hist_old(
    data: AtomicData,
    target: str,
    nbins: int,
    bmin: float,
    bmax: float,
    TargetPrior: _Prior,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.


    """
    if target_fit_kwargs == None:
        target_fit_kwargs = {}
    unique_types = torch.unique(data.atom_types)
    order = data.neighbor_list[target]["index_mapping"].shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    mapping = data.neighbor_list[target]["index_mapping"]
    values = TargetPrior.compute_features(data.pos, mapping)

    interaction_types = torch.vstack(
        [data.atom_types[mapping[ii]] for ii in range(order)]
    )

    interaction_types = _symmetrise_map[order](interaction_types)

    histograms = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [interaction_types[ii, :] == unique_key[ii] for ii in range(order)]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue

        hist = torch.histc(val, bins=nbins, min=bmin, max=bmax)

        kf = tensor2tuple(_flip_map[order](unique_key))
        histograms[kf] = hist

    return histograms
