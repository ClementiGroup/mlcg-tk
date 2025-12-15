import torch
from typing import Dict, Optional
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import numpy as np


def repulsion(x, sigma):
    """Method defining the repulsion interaction"""
    rr = (sigma / x) * (sigma / x)
    return rr * rr * rr


def fit_repulsion_from_potential_estimates(
    bin_centers_nz: torch.Tensor, **kwargs
) -> Dict:
    r"""Method for fitting interaction parameters from data

    Parameters
    ----------
    bin_centers:
        Bin centers from a discrete histgram used to estimate the energy
        through logarithmic inversion of the associated Boltzmann factor
    dG_nz:
        The value of the energy :math:`U` as a function of the bin
        centers, as retrived via:

        ..math::

            U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}

        where :math:`\beta` is the inverse thermodynamic temperature and
        :math:`p(x)` is the normalized probability distribution of
        :math:`x`.


    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """

    delta = bin_centers_nz[1] - bin_centers_nz[0]
    sigma = bin_centers_nz[0] - 0.5 * delta
    stat = {"sigma": sigma}
    return stat


def fit_repulsion_from_values(
    bin_centers_nz: torch.Tensor,
    ncounts_nz: torch.Tensor,
    percentile: float,
    cutoff: Optional[float] = None,
    **kwargs
) -> Dict:
    """Method for fitting interaction parameters directly from input features

    Parameters
    ----------
    values:
        Input features as a tensor of shape (n_frames)
    percentile:
        If specified, the sigma value is calculated using the specified
        distance percentile (eg, percentile = 1) sets the sigma value
        at the location of the 1th percentile of pairwise distances. This
        option is useful for estimating repulsions for distance distribtions
        with long lower tails or lower distance outliers. Must be a number from
        0 to 1
    cutoff:
        If specified, only those input values below this cutoff will be used in
        evaluating the percentile

    Returns
    -------
    Dict:
        Dictionary of interaction parameters as retrived through
        `scipy.optimize.curve_fit`
    """
    values = np.repeat(bin_centers_nz.numpy(), ncounts_nz.int().numpy())
    if cutoff != None:
        values = values[values < cutoff]
    sigma = torch.tensor(np.percentile(values, percentile))
    stat = {"sigma": sigma}
    return stat
