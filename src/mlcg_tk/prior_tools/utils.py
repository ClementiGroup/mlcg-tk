from itertools import combinations_with_replacement
import numpy as np
import torch
from typing import List, Tuple

from mlcg.nn.prior import _Prior, Harmonic, Repulsion, Dihedral


def symmetrized_keys_generator(order: int, emb_max: int = 20) -> List[Tuple]:
    r"""
    Auxiliar function to generate the symmetric tuples of size `order` with integers from 1 to `emb_max`
    """
    lim = emb_max + 1
    if order == 1:
        return [(i) for i in range(1, lim)]
    elif order == 2:
        return list(combinations_with_replacement(range(1, lim), 2))
    elif order == 3:
        aux = symmetrized_keys_generator(order=2, emb_max=emb_max)
        fin_list = [(arr[0], i, arr[1]) for i in range(1, lim) for arr in aux]
        return fin_list
    elif order == 4:
        aux = symmetrized_keys_generator(order=2, emb_max=emb_max)
        fin_list = [
            (arr1[0], arr2[0], arr2[1], arr1[1]) for arr2 in aux for arr1 in aux
        ]
        return fin_list
    else:
        raise ValueError(f"Not implemente for order {order}")

def get_nonzero_keys(prior_module: _Prior) -> torch.Tensor:
    r"""
    Function to extract the key combinations of a prior that have a non-zero value for the parameters.
    """
    if issubclass(type(prior_module), Harmonic):
        keys = prior_module.k.nonzero()
    elif isinstance(prior_module, Repulsion):
        keys = prior_module.sigma.nonzero()
    elif issubclass(type(prior_module), Dihedral):
        keys = prior_module.v_0.nonzero()
    else:
        raise ValueError(f"Prior of type {prior_module.__class__} not supported")
    return keys


def optimal_offset(fit_arr: np.ndarray, data_arr: np.ndarray) -> float:
    r"""Find optimal offset such that the difference between arrays is minimized.

    This functions returns the solution to the optimization problem of
    minimizing:

    \min_{\lambda \in R} \sum_{k=1}^{n} (data_arr[k]-fit_arr[k] + \lambda)

    This is useful to plot two curves as overlapping as possible.
    """
    if len(data_arr) != len(fit_arr):
        raise ValueError("Arrays should be of the same length")
    mask = ~np.isinf(data_arr)
    diff = fit_arr[mask] - data_arr[mask]
    return np.sum(diff) / np.sum(mask)


def get_prior_domain(name: str, n=201) -> torch.Tensor:
    r"""
    Function to return a tensor with the domain where some common priors are defined
    """
    if "angles" in name:
        # we use the cosine to parametrize the angles
        a, b = -1.1, 1.1
    elif "bonds" in name:
        # usual range for CA-bonds
        a, b = 3.4, 4.4
    elif name == "non_bonded" or name == "repulsion":
        # standard range for an excluded value
        a, b = 0.1, 7
    elif "dihedral" in name:
        # dihedrals can range
        a, b = -torch.pi, torch.pi
    return torch.linspace(a, b, n)


def prior_evaluator(prior_module: _Prior, key: Tuple, x: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluate the `prior_module` for bead combination `key` over tensor `x`
    """
    if issubclass(type(prior_module), Harmonic):
        x_0 = prior_module.x_0[key].item()
        k = prior_module.k[key].item()
        res = prior_module.compute(x, x_0, k)
    elif isinstance(prior_module, Repulsion):
        sigma = prior_module.sigma[key[0], key[1]].item()
        res = prior_module.compute(x, sigma)
    elif issubclass(type(prior_module), Dihedral):
        v_0 = prior_module.v_0[key]
        k1s = [prior_module.k1s[idx][key] for idx in range(prior_module.n_degs)]
        k1s = torch.tensor(k1s).view(1, -1)
        k2s = [prior_module.k2s[idx][key] for idx in range(prior_module.n_degs)]
        k2s = torch.tensor(k2s).view(1, -1)
        res = prior_module.compute(x.view(1, -1), v_0, k1s, k2s)
    else:
        raise ValueError(f"Prior of type {prior_module.__class__} is not supported")
    return res
