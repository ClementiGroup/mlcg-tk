import torch
from typing import Dict, Optional
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import numpy as np
from .utils import neg_log_likelihood


def dihedral(
    theta: torch.Tensor,
    v_0: torch.Tensor,
    k1s: torch.Tensor,
    k2s: torch.Tensor,
) -> torch.Tensor:
    """Compute the dihedral interaction for a list of angles and models
    parameters. The ineraction is computed as a sin/cos basis expansion up
    to N basis functions.

    Parameters
    ----------
    theta :
        angles to compute the value of the dihedral interaction on
    v_0 :
        constant offset
    k1s :
        list of sin parameters
    k2s :
        list of cos parameters
    Returns
    -------
    torch.Tensor:
        Dihedral interaction energy
    """
    _, n_k = k1s.shape
    n_degs = torch.arange(1, n_k + 1, dtype=theta.dtype, device=theta.device)
    # expand the features w.r.t the mult integer so that it has the
    # shape of k1s and k2s
    angles = theta.view(-1, 1) * n_degs.view(1, -1)
    V = k1s * torch.sin(angles) + k2s * torch.cos(angles)
    # HOTFIX to avoid shape mismatch when using specialized priors
    # TODO: think of a better fix
    if v_0.ndim > 1:
        v_0 = v_0[:, 0]

    return V.sum(dim=1) + v_0


def dihedral_wrapper_fit_func(theta: torch.Tensor, *args) -> torch.Tensor:
    args = args[0]
    v_0 = torch.tensor(args[0])
    k_args = args[1:]
    num_ks = len(k_args) // 2
    k1s, k2s = k_args[:num_ks], k_args[num_ks:]
    k1s = torch.tensor(k1s).view(-1, num_ks)
    k2s = torch.tensor(k2s).view(-1, num_ks)
    return dihedral(theta, v_0, k1s, k2s)


def _init_parameters(n_degs):
    """Helper method for guessing initial parameter values"""
    p0 = [1.00]  # start with constant offset
    k1s_0 = [1 for _ in range(n_degs)]
    k2s_0 = [1 for _ in range(n_degs)]
    p0.extend(k1s_0)
    p0.extend(k2s_0)
    return p0


def _init_parameter_dict(n_degs):
    """Helper method for initializing the parameter dictionary"""
    stat = {"k1s": {}, "k2s": {}, "v_0": 0.00}
    k1_names = ["k1_" + str(ii) for ii in range(1, n_degs + 1)]
    k2_names = ["k2_" + str(ii) for ii in range(1, n_degs + 1)]
    for ii in range(n_degs):
        k1_name = k1_names[ii]
        k2_name = k2_names[ii]
        stat["k1s"][k1_name] = {}
        stat["k2s"][k2_name] = {}
    return stat


def _make_parameter_dict(stat, popt, n_degs):
    """Helper method for constructing a fitted parameter dictionary"""
    v_0 = popt[0]
    k_popt = popt[1:]
    num_k1s = int(len(k_popt) / 2)
    k1_names = sorted(list(stat["k1s"].keys()))
    k2_names = sorted(list(stat["k2s"].keys()))
    for ii in range(n_degs):
        k1_name = k1_names[ii]
        k2_name = k2_names[ii]
        stat["k1s"][k1_name] = {}
        stat["k2s"][k2_name] = {}
        if len(k_popt) > 2 * ii:
            stat["k1s"][k1_name] = k_popt[ii]
            stat["k2s"][k2_name] = k_popt[num_k1s + ii]
        else:
            stat["k1s"][k1_name] = 0
            stat["k2s"][k2_name] = 0
    stat["v_0"] = v_0
    return stat


def _compute_adjusted_R2(bin_centers_nz, dG_nz, mask, popt, free_parameters):
    """
    Method for model selection using adjusted R2
    Higher values imply better model selection
    """
    dG_fit = dihedral_wrapper_fit_func(bin_centers_nz[mask], *[popt])
    SSres = torch.sum(torch.square(dG_nz[mask] - dG_fit))
    SStot = torch.sum(torch.square(dG_nz[mask] - torch.mean(dG_nz[mask])))
    n_samples = len(dG_nz[mask])
    R2 = 1 - (SSres / (n_samples - free_parameters - 1)) / (SStot / (n_samples - 1))
    return R2


def _compute_aic(bin_centers_nz, dG_nz, mask, popt, free_parameters):
    """Method for computing the AIC"""
    aic = (
        2
        * neg_log_likelihood(
            dG_nz[mask],
            dihedral_wrapper_fit_func(bin_centers_nz[mask], *[popt]),
        )
        + 2 * free_parameters
    )
    return aic


def _linear_regression(bin_centers, targets, n_degs):
    """Vanilla linear regression"""
    features = [torch.ones_like(bin_centers)]
    for n in range(n_degs):
        features.append(torch.sin((n + 1) * bin_centers))
    for n in range(n_degs):
        features.append(torch.cos((n + 1) * bin_centers))
    features = torch.stack(features).t()
    targets = targets.to(features.dtype)
    sol = torch.linalg.lstsq(features, targets.t())
    return sol


def fit_dihedral_from_potential_estimates(
    bin_centers_nz: torch.Tensor,
    dG_nz: torch.Tensor,
    n_degs: int = 6,
    constrain_deg: Optional[int] = None,
    regression_method: str = "linear",
    metric: str = "aic",
    **kwargs,
) -> Dict:
    """
    Loop over n_degs basins and use either the AIC criterion
    or a prechosen degree to select best fit. Parameter fitting
    occurs over unmaksed regions of the free energy only.
    Parameters
    ----------
    bin_centers_nz:
        Bin centers over which the fit is carried out
    dG_nz:
        The emperical free energy correspinding to the bin centers
    n_degs:
        The maximum number of degrees to attempt to fit if using the AIC
        criterion for prior model selection
    constrain_deg:
        If not None, a single fit is produced for the specified integer
        degree instead of using the AIC criterion for fit selection between
        multiple degrees
    regression_method:
        String specifying which regression method to use. If "nonlinear",
        the default `scipy.optimize.curve_fit` method is used. If 'linear',
        linear regression via `torch.linalg.lstsq` is used
    metric:
        If a constrain deg is not specified, this string specifies whether to
        use either AIC ('aic') or adjusted R squared ('r2') for automated degree
        selection. If the automatic degree determination fails, users should
        consider searching for a proper constrained degree.

    Returns
    -------
    Dict:
        Statistics dictionary with fitted interaction parameters
    """

    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )

    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)

    if constrain_deg != None:
        assert isinstance(constrain_deg, int)
        stat = _init_parameter_dict(constrain_deg)
        if regression_method == "linear":
            popt = (
                _linear_regression(bin_centers_nz[mask], dG_nz[mask], constrain_deg)
                .solution.numpy()
                .tolist()
            )
        elif regression_method == "nonlinear":
            p0 = _init_parameters(constrain_deg)
            popt, _ = curve_fit(
                lambda theta, *p0: dihedral_wrapper_fit_func(theta, p0),
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=p0,
            )
        else:
            raise ValueError(
                "regression method {} is neither 'linear' nor 'nonlinear'".format(
                    regression_method
                )
            )
        stat = _make_parameter_dict(stat, popt, constrain_deg)

    else:
        if metric == "aic":
            metric_func = _compute_aic
            best_func = min
        elif metric == "r2":
            metric_func = _compute_adjusted_R2
            best_func = max
        else:
            raise ValueError("metric {} is neither 'aic' nor 'r2'".format(metric))

        # Determine best fit for unknown # of parameters
        stat = _init_parameter_dict(n_degs)
        popts = []
        metric_vals = []

        try:
            for deg in range(1, n_degs + 1):
                free_parameters = 1 + (2 * deg)
                if regression_method == "linear":
                    popt = (
                        _linear_regression(bin_centers_nz[mask], dG_nz[mask], deg)
                        .solution.numpy()
                        .tolist()
                    )
                elif regression_method == "nonlinear":
                    p0 = _init_parameters(deg)
                    popt, _ = curve_fit(
                        lambda theta, *p0: dihedral_wrapper_fit_func(theta, p0),
                        bin_centers_nz[mask],
                        dG_nz[mask],
                        p0=p0,
                    )
                else:
                    raise ValueError(
                        "regression method {} is neither 'linear' nor 'nonlinear'".format(
                            regression_method
                        )
                    )
                metric_val = metric_func(
                    bin_centers_nz, dG_nz, mask, popt, free_parameters
                )
                popts.append(popt)
                metric_vals.append(metric_val)
            best_val = best_func(metric_vals)
            best_i_val = metric_vals.index(best_val)
            popt = popts[best_i_val]
            stat = _make_parameter_dict(stat, popt, n_degs)
        except:
            print(f"failed to fit potential estimate for Dihedral")
            stat = _init_parameter_dict(n_degs)
            k1_names = sorted(list(stat["k1s"].keys()))
            k2_names = sorted(list(stat["k2s"].keys()))
            for ii in range(n_degs):
                k1_name = k1_names[ii]
                k2_name = k2_names[ii]
                stat["k1s"][k1_name] = torch.tensor(float("nan"))
                stat["k2s"][k2_name] = torch.tensor(float("nan"))
    return stat
