import numpy as np
import warnings
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import torch
from typing import Optional

from .utils import compute_aic


def polynomial(x: torch.Tensor, ks: torch.Tensor, V0: torch.Tensor):
    """Harmonic interaction in the form of a series. The shape of the tensors
        should match between each other.

    .. math:

        V(r) = V0 + \sum_{n=1}^{deg} k_n x^n

    """
    V = ks[0] * x
    for p, k in enumerate(ks[1:], start=2):
        V += k * torch.pow(x, p)
    V += V0
    return V


def polynomial_wrapper_fit_func(x, *args):
    args = args[0]
    V0 = torch.tensor(args[0])
    ks = torch.tensor(args[1:])
    return polynomial(x, ks, V0)


def _init_parameters(n_degs, bin_centers_nz=None, dG_nz=None):
    ks = [1.0 for _ in range(n_degs)]
    V0 = -1.0
    p0 = [V0]
    p0.extend(ks)
    return p0


def _init_parameter_dict(n_degs):
    """Helper method for initializing the parameter dictionary"""
    stat = {"ks": {}, "v_0": 0.0}
    k_names = ["k_" + str(ii) for ii in range(1, n_degs + 1)]
    for ii in range(n_degs):
        k_name = k_names[ii]
        stat["ks"][k_name] = {}
    return stat


def _make_parameter_dict(stat, popt, n_degs):
    """Helper method for constructing a fitted parameter dictionary"""
    stat["v_0"] = popt[0]
    k_names = sorted(list(stat["ks"].keys()))
    for ii in range(n_degs):
        k_name = k_names[ii]
        stat["ks"][k_name] = popt[ii + 1]

    return stat


def _linear_regression(xs: torch.Tensor, ys: torch.Tensor, n_degs: int):
    """Vanilla linear regression"""
    features = [torch.ones_like(xs)]
    for n in range(n_degs):
        features.append(torch.pow(xs, n + 1))
    features = torch.stack(features).t()
    ys = ys.to(features.dtype)
    sol = torch.linalg.lstsq(features, ys.t())
    return sol


def _numpy_fit(xs: torch.Tensor, ys: torch.Tensor, n_degs: int):
    """Regression through numpy"""
    sol = np.polynomial.Polynomial.fit(xs.numpy(), ys.numpy(), deg=n_degs)
    return sol


def _scipy_fit(
    xs: torch.Tensor, ys: torch.Tensor, n_degs: int, bounds: Optional = None
):
    """Regression through scipy

    The `bounds` argument is passed to `scipy.optimize.curve_fit` to ensure bounds in the polynomial
    """
    if bounds is None:
        bounds = (-np.inf, np.inf)
    p0 = _init_parameters(n_degs, xs, ys)
    p0[-2] = 0
    popt, _ = curve_fit(
        lambda theta, *p0: polynomial_wrapper_fit_func(theta, p0),
        xs,
        ys,
        p0=p0,
        bounds=bounds,
        ftol=1e-5,
    )
    return popt


def _polynomial_fit(
    xs: torch.Tensor, ys: torch.Tensor, n_degs: int, regression_method: str = "scipy"
):
    _valid_regression_methods = ["linear", "numpy", "scipy"]
    popt = []
    if regression_method == "linear":
        sol = _linear_regression(xs, ys, n_degs)
        popt = sol.solution.numpy().tolist()
    elif regression_method == "numpy":
        sol = _numpy_fit(xs, ys, n_degs)
        popt = list(sol.convert().coef)
    elif regression_method == "scipy":
        low_bounds = [-np.inf for _ in range(n_degs + 1)]
        # enforce that the coefficient of the leading degree is positive
        low_bounds[-1] = 0
        high_bounds = [np.inf for _ in range(n_degs + 1)]
        popt = _scipy_fit(xs, ys, n_degs, bounds=(low_bounds, high_bounds))
        dev = [idx * popt[idx] for idx in range(1, n_degs + 1)]
        dev_val_at_min_1 = polynomial_wrapper_fit_func(torch.tensor(-1), dev)
        dev_val_at_plus_1 = polynomial_wrapper_fit_func(torch.tensor(1), dev)
        if dev_val_at_min_1 * dev_val_at_plus_1 > 0:
            # relaunch the fit making sure that the second largest degree has no coefficient
            low_bounds[-2] = -1e-2
            high_bounds[-2] = 1e-2
            popt = _scipy_fit(xs, ys, n_degs, bounds=(low_bounds, high_bounds))
    else:
        raise ValueError(
            f"regression method {regression_method} is not in {_valid_regression_methods} "
        )
    if popt[-1] <= 0:
        print()
    return popt


def fit_polynomial_from_potential_estimates(
    bin_centers_nz: torch.Tensor,
    dG_nz: torch.Tensor,
    n_degs: int = 4,
    constrain_deg: Optional[int] = 4,
    regression_method: str = "scipy",
    **kwargs,
):
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

    Returns
    -------
        Statistics dictionary with fitted interaction parameters
    """

    integral = torch.tensor(
        float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
    )
    mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
    if constrain_deg is not None:
        stat = _init_parameter_dict(n_degs)
        popt = _polynomial_fit(
            bin_centers_nz[mask], dG_nz[mask], n_degs, regression_method
        )
        stat = _make_parameter_dict(stat, popt, n_degs)
    else:
        try:
            # Determine best fit for unknown # of parameters
            stat = _init_parameter_dict(n_degs)
            popts = []
            aics = []
            degs = range(2, n_degs + 1, 2)
            for deg in degs:
                popt = _polynomial_fit(
                    bin_centers_nz[mask], dG_nz[mask], n_degs, regression_method
                )
                fitted_dG = polynomial_wrapper_fit_func(bin_centers_nz[mask], popt)
                free_parameters = deg + 1
                aic = compute_aic(fitted_dG, dG_nz[mask], free_parameters)
                popts.append(popt)
                aics.append(aic)
            # Select only priors that have upward curve at both regions of unexplored
            # phase space. i.e. (powers of 2)
            min_aic = min(aics)
            min_i_aic = aics.index(min_aic)
            popt = popts[min_i_aic]
            stat = _make_parameter_dict(stat, popt, n_degs)
        except RuntimeError:
            print("failed to fit potential estimate for Polynomial")
            stat = _init_parameter_dict(n_degs)
            k_names = sorted(list(stat["ks"].keys()))
            x_0_names = sorted(list(stat["x_0s"].keys()))
            for ii in range(n_degs):
                k1_name = k_names[ii]
                k2_name = x_0_names[ii]
                stat["ks"][k1_name] = torch.tensor(float("nan"))
                stat["x_0s"][k2_name] = torch.tensor(float("nan"))
    return stat
