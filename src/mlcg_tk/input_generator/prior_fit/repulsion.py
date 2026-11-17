import torch
from typing import Callable, Dict, Optional, Tuple
import numpy as np

import pygad
from scipy.signal import savgol_filter
from mlcg_tk.prior_tools.utils import optimal_offset
import warnings

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
        centers, as retrieved via:

        :math:`U(x) = -\frac{1}{\beta}\log{ \left( p(x)\right)}`

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
    **kwargs,
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

def exp_repulsion(x, alpha, r_0):
    """Method defining the repulsion interaction"""
    rr = 1 - (x / r_0)
    return (6 / alpha) * np.exp( alpha * rr)

def interpolation_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        new_offspring = 0.5*(parent1+parent2)
        offspring.append(new_offspring)

        idx += 1

    return np.array(offspring)

# Savitzky-Golay smoothing parameters applied to the PMF before fitting
_SAVGOL_WINDOW_LENGTH = 5
_SAVGOL_POLYORDER = 3

# Genetic-algorithm hyperparameters
_GA_NUM_PARENTS_MATING = 20
_GA_SOL_PER_POP = 80
_GA_MUTATION_PERCENT_GENES = 50
_GA_PENALTY_FITNESS = -1e10
_GA_PERFECT_FITNESS = 1e10

_DEFAULT_R0_BOUNDS = (3.5, 5.5)


def _select_repulsive_region(
    bin_centers_nz: torch.Tensor,
    dG_nz: torch.Tensor,
    fit_window: Optional[float],
) -> torch.Tensor:
    """Return a mask selecting the short-range (repulsive) part of the PMF.

    A quartic is fitted to the PMF over the bins below ``fit_window`` (all
    bins if ``None``). The region ends at the second-smallest r among the
    quartic's local minima and inflection points, or at the smallest one if
    there is only one. If there are none, the ``fit_window`` mask is returned.

    Raises
    ------
    ValueError
        If fewer than 3 bins lie below ``fit_window``.

    Warns
    -----
    UserWarning
        If fewer than 5 bins lie below ``fit_window``; the quartic is then
        underdetermined and the region boundary is unreliable.
    """
    if fit_window is not None:
        mask = bin_centers_nz < fit_window
    else:
        mask = torch.ones_like(bin_centers_nz, dtype=torch.bool)

    num_bin_mask = int(mask.sum())

    if num_bin_mask < 3:
        raise ValueError(
            f"only {num_bin_mask} bins below fit_window={fit_window}; "
            "at least 3 are needed to locate the repulsive region"
        )
    if num_bin_mask < 5:
        warnings.warn(
            f"only {num_bin_mask} bins below fit_window={fit_window}; "
            "the quartic used to locate the repulsive region is underdetermined"
        )

    x_masked = bin_centers_nz[mask]
    coeffs = np.polyfit(x_masked, dG_nz[mask], deg=4)
    points = quartic_points_all(coeffs, x_masked[0], x_masked[-1])
    candidates = np.sort(
        [x for x, _ in points["minima"]] + [x for x, _ in points["inflections"]]
    )

    if len(candidates) == 0:
        return mask

    upper = candidates[1] if len(candidates) > 1 else candidates[0]

    return bin_centers_nz < upper


def fit_exp_repulsion_using_genetic_algorithm(
    bin_centers_nz: torch.Tensor,
    dG_nz: torch.Tensor,
    ncounts_nz: Optional[torch.Tensor] = None,
    fit_window: Optional[float] = None,
    repulsion_function: Callable[
        [np.ndarray, float, float], np.ndarray
    ] = exp_repulsion,
    alphas: Optional[Tuple[float, float]] = None,
    r0s: Optional[Tuple[float, float]] = None,
    iters: int = 500,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    r"""Fit an exponential repulsion to a PMF using a genetic algorithm.

    The PMF is shifted so that the median of its last five bins is zero and
    smoothed with a Savitzky-Golay filter. Its short-range repulsive region
    is then located (see :func:`_select_repulsive_region`), and the
    parameters ``(alpha, r_0)`` of ``repulsion_function`` are optimised over
    that region with PyGAD. The fitness is the inverse L2 distance to the
    PMF, after an optimal constant offset.

    The fit uses the plain (untruncated) ``repulsion_function``. When the
    result is used in ``CutoffExpRepulsion``, the deployed prior also has a
    shifted-force correction at the cutoff, which is not part of this fit.

    Parameters
    ----------
    bin_centers_nz:
        Bin centers with non-zero counts.
    dG_nz:
        PMF values :math:`U(x) = -\frac{1}{\beta}\log p(x)` at
        ``bin_centers_nz``.
    ncounts_nz:
        Unused. Kept so the signature stays compatible with the other prior
        fitting functions.
    fit_window:
        Only bins below this distance are used to locate the repulsive
        region; the GA then fits up to the detected region boundary
        (<= fit_window). If ``None``, all bins are used.
    repulsion_function:
        Model ``f(x, alpha, r_0)`` to fit. Defaults to :func:`exp_repulsion`.
    alphas:
        ``(low, high)`` search bounds for ``alpha``, with 0 < low < high.
        If ``None``, they are estimated from the data with
        :func:`estimate_alpha_bounds`.
    r0s:
        ``(low, high)`` search bounds for ``r_0``, in the units of
        ``bin_centers_nz``, with 0 < low < high. Defaults to ``(3.5, 5.5)``.
    iters:
        Number of GA generations.
    random_seed:
        Seed passed to PyGAD for reproducible fits. If ``None``, results can
        differ between runs.

    Returns
    -------
    Dict[str, float]
        ``{"alpha": ..., "r_0": ...}`` for the best solution found.

    Raises
    ------
    ValueError
        If there are fewer than 5 non-empty bins (needed for smoothing),
        fewer than 3 bins below ``fit_window``, fewer than 4 bins in the
        detected repulsive region, or if the ``alphas``/``r0s`` bounds are
        invalid.
    """

    if len(dG_nz) < _SAVGOL_WINDOW_LENGTH:
        raise ValueError(
            f"only {len(dG_nz)} non-empty bins; at least "
            f"{_SAVGOL_WINDOW_LENGTH} are needed for smoothing"
        )

    bin_centers_nz = bin_centers_nz.detach().cpu()
    dG_nz = dG_nz.detach().cpu()
    dG_shifted = dG_nz - dG_nz[-5:].median()
    dG_smooth = torch.tensor(
        savgol_filter(
            dG_shifted,
            window_length=_SAVGOL_WINDOW_LENGTH,
            polyorder=_SAVGOL_POLYORDER,
        )
    )

    region = _select_repulsive_region(bin_centers_nz, dG_smooth, fit_window)

    n_region = int(region.sum())
    if n_region < 4:
        raise ValueError(
            f"repulsive region has only {n_region} bins; at least 4 are "
            "needed to fit exp_repulsion (alpha, r_0 and an offset)"
        )

    xs = bin_centers_nz[region].numpy()
    ys = dG_smooth[region].numpy()

    def fitness_func(
        ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int
    ) -> float:
        pred = repulsion_function(xs, solution[0], solution[1])
        if not np.all(np.isfinite(pred)):
            return _GA_PENALTY_FITNESS
        offset = optimal_offset(pred, ys)
        residual = np.linalg.norm(ys - (pred - offset))
        if not np.isfinite(residual):
            return _GA_PENALTY_FITNESS
        if residual == 0:
            return _GA_PERFECT_FITNESS
        return 1.0 / residual

    r0_lo, r0_hi = r0s if r0s is not None else _DEFAULT_R0_BOUNDS
    if not 0 < r0_lo < r0_hi:
        raise ValueError(f"r0s must satisfy 0 < low < high, got {(r0_lo, r0_hi)}")

    if alphas is not None:
        alpha_lo, alpha_hi = alphas
    else:
        alpha_lo, alpha_hi = estimate_alpha_bounds(xs, ys, r0_lo, r0_hi)

    if not 0 < alpha_lo < alpha_hi:
        raise ValueError(f"alphas must satisfy 0 < low < high, got {(alpha_lo, alpha_hi)}")

    gene_space = [
        {"low": alpha_lo, "high": alpha_hi},
        {"low": r0_lo, "high": r0_hi},
    ]

    ga_instance = pygad.GA(
        num_generations=iters,
        num_parents_mating=_GA_NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=_GA_SOL_PER_POP,
        num_genes=2,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=0,
        crossover_type=interpolation_crossover,
        mutation_type="random",
        mutation_percent_genes=_GA_MUTATION_PERCENT_GENES,
        parallel_processing=None,
        random_seed=random_seed,
    )
    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()

    stat = {"alpha": float(solution[0]), "r_0": float(solution[1])}
    return stat


def quartic_points_all(coeffs, a, b, tol=1e-10):
    """Find the stationary and inflection points of a polynomial in [a, b].

    Written for the quartic used in :func:`_select_repulsive_region`, but
    works for a polynomial of any degree.

    Parameters
    ----------
    coeffs:
        Polynomial coefficients, highest power first, as returned by
        ``np.polyfit`` (for a quartic: ``[c4, c3, c2, c1, c0]`` with
        p(x) = c4 x^4 + c3 x^3 + c2 x^2 + c1 x + c0).
    a, b:
        Interval bounds. They are swapped if ``a > b``.
    tol:
        Tolerance used to treat a root as real (``|imag| <= tol``), to accept
        roots just outside [a, b], and to classify stationary points by the
        sign of p''.

    Returns
    -------
    dict
        Lists of ``(x, p(x))`` tuples, each sorted by x:

        - ``"minima"``: local minima (p' = 0, p'' > tol)
        - ``"maxima"``: local maxima (p' = 0, p'' < -tol)
        - ``"inflections"``: points where p'' = 0 and changes sign
        - ``"stationary_flat"``: stationary points with |p''| <= tol
          (degenerate; rare for quartics)

    Notes
    -----
    Repeated roots are merged after rounding to 14 decimals. An inflection
    point lying exactly on ``a`` or ``b`` is not reported, because the
    sign-change test needs p'' on both sides of it.
    """
    a, b = (a, b) if a <= b else (b, a)

    p = np.poly1d(coeffs)
    dp = p.deriv(1)
    ddp = p.deriv(2)

    def real_roots_in_interval(poly):
        roots = np.roots(poly)
        xs = []
        for r in roots:
            if abs(r.imag) <= tol:
                x = float(r.real)
                if a - tol <= x <= b + tol:
                    xs.append(x)
        # deduplicate (important if a root has multiplicity)
        xs = sorted(set(np.round(xs, 14)))
        return [float(x) for x in xs]

    # 1) All critical points (dp=0)
    crit_x = real_roots_in_interval(dp)

    minima = []
    maxima = []
    saddles = []  # degenerate stationary points if any

    for x in crit_x:
        y = float(p(x))
        s2 = float(ddp(x))
        if s2 > tol:
            minima.append((x, y))
        elif s2 < -tol:
            maxima.append((x, y))
        else:
            saddles.append((x, y))

    # 2) All inflection points (ddp=0, with concavity change check)
    infl_x = real_roots_in_interval(ddp)
    inflections = []
    for x in infl_x:
        # check sign change of ddp around x -> true inflection
        eps = max(1e-6, 1e-6 * (b - a))
        xl = max(a, x - eps)
        xr = min(b, x + eps)
        if float(ddp(xl)) * float(ddp(xr)) < 0:
            inflections.append((x, float(p(x))))

    return {
        "minima": minima,          # ALL local minima in [a,b]
        "maxima": maxima,          # ALL local maxima in [a,b]
        "inflections": inflections,
        "stationary_flat": saddles # rare for quartics but possible
    }

def estimate_alpha_bounds(
    xs: np.ndarray,
    ys: np.ndarray,
    r0_lo: float,
    r0_hi: float,
    factor: float = 3.0,
    alpha_min: float = 1.0,
    alpha_max: float = 30.0,
    min_points: int = 4,
) -> Tuple[float, float]:
    """Estimate a per-pair search window (alpha_lo, alpha_hi) for the GA.

    The genetic algorithm (GA) is used to fit the repulsion potentials.
    :func:`exp_repulsion` is log-linear in x:

    .. math::

        \\log y = \\log(6/\\alpha) + \\alpha (1 - x / r_0),

    so a linear fit of log(y) against x has slope -alpha / r_0. Taking r_0
    at the midpoint of ``[r0_lo, r0_hi]`` gives an estimate alpha_est, and
    the window ``[alpha_est / factor, alpha_est * factor]`` is returned,
    clipped to ``[alpha_min, alpha_max]``. This adapts the search range to
    each pair instead of using one fixed range for all pairs.

    The estimate ignores the constant offset that the GA fits, so it is only
    a rough guide; ``factor`` keeps the window wide enough to allow for that.

    Parameters
    ----------
    xs:
        Bin centers of the repulsive region.
    ys:
        Shifted, smoothed PMF at ``xs``. Only positive values are used, since
        log(y) is required.
    r0_lo, r0_hi:
        Search bounds for r_0; their midpoint is used to convert the slope
        into alpha.
    factor:
        Width of the window around alpha_est (multiplicative, > 1).
    alpha_min, alpha_max:
        Hard limits for the returned window, also used as the fallback.
    min_points:
        Minimum number of positive ``ys`` needed to attempt the estimate.

    Returns
    -------
    Tuple[float, float]
        ``(alpha_lo, alpha_hi)``. Falls back to ``(alpha_min, alpha_max)`` if
        fewer than ``min_points`` values of ``ys`` are positive, or if the
        estimate is not finite and positive.

    Notes
    -----
    If alpha_est / factor >= alpha_max, both bounds are clipped to
    ``alpha_max`` and the returned window has zero width.
    """
    mask = ys > 0
    if mask.sum() < min_points:
        return (alpha_min, alpha_max)

    slope, _ = np.polyfit(xs[mask], np.log(ys[mask]), deg=1)
    r0_mid = 0.5 * (r0_lo + r0_hi)
    alpha_est = -slope * r0_mid

    if not np.isfinite(alpha_est) or alpha_est <= 0:
        return (alpha_min, alpha_max)

    alpha_lo = float(np.clip(alpha_est / factor, alpha_min, alpha_max))
    alpha_hi = float(np.clip(alpha_est * factor, alpha_min, alpha_max))
    if alpha_hi <= alpha_lo:
        alpha_lo = max(alpha_hi / 1.5, alpha_min)

    return alpha_lo, alpha_hi