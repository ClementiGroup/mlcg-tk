from typing import Callable, Optional, Tuple
from functools import partial
import mdtraj as md
import warnings

from mlcg.nn.prior import (
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
    Repulsion,
    _Prior,
    GeneralBonds,
    GeneralAngles,
    ExpRepulsion,
    CutoffExpRepulsion
)

from mlcg.nn.gradients import GradientsOut
from mlcg.data import AtomicData
from .prior_fit.histogram import HistogramsNL


class PriorBuilder:
    """
    General prior builder object holding statistics computed for a given prior
    feature and functions that are used to build neighbour lists and fit potentials
    to the computed statistics.

    Attributes
    ----------
    histograms:
        HistogramsNL object for storing binned feature statistics
    nl_builder:
        Neighbour list class to be used in building neighbour list
    prior_fit_fn:
        Function to be used in fitting potential from statistics
    prior_cls:
        Prior class for fitting features
    """

    def __init__(
        self,
        histograms: HistogramsNL,
        nl_builder: Callable,
        prior_fit_fn: Callable,
        prior_cls: _Prior,
    ) -> None:
        self.histograms = histograms
        self.prior_fit_fn = prior_fit_fn
        self.nl_builder = nl_builder
        self.prior_cls = prior_cls

    def build_nl(
        self,
        topology: md.Topology,
        **kwargs,
    ):
        """
        Generates tagged and ordered edges using neighbour list builder function

        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created

        Returns
        -------
        Edges, orders, and tag for given prior term
        """
        return self.nl_builder(topology=topology)

    def accumulate_statistics(
        self, nl_name: str, data: AtomicData, key_dict: dict
    ) -> None:
        """
        Computes atom-type specific features and calculates statistics from a collated
        AtomicData stucture

        Parameters
        ----------
        nl_name:
            Neighbour list tag
        data:
            Collated list of individual AtomicData structures.
        """
        mapping = data.neighbor_list[nl_name]["index_mapping"]
        values = self.prior_cls.compute_features(data.pos, mapping)
        if hasattr(data, "weights"):
            weights = data.weights
        else:
            weights = None
        self.histograms.accumulate_statistics(nl_name, values, key_dict, weights)


class Bonds(PriorBuilder):
    """
    Builder for order-2 groups of bond priors.

    Attributes
    ----------
    name:
        Name of specific prior (to match neighbour list name)
    nl_builder:
        Neighbour list class to be used in building neighbour list
    separate_termini:
        Whether statistics should be computed separately for terminal atoms
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin:
        Lower bound of bin edges
    bmax:
        Upper bound of bin edges
    prior_fit_fn:
        Function to be used in fitting potential from statistics
    """

    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
        prior_cls = GeneralBonds,
    ) -> None:

        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=prior_cls,
        )
        self.name = name
        self.type = "bonds"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        """
        Generates edges for order-2 atom groups for bond prior

        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created

        Returns
        -------
        Edges, orders, and tag for angle prior term
        """
        return self.nl_builder(
            topology=topology,
            separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        """
        Parameters
        ----------
        statistics:
            Gathered bond statistics
        name: str
            Name of the prior object (corresponding to nls name)
        targets:
            The gradient targets to produce from a model output. These can be any
            of the gradient properties referenced in `mlcg.data._keys`.
            At the moment only forces are implemented.
        """
        stat0 = next(iter(statistics.values()))
        if "ks" not in stat0:
            return GradientsOut(self.prior_cls(statistics, name=name), targets=targets)
        else:    
            prior = self.prior_cls(statistics, n_degs=len(stat0["ks"]))
            prior.name = name
            return GradientsOut(prior, targets=targets)



class Angles(PriorBuilder):
    """
    Builder for order-3 groups of angle priors.

    Attributes
    ----------
    name:
        Name of specific prior (to match neighbour list name)
    nl_builder:
        Neighbour list class to be used in building neighbour list
    separate_termini:
        Whether statistics should be computed separately for terminal atoms
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin:
        Lower bound of bin edges
    bmax:
        Upper bound of bin edges
    prior_fit_fn:
        Function to be used in fitting potential from statistics
    prior_cls:
        Prior class to be used. It must be able to be initialized from the output
        of the `prior_fit_fn`
    """

    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
        prior_cls=GeneralAngles,
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=prior_cls,
        )
        self.name = name
        self.type = "angles"
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        """
        Generates edges for order-3 atom groups for angle prior

        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created

        Returns
        -------
        Edges, orders, and tag for angle prior term
        """
        return self.nl_builder(
            topology=topology,
            separate_termini=self.separate_termini,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        """
        Parameters
        ----------
        statistics:
             Gathered angle statistics
        name: str
            Name of the prior object (corresponding to nls name)
        targets:
            The gradient targets to produce from a model output. These can be any
            of the gradient properties referenced in `mlcg.data._keys`.
            At the moment only forces are implemented.
        """
        return GradientsOut(self.prior_cls(statistics, name=name), targets=targets)


class NonBonded(PriorBuilder):
    """
    Builder for order-2 groups of nonbonded (repulsive) priors.

    Collects pair-distance histograms for the non-bonded neighbour list, fits
    a repulsion to each pair type with ``prior_fit_fn``, and builds the
    corresponding ``prior_cls`` module.

    Parameters
    ----------
    name:
        Name of the prior (must match the neighbour list name).
    nl_builder:
        Neighbour list class used to build the non-bonded neighbour list.
    min_pair:
        Minimum number of bond edges between two atoms for the pair to be
        included in the non-bonded set.
    res_exclusion:
        Pairs within ``res_exclusion`` residues of each other are excluded
        from the non-bonded set.
    separate_termini:
        Whether statistics are computed separately for terminal atoms.
    n_bins:
        Number of bins of the 1-D pair-distance histograms.
    bmin:
        Lower bound of the histogram bin edges.
    bmax:
        Upper bound of the histogram bin edges. Distances above ``bmax`` are
        not seen by the fit.
    prior_fit_fn:
        Function used to fit the potential from the histograms, e.g.
        ``fit_repulsion_from_values`` for Repulsion priors or
        ``fit_exp_repulsion_using_genetic_algorithm`` for ExpRepulsion priors.
        It must accept the keyword arguments this builder passes for the
        chosen ``prior_cls`` (see below).
    percentile:
        Repulsion priors only. Sigma is set at this percentile of the
        pair-distance distribution (``np.percentile`` convention, 0 to 100;
        e.g. 1 = 1st percentile). Useful for distributions with long lower
        tails or short-distance outliers.
    prior_cls:
        Prior class to build: ``Repulsion`` or ``ExpRepulsion``, or a subclass
        such as ``CutoffExpRepulsion``.
    cutoff:
        For ``CutoffExpRepulsion`` (required): truncation radius of the
        prior, which is shifted so that energy and force vanish at the cutoff.
        For Repulsion priors: only distances below the cutoff are used to
        compute the percentile. Not used to fit ExpRepulsion priors (see
        ``fit_window``).
    fit_window:
        ExpRepulsion priors only. Bins below this distance are used to
        locate the repulsive region; the fit then runs up to the detected
        region boundary (<= fit_window). If None, all bins (up to ``bmax``)
        are used. A warning is issued if it exceeds ``cutoff``.
    alphas:
        ExpRepulsion priors only. ``(low, high)`` search bounds for alpha.
        If None, they are estimated from the data per pair type.
    r0s:
        ExpRepulsion priors only. ``(low, high)`` search bounds for r_0, in
        the distance units of the histograms. If None, the fit function's
        default is used.
    random_seed:
        ExpRepulsion priors only. Seed for the genetic-algorithm fit, for
        reproducible priors. If None, results can differ between runs.

    Raises
    ------
    ValueError
        If ``prior_cls`` is not a Repulsion or ExpRepulsion subclass, or if
        ``prior_cls`` is ``CutoffExpRepulsion`` and ``cutoff`` is None.
    """
    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        min_pair: int,
        res_exclusion: int,
        separate_termini: bool,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
        percentile: float = 1,
        prior_cls = Repulsion,
        cutoff: Optional[float] = None,
        fit_window: Optional[float] = None,
        alphas: Optional[Tuple[float, float]] = None,
        r0s: Optional[Tuple[float, float]] = None,
        random_seed: Optional[int] = None,    
    ) -> None:
        if issubclass(prior_cls, Repulsion):
            prior_fit_fn = partial(prior_fit_fn, percentile=percentile, cutoff=cutoff)
        elif issubclass(prior_cls, ExpRepulsion):                
            prior_fit_fn = partial(prior_fit_fn, fit_window=fit_window, alphas=alphas, r0s=r0s, random_seed=random_seed)
        else:
            raise ValueError(f"prior_cls must be either Repulsion or ExpRepulsion, got {prior_cls}")
        
        if issubclass(prior_cls, CutoffExpRepulsion):
            if cutoff is None:
                raise ValueError("CutoffExpRepulsion requires a cutoff")
            if fit_window is not None and fit_window > cutoff:
                warnings.warn(
                    f"fit_window ({fit_window}) > cutoff ({cutoff}): the fitted region "
                    "may extend beyond where the prior is truncated to zero"
                )
            self.prior_kwargs = {"cutoff": cutoff}
        else:
            self.prior_kwargs = {}

        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=prior_cls,
        )
        self.name = name
        self.type = "non_bonded"
        self.min_pair = min_pair
        self.res_exclusion = res_exclusion
        self.separate_termini = separate_termini
        # if separate_termini == True then these will be set in get_terminal_atoms
        self.n_term_atoms = None
        self.c_term_atoms = None
        self.n_atoms = None
        self.c_atoms = None

    def build_nl(self, topology, **kwargs):
        """
        Generates edges for order-2 atom groups for nonbond prior

        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created
        kwargs:
            bond_edges:
                Edges of bonded prior, to be omitted from nonbonded interactions
            angle_edges:
                Edges of angle prior, to be omitted from nonbonded interactions

        Returns
        -------
        Edges, orders, and tag for nonbonded prior term
        """
        bond_edges = kwargs["bond_edges"]
        angle_edges = kwargs["angle_edges"]
        return self.nl_builder(
            topology=topology,
            bond_edges=bond_edges,
            angle_edges=angle_edges,
            separate_termini=self.separate_termini,
            min_pair=self.min_pair,
            res_exclusion=self.res_exclusion,
            n_term_atoms=self.n_term_atoms,
            c_term_atoms=self.c_term_atoms,
            n_atoms=self.n_atoms,
            c_atoms=self.c_atoms,
        )

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        """
        Parameters
        ----------
        statistics:
             Gathered nonbonded statistics
        name: str
            Name of the prior object (corresponding to nls name)
        targets:
            The gradient targets to produce from a model output. These can be any
            of the gradient properties referenced in `mlcg.data._keys`.
            At the moment only forces are implemented.
        """
        prior = self.prior_cls(statistics, **self.prior_kwargs)
        prior.name = name
        return GradientsOut(prior, targets=targets)


class Dihedrals(PriorBuilder):
    """
    Builder for order-4 groups of dihedral priors.

    Attributes
    ----------
    name:
        Name of specific prior (to match neighbour list name)
    nl_builder:
        Neighbour list class to be used in building neighbour list
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    bmin:
        Lower bound of bin edges
    bmax:
        Upper bound of bin edges
    prior_fit_fn:
        Function to be used in fitting potential from statistics
    """

    def __init__(
        self,
        name: str,
        nl_builder: Callable,
        n_bins: int,
        bmin: float,
        bmax: float,
        prior_fit_fn: Callable,
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=Dihedral,
        )
        self.name = name
        self.type = "dihedrals"

    def get_prior_model(self, statistics, name, targets="forces", **kwargs):
        """
        Parameters
        ----------
        statistics:
             Gathered dihedral statistics
        name: str
            Name of the prior object (corresponding to nls name)
        targets:
            The gradient targets to produce from a model output. These can be any
            of the gradient properties referenced in `mlcg.data._keys`.
            At the moment only forces are implemented.
        kwargs:
            n_degs:
                The maximum number of degrees to attempt to fit if using the AIC
                criterion for prior model selection
        """
        prior = self.prior_cls(statistics, n_degs=kwargs["n_degs"])
        prior.name = name
        return GradientsOut(prior, targets=targets)
