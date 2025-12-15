from typing import Callable, Optional
from functools import partial
import mdtraj as md

from mlcg.nn.prior import (
    HarmonicBonds,
    HarmonicAngles,
    Dihedral,
    Repulsion,
    _Prior,
    GeneralBonds,
    GeneralAngles,
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
    ) -> None:
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=GeneralBonds,
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
        return GradientsOut(self.prior_cls(statistics, name=name), targets=targets)


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
    Builder for order-2 groups of nonbonded priors.

    Attributes
    ----------
    name:
        Name of specific prior (to match neighbour list name)
    nl_builder:
        Neighbour list class to be used in building neighbour list
    min_pair:
        Minimum number of bond edges between two atoms in order to be considered
        a member of the non-bonded set
    res_exclusion:
        If supplied, pairs within res_exclusion residues of each other are removed
        from the non-bonded set
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
        cutoff: Optional[float] = None,
    ) -> None:
        prior_fit_fn = partial(prior_fit_fn, percentile=percentile, cutoff=cutoff)
        super().__init__(
            histograms=HistogramsNL(
                n_bins=n_bins,
                bmin=bmin,
                bmax=bmax,
            ),
            nl_builder=nl_builder,
            prior_fit_fn=prior_fit_fn,
            prior_cls=Repulsion,
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
        prior = self.prior_cls(statistics)
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
