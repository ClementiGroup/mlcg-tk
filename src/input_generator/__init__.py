import warnings
from mlcg_tk import input_generator as _new

warnings.simplefilter("once", DeprecationWarning)

warnings.warn(
    "You are importing 'input_generator' directly. "
    "This is deprecated and will be removed in a future release. "
    "Please update your imports to use 'mlcg_tk.input_generator'. "
    "To update old pickle files, just save them again once loaded.",
    DeprecationWarning,
    stacklevel=2,
)

# re-export everything
__all__ = getattr(_new, "__all__", None)
for name in dir(_new):
    if not name.startswith("_"):
        globals()[name] = getattr(_new, name)

from mlcg_tk.input_generator.raw_dataset import RawDataset, SampleCollection
from mlcg_tk.input_generator.raw_data_loader import (
    DatasetLoader,
    CATH_loader,
    CATH_ext_loader,
    DIMER_loader,
    DIMER_ext_loader,
    Villin_loader,
    Trpcage_loader,
    Cln_loader,
    BBA_loader,
    ProteinG_loader,
    A3D_loader,
    OPEP_loader,
    NTL9_loader,
    HDF5_loader,
)


from mlcg_tk.input_generator.embedding_maps import (
    CGEmbeddingMap,
    CGEmbeddingMapFiveBead,
    CGEmbeddingMapCA,
    embedding_fivebead,
    embedding_ca,
)


from mlcg_tk.input_generator.prior_nls import (
    StandardBonds,
    StandardAngles,
    Non_Bonded,
    Phi,
    Psi,
    Omega,
    Gamma1,
    Gamma2,
    CA_pseudo_dihedral,
)

from mlcg_tk.input_generator.prior_fit import fit_harmonic_from_potential_estimates, harmonic
from mlcg_tk.input_generator.prior_fit import (
    fit_repulsion_from_potential_estimates,
    fit_repulsion_from_values,
    repulsion,
)
from mlcg_tk.input_generator.prior_fit.fit_potentials import fit_potentials
from mlcg_tk.input_generator.prior_fit import fit_dihedral_from_potential_estimates, dihedral

from mlcg_tk.input_generator.prior_gen import Bonds, Angles, NonBonded, Dihedrals
