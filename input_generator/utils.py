import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import mdtraj as md
import warnings
from functools import wraps

from aggforce import (LinearMap, 
                    guess_pairwise_constraints, 
                    project_forces, 
                    constraint_aware_uni_map,
                    qp_linear_map
                    )


from .prior_gen import PriorBuilder


def with_attrs(**func_attrs):
    """Set attributes in the decorated function, at definition time.
    Only accepts keyword arguments.
    """

    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator


def get_output_tag(
        tag_label: Union[List,str], placement: str="before"
):
    """
    Helper function for combining output tag labels neatly.
    Fixes issues of connecting/preceding '_' being included in some labels but not others.

    Parameters
    ----------
    tag_label : List, str
        Either a list of labels to include (ex: for datasets, delta force computation) or individual label item.
    placement : str
        Placement of tag in output name. One of: 'before', 'after'.
    """

    if isinstance(tag_label, str):
        if tag_label in [None, "", " "]:
            return ""
        else:
            return f"_{tag_label.strip('_')}"
    elif isinstance(tag_label, List):
        for l in tag_label:
            if l in [None, "", " "]:
                tag_label.remove(l)
        joined_label = "_".join([l.strip('_') for l in tag_label])
        if placement == "before":
            return f"{joined_label}_"
        elif placement == "after":
            return f"_{joined_label}"
        else:
            raise ValueError("Please specify placement from: 'before', 'after'.")


def map_cg_topology(
    atom_df: pd.DataFrame,
    cg_atoms: List[str],
    embedding_function: str,
    skip_residues: Optional[Union[List, str]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    atom_df:
        Pandas DataFrame row from mdTraj topology.
    cg_atoms:
        List of atoms needed in CG mapping.
    embedding_function:
        Function that slices coodinates, if not provided will fail.
    special_typing:
        Optional dictionary of alternative atom properties to use in assigning types instead of atom names.
    skip_residues:
        Optional list of residues to skip when assigning CG atoms (can be used to skip caps for example);
        As of now, this skips all instances of a given residue.


    Returns
    -------
    New DataFrame columns indicating atom involvement in CG mapping and type assignment.

    Example
    -------
    First obtain a Pandas DataFrame object using the built-in MDTraj function:
    >>> top_df = aa_traj.topology.to_dataframe()[0]

    For a five-bead resolution mapping without including caps:
    >>> cg_atoms = ["N", "CA", "CB", "C", "O"]
    >>> embedding_function = embedding_fivebead
    >>> skip_residues = ["ACE", "NME"]

    Apply row-wise function:
    >>> top_df = top_df.apply(map_cg_topology, axis=1, cg_atoms, embedding_dict, skip_residues)
    """
    if isinstance(embedding_function, str):
        try:
            embedding_function = eval(embedding_function)
        except NameError:
            print("The specified embedding function has not been defined.")
            exit
    name, res = atom_df["name"], atom_df["resName"]
    if skip_residues != None and res in skip_residues:
        atom_df["mapped"] = False
        atom_df["type"] = "NA"

    else:
        if name in cg_atoms:
            atom_df["mapped"] = True
            atom_type = embedding_function(atom_df)
            atom_df["type"] = atom_type
        else:
            atom_df["mapped"] = False
            atom_df["type"] = "NA"

    return atom_df


def batch_matmul(map_matrix, X, batch_size):
    """
    Perform matrix multiplication in chunks.
    
    Parameters:
      map_matrix: np.ndarray of shape (N_CG_ats, N_FG_ats)
      X: np.ndarray of shape (M_frames, N_FG_ats, 3)
      batch_size: int, the number of rows (from the M dimension) to process at a time.
    
    Returns:
      result: np.ndarray of shape (M_frames, N_CG_ats, 3)
    """
    results = []
    M = X.shape[0]
    for i in range(0, M, batch_size):
        # Slice a batch along the M dimension
        X_batch = X[i:i+batch_size]  # shape: (batch, N, 3)
        # Perform matrix multiplication:
        # map_matrix (CG, FG) multiplied by each X_batch (FG, 3) gives (GC, 3) for each sample.
        # The broadcasting ensures the result is (batch, CG, 3)
        result_batch = map_matrix @ X_batch  
        results.append(result_batch)
    # Concatenate all chunks along the first axis (M dimension)
    return np.concatenate(results, axis=0)


def slice_coord_forces(
    coords, forces, cg_map, mapping: str = "slice_aggregate", force_stride: int = 100, batch_size: Optional[int] = None
) -> Tuple:
    """
    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Numpy array of atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Numpy array of atomistic forces
    cg_map: [n_cg_atoms, n_atomistic_atoms]
        Linear map characterizing the atomistic to CG configurational map with shape.
    mapping:
        Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize'.
    force_stride:
        Striding to use for force projection results
    batch_size:
        Optional length of batch in which divide the AA mapping of coords and forces 
        to CG ones

    Returns
    -------
    Coarse-grained coordinates and forces
    """
    config_map = LinearMap(cg_map)
    config_map_matrix = config_map.standard_matrix
    # taking only first 100 frames gives same results in ~1/15th of time
    constraints = guess_pairwise_constraints(coords[:100], threshold=5e-3)
    if mapping == "slice_aggregate":
        method = constraint_aware_uni_map
        force_agg_results = project_forces(
            coords=coords[::force_stride],
            forces=forces[::force_stride],
            coord_map=config_map,
            constrained_inds=constraints,
            method=method,
        )
    elif mapping == "slice_optimize":
        method = qp_linear_map
        l2 = 1e3
        force_agg_results = project_forces(
            coords=coords[::force_stride],
            forces=forces[::force_stride],
            coord_map=config_map,
            constrained_inds=constraints,
            method=method,
            l2_regularization=l2,
        )
    else:
        raise RuntimeError(
            f"Force mapping {mapping} is neither 'slice_aggregate' nor 'slice_optimize'."
        )
    force_map_matrix = force_agg_results["tmap"].force_map.standard_matrix

    if batch_size != None: 
        cg_coords = batch_matmul(config_map_matrix, coords, batch_size=batch_size)
        cg_forces = batch_matmul(force_map_matrix, forces, batch_size=batch_size)
    else:
        cg_coords = config_map_matrix @ coords
        cg_forces = force_map_matrix @ forces

    return cg_coords, cg_forces, force_map_matrix

def filter_cis_frames(
        coords: np.ndarray,
        forces: np.ndarray,
        topology: md.Topology,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        filters out frames containing cis-omega angles

        Parameters
        ----------
        coords: [n_frames, n_atoms, 3]
            Non-filtered atomistic coordinates
        forces: [n_frames, n_atoms, 3]
            Non-filtered atomistic forces
        topology: 
            mdtraj topology to load the coordinates with
        verbose:
            If True, will print a warning containing the number of discarded frames for this sample

        Returns
        -------
        Tuple of np.ndarray's for filtered coarse grained coordinates and forces
        """
        min_omega_atoms = set(["N", "CA", "C"])
        unique_atom_types = set([atom.name for atom in topology.atoms])
        if not min_omega_atoms.issubset(unique_atom_types):
            raise ValueError("Provided pdb file must contain at least N, CA and C atoms for cis-omega filtering")
        
        cis_omega_mask = np.zeros(coords.shape[0], dtype=bool)
        md_traj = md.Trajectory(coords, topology)

        omega_idx, omega_values = md.compute_omega(md_traj)

        cis_omega_threshold = 1.0 #rad
        mask = np.all(np.abs(omega_values) > 1, axis=1)
        if not np.all(mask):
            warnings.warn(f"Discarding {len(mask) - np.sum(mask)} cis frames")
        if np.sum(mask) == 0:
            warnings.warn(f"This amounts to removing all frames for this molecule")

        return  coords[mask], forces[mask]

def get_terminal_atoms(
    prior_builder: PriorBuilder,
    cg_dataframe: pd.DataFrame,
    N_term: Union[None, str] = None,
    C_term: Union[None, str] = None,
) -> Dict:
    """
    Parameters
    ----------
    prior_builder:

    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    N_term: (Optional)
        Atom used in definition of N-terminus embedding.
    C_term: (Optional)
        Atom used in definition of C-terminus embedding.
    """
    chains = cg_dataframe.chainID.unique()
    # all atoms belonging to monopeptide chains will be removed from termini list
    monopeptide_atoms = []
    for chain in chains:
        residues = cg_dataframe.loc[cg_dataframe.chainID == chain].resSeq.unique()
        if len(residues) == 1:
            monopeptide_atoms.extend(
                cg_dataframe.loc[cg_dataframe.chainID == chain].index.to_list()
            )

    n_term_atoms = []
    c_term_atoms = []

    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain) & chain_filter
            ].index.to_list()
        )
        c_term_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain) & chain_filter
            ].index.to_list()
        )

    prior_builder.n_term_atoms = [a for a in n_term_atoms if a not in monopeptide_atoms]
    prior_builder.c_term_atoms = [a for a in c_term_atoms if a not in monopeptide_atoms]

    N_term_name = "N" if N_term is None else N_term
    C_term_name = "C" if C_term is None else C_term

    n_term_name_atoms = []
    c_term_name_atoms = []
    for chain in chains:
        chain_filter = cg_dataframe["chainID"] == chain
        first_res_chain, last_res_chain = (
            cg_dataframe[chain_filter]["resSeq"].min(),
            cg_dataframe[chain_filter]["resSeq"].max(),
        )
        n_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == first_res_chain)
                & (cg_dataframe["name"] == N_term_name)
                & chain_filter
            ].index.to_list()
        )

        c_term_name_atoms.extend(
            cg_dataframe.loc[
                (cg_dataframe["resSeq"] == last_res_chain)
                & (cg_dataframe["name"] == C_term_name)
                & chain_filter
            ].index.to_list()
        )

    prior_builder.n_atoms = n_term_name_atoms
    prior_builder.c_atoms = c_term_name_atoms

    return prior_builder


def get_edges_and_orders(
    prior_builders: List[PriorBuilder],
    topology: md.Topology,
) -> List:
    """
    Parameters
    ----------
    prior_builders:
        List of PriorBuilder's to use for defining neighbour lists
    topology:
        MDTraj topology object from which atom groups defining each prior term will be created.
    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).

    Returns
    -------
    List of edges, orders, and tag for each prior term specified in prior_dict.
    """
    all_edges_and_orders = []
    # process bond priors
    bond_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "bonds"
    ]
    all_bond_edges = []
    for prior_builder in bond_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_bond_edges.extend([p[2] for p in edges_and_orders])
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_bond_edges.append(edges_and_orders[2])

    # process angle priors
    angle_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "angles"
    ]
    all_angle_edges = []
    for prior_builder in angle_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_angle_edges.extend([p[2] for p in edges_and_orders])
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_angle_edges.append(edges_and_orders[2])

    # get nonbonded priors using bonded and angle edges
    if len(all_bond_edges) != 0:
        all_bond_edges = np.concatenate(all_bond_edges, axis=1)
    if len(all_angle_edges) != 0:
        all_angle_edges = np.concatenate(all_angle_edges, axis=1)

    nonbonded_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "non_bonded"
    ]
    for prior_builder in nonbonded_builders:
        edges_and_orders = prior_builder.build_nl(
            topology, bond_edges=all_bond_edges, angle_edges=all_angle_edges
        )
        # edges_and_orders = prior_dict[nbdict]["prior_function"](topology, all_bond_edges, all_angle_edges, **prior_dict[nbdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    # process dihedral priors
    dihedral_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "dihedrals"
    ]
    for prior_builder in dihedral_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)

    return all_edges_and_orders


def split_bulk_termini(N_term, C_term, all_edges) -> Tuple:
    """
    Parameters
    ----------
    N_term:
        List of atom indices to be split as part of the N-terminal.
    C_term:
        List of atom indices to be split as part of the C-terminal.
    all_edges:
        All atom groups forming part of prior term.

    Returns
    -------
    Separated edges for bulk and terminal groups
    """
    n_term_idx = np.where(np.isin(all_edges.T, N_term))
    n_term_edges = all_edges[:, np.unique(n_term_idx[0])]

    c_term_idx = np.where(np.isin(all_edges.T, C_term))
    c_term_edges = all_edges[:, np.unique(c_term_idx[0])]

    term_edges = np.concatenate([n_term_edges, c_term_edges], axis=1)
    bulk_edges = np.array(
        [e for e in all_edges.T if not np.all(term_edges == e[:, None], axis=0).any()]
    ).T

    return n_term_edges, c_term_edges, bulk_edges


def get_dihedral_groups(
    top: md.Topology, atoms_needed: List[str], offset: List[int], tag: Optional[str]
) -> Dict:
    """
    Parameters
    ----------
    top:
        MDTraj topology object.
    atoms_needed: [4]
        Names of atoms forming dihedrals, should correspond to existing atom name in topology.
    offset: [4]
        Residue offset of each atom in atoms_needed from starting point.
    tag:
        Dihedral prior tag.

    Returns
    -------
    Dictionary of atom groups for each residue corresponding to dihedrals.

    Example
    -------
    To obtain all phi dihedral atom groups for a backbone-preserving resolution:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["C", "N", "CA", "C"], offset=[-1.,0.,0.,0.], tag="_phi"
    >>> )

    For a one-bead-per-residue mapping with only CA atoms preserved:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["CA", "CA", "CA", "CA"], offset=[-3.,-2.,-1.,0.]
    >>> )
    """
    res_per_chain = [[res for res in chain.residues] for chain in top.chains]
    atom_groups = {}
    for chain_idx, chain in enumerate(res_per_chain):
        for res in chain:
            res_idx = chain.index(res)
            if any(res_idx + ofs < 0 or res_idx + ofs >= len(chain) for ofs in offset):
                continue
            if any(atom not in [a.name for a in res.atoms] for atom in atoms_needed):
                continue
            label = f"{res.name}{tag}"
            if label not in atom_groups:
                atom_groups[label] = []
            dihedral = []
            for i, atom in enumerate(atoms_needed):
                atom_idx = top.select(
                    f"(chainid {chain_idx}) and (resid {res.index+offset[i]}) and (name {atom})"
                )
                dihedral.append(atom_idx)
            atom_groups[label].append(np.concatenate(dihedral))

    return atom_groups
