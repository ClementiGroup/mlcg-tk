import mdtraj as md
import pickle

from typing import List, Dict, Tuple, Optional, Union
from copy import deepcopy
import numpy as np
import mdtraj as md
import torch
import warnings
import os
from importlib import import_module

from torch_geometric.data.collate import collate

from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.data.atomic_data import AtomicData

from .utils import (
    map_cg_topology,
    filter_cis_frames,
    slice_coord_forces,
    get_terminal_atoms,
    get_edges_and_orders,
    get_output_tag,
)
from .prior_gen import PriorBuilder


def get_strides(n_structure: int, batch_size: int):
    """
    Helper function to stride batched data
    """
    n_elem, remain = np.divmod(n_structure, batch_size)
    assert remain > -1, f"remain: {remain}"
    if remain == 0:
        batches = np.zeros(n_elem + 1)
        batches[1:] = batch_size
    else:
        batches = np.zeros(n_elem + 2)
        batches[1:-1] = batch_size
        batches[-1] = remain
    strides = np.cumsum(batches, dtype=int)
    strides = np.vstack([strides[:-1], strides[1:]]).T
    return strides


class CGDataBatch:
    """
    Splits input CG data into batches for further memory-efficient processing

    Attributes
    ----------
    batch_size:
        Number of frames to use in each batch
    stride:
        Integer by which to stride frames
    concat_forces:
        Boolean indicating whether forces should be added to batch
    cg_coords:
        Coarse grained coordinates
    cg_forces:
        Coarse grained forces
    cg_embeds:
        Atom embeddings
    cg_prior_nls:
        Dictionary of prior neighbour list
    """

    def __init__(
        self,
        cg_coords: np.ndarray,
        cg_forces: np.ndarray,
        cg_embeds: np.ndarray,
        cg_prior_nls: Dict,
        batch_size: int,
        stride: int,
        weights: Optional[np.ndarray] = None,
        concat_forces: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.stride = stride
        self.concat_forces = concat_forces
        if cg_coords is None:
            self.cg_coords = None
        else:
            self.cg_coords = torch.from_numpy(cg_coords[::stride])
        if cg_forces is None:
            self.cg_forces = None
        else:
            self.cg_forces = torch.from_numpy(cg_forces[::stride])
        self.cg_embeds = torch.from_numpy(cg_embeds)
        self.cg_prior_nls = cg_prior_nls
        if isinstance(weights, np.ndarray):
            self.weights = torch.from_numpy(weights[::stride])
            if stride != 1:
                self.weights = self.weights/torch.sum(self.weights)
        else:
            self.weights = None
        if self.cg_coords is None:
            self.n_structure = 0
        else:
            self.n_structure = self.cg_coords.shape[0]
        if batch_size > self.n_structure:
            self.batch_size = self.n_structure

        self.strides = get_strides(self.n_structure, self.batch_size)
        self.n_elem = self.strides.shape[0]

    def __len__(self):
        return self.n_elem

    def __getitem__(self, idx):
        """
        Returns list of AtomicData objects for indexed batch
        """
        st, nd = self.strides[idx]
        data_list = []
        # TODO: build the collated AtomicData by hand to avoid copy/concat ops
        for ii in range(st, nd):
            dd = dict(
                pos=self.cg_coords[ii],
                atom_types=self.cg_embeds,
                masses=None,
                neighborlist=self.cg_prior_nls,
            )
            if self.concat_forces:
                dd["forces"] = self.cg_forces[ii]

            data = AtomicData.from_points(**dd)
            if isinstance(self.weights, torch.Tensor):
                data.weights = self.weights[ii]
            data_list.append(data)
        datas, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return datas


class SampleCollection:
    """
    Input generation object for loading, manupulating, and saving training data samples.

    Attributes
    ----------
    name:
        String associated with atomistic trajectory output.
    tag:
        String to identify dataset in output files.
    pdb_fn:
        File location of atomistic structure to be used for topology.
    """

    def __init__(
        self,
        name: str,
        tag: str,
    ) -> None:
        self.name = name
        self.tag = tag

    def apply_cg_mapping(
        self,
        cg_atoms: List[str],
        embedding_function: str,
        embedding_dict: str,
        skip_residues: Optional[List[str]] = None,
    ):
        """
        Applies mapping function to atomistic topology to obtain CG representation.

        Parameters
        ----------
        cg_atoms:
            List of atom names to preserve in CG representation.
        embedding_function:
            Name of function (should be defined in embedding_maps) to apply CG mapping.
        embedding_dict:
            Name of dictionary (should eb defined in embedding_maps) to define embeddings of CG beads.
        skip_residues: (Optional)
            List of residue names to skip (can be used to skip terminal caps, for example).
            Currently, can only be used to skip all residues with given name.
        """
        if isinstance(embedding_dict, str):
            self.embedding_dict = eval(embedding_dict)

        self.top_dataframe = self.top_dataframe.apply(
            map_cg_topology,
            axis=1,
            cg_atoms=cg_atoms,
            embedding_function=embedding_function,
            skip_residues=skip_residues,
        )
        cg_df = deepcopy(self.top_dataframe.loc[self.top_dataframe["mapped"] == True])

        cg_atom_idx = cg_df.index.values.tolist()
        self.cg_atom_indices = cg_atom_idx

        cg_df.index = [i for i in range(len(cg_df.index))]
        cg_df.serial = [i + 1 for i in range(len(cg_df.index))]
        #
        # to avoid a bug  related to the mdtraj convertion of the
        # topology dataframe back into a md.Topology object when dealing
        # with homo-mono-dimers, we need to shift the resseq so that
        # each chain has different resseq numbers.
        #
        # See https://github.com/ClementiGroup/mlcg-playground/pull/9
        # for more details
        #
        cg_df.resSeq = [
            cg_df.resSeq[i] + cg_df.value_counts("chainID")[0 : cg_df.chainID[i]].sum()
            for i in range(len(cg_df.resSeq))
        ]
        self.cg_dataframe = cg_df

        cg_map = np.zeros((len(cg_atom_idx), self.input_traj.n_atoms))
        cg_map[[i for i in range(len(cg_atom_idx))], cg_atom_idx] = 1
        if not all([sum(row) == 1 for row in cg_map]):
            warnings.warn("WARNING: Slice mapping matrix is not unique.")
        if not all([row.tolist().count(1) == 1 for row in cg_map]):
            warnings.warn("WARNING: Slice mapping matrix is not linear.")

        self.cg_map = cg_map

        # save N_term and C_term as None, to be overwritten if terminal embeddings used
        self.N_term = None
        self.C_term = None

    def add_terminal_embeddings(
        self, N_term: Union[str, None] = "N", C_term: Union[str, None] = "C"
    ):
        """
        Adds separate embedding to terminals (do not need to be defined in original embedding_dict).

        Parameters
        ----------
        N_term:
            Atom of N-terminus to which N_term embedding will be assigned.
        C_term:
            Atom of C-terminus to which C_term embedding will be assigned.

        Either of N_term and/or C_term can be None; in this case only one (or no) terminal embedding(s) will be assigned.
        """
        df_cg = self.cg_dataframe
        # proteins with multiple chains will have multiple N- and C-termini
        self.N_term = N_term
        self.C_term = C_term

        chains = df_cg.chainID.unique()
        if N_term is not None:
            if "N_term" not in self.embedding_dict:
                self.embedding_dict["N_term"] = max(self.embedding_dict.values()) + 1
            N_term_atom = []
            # as the search for N- and C- is based on resseq, we need to proceed
            # chain by chain
            for chain in chains:
                chain_filter = df_cg["chainID"] == chain
                chain_resseq_min = df_cg[chain_filter]["resSeq"].min()
                N_term_atom.extend(
                    df_cg.loc[
                        (df_cg["resSeq"] == chain_resseq_min)
                        & (df_cg["name"] == N_term)
                        & chain_filter
                    ].index.to_list()
                )
            for idx in N_term_atom:
                self.cg_dataframe.at[idx, "type"] = self.embedding_dict["N_term"]

        if C_term is not None:
            if "C_term" not in self.embedding_dict:
                self.embedding_dict["C_term"] = max(self.embedding_dict.values()) + 1
            C_term_atom = []
            for chain in chains:
                chain_filter = df_cg["chainID"] == chain
                chain_resseq_max = df_cg[chain_filter]["resSeq"].max()
                C_term_atom.extend(
                    df_cg.loc[
                        (df_cg["resSeq"] == chain_resseq_max)
                        & (df_cg["name"] == C_term)
                        & chain_filter
                    ].index.to_list()
                )
            for idx in C_term_atom:
                self.cg_dataframe.at[idx, "type"] = self.embedding_dict["C_term"]

    def process_coords_forces(
        self,
        coords: np.ndarray,
        forces: np.ndarray,
        topology: md.Topology,
        mapping: str = "slice_aggregate",
        filter_cis: bool = False,
        force_stride: int = 100,
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps coordinates and forces to CG resolution

        Parameters
        ----------
        coords: [n_frames, n_atoms, 3]
            Atomistic coordinates
        forces: [n_frames, n_atoms, 3]
            Atomistic forces
        topology: 
            mdtraj topology to lead atomistic coordinates (used for cis-omega angles filtering)
        mapping:
            Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize'.
        filter_cis:
            If True, frames containing a cis-omega angle will be filtered out
        force_stride:
            Striding to use for force projection results
        batch_size:
            Batching the coords and forces projection to CG

        Returns
        -------
        Tuple of np.ndarray's for coarse grained coordinates and forces
        """
        if coords.shape != forces.shape:
            warnings.warn(
                "Cannot process coordinates and forces: mismatch between array shapes."
            )
            return
        else:
            if filter_cis:
                coords, forces = filter_cis_frames(
                    coords, forces, topology, verbose=True
                )
            if coords.shape[0] != 0:

                # since the cis-pro filtering might have removed a lot of frames
                # we need to make sure the force_stride is not too large
                # ie there are at least min(n_frames, 100) frames left after striding
                while coords.shape[0]<100*force_stride:
                    force_stride /= 10
                    if force_stride == 1:
                        break

                cg_coords, cg_forces, force_map = slice_coord_forces(
                    coords, forces, self.cg_map, mapping, force_stride, batch_size
                )
                self.force_map = force_map
            else: # all frames were removed by cis-filtering
                cg_coords = None
                cg_forces = None

            self.cg_coords = cg_coords
            self.cg_forces = cg_forces

            return cg_coords, cg_forces

    def save_cg_output(
        self,
        save_dir: str,
        save_coord_force: bool = True,
        save_cg_maps: bool = True,
        cg_coords: Union[np.ndarray, None] = None,
        cg_forces: Union[np.ndarray, None] = None,
    ):
        """
        Saves processed CG data.

        Parameters
        ----------
        save_dir:
            Path of directory to which output will be saved.
        save_coord_force:
            Whether coordinates and forces should also be saved.
        cg_coords:
            CG coordinates; if None, will check whether these are saved as attribute.
        cg_forces:
            CG forces; if None, will check whether these are saved as an object attribute.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not hasattr(self, "cg_atom_indices"):
            warnings.warn("CG mapping must be applied before outputs can be saved.")
            return

        save_templ = os.path.join(save_dir, get_output_tag([self.tag, self.name], placement="before"))
        cg_xyz = self.input_traj.atom_slice(self.cg_atom_indices).xyz
        cg_traj = md.Trajectory(cg_xyz, md.Topology.from_dataframe(self.cg_dataframe))
        cg_traj.save_pdb(f"{save_templ}cg_structure.pdb")

        embeds = np.array(self.cg_dataframe["type"].to_list())
        np.save(f"{save_templ}cg_embeds.npy", embeds)

        if save_coord_force:
            if cg_coords == None:
                if not hasattr(self, "cg_coords"):
                    warnings.warn(
                        "No coordinates found; only CG structure, embeddings and loaded forces will be saved."
                    )
                else:
                    if self.cg_coords is None:
                        warnings.warn(
                            "No coordinates found; only CG structure, embeddings and loaded forces will be saved."
                        )
                    else:
                        np.save(f"{save_templ}cg_coords.npy", self.cg_coords)
            else:
                np.save(f"{save_templ}cg_coords.npy", cg_coords)

            if cg_forces == None:
                if not hasattr(self, "cg_forces"):
                    warnings.warn(
                        "No forces found;  only CG structure, embeddings, and loaded coordinates will be saved."
                    )
                else:
                    if self.cg_forces is None:
                        warnings.warn(
                            "No forces found;  only CG structure, embeddings, and loaded coordinates will be saved."
                        )
                    else:
                        np.save(f"{save_templ}cg_forces.npy", self.cg_forces)
            else:
                np.save(f"{save_templ}cg_forces.npy", cg_forces)

        if save_cg_maps:
            if not hasattr(self, "cg_map"):
                warnings.warn(
                    "No cg coordinate map found. Skipping save."
                )
            else:
                np.save(f"{save_templ}cg_coord_map.npy", self.cg_map)

            if not hasattr(self, "force_map"):
                warnings.warn(
                    "No cg force map found. Skipping save."
                )
            else:
                np.save(f"{save_templ}cg_force_map.npy", self.force_map)


    def get_prior_nls(
        self, prior_builders: List[PriorBuilder], save_nls: bool = True, **kwargs
    ) -> Dict:
        """
        Creates neighbourlists for all prior terms specified in the prior_dict.

        Parameters
        ----------
        prior_builders:
            List of PriorBuilder objects and their corresponding parameters.
            Input config file must minimally contain the following information for
            each builder:
                class_path: class specifying PriorBuilder object implemented in `prior_gen.py`
                init_args:
                    name: string specifying type as one of 'bonds', 'angles', 'dihedrals', 'non_bonded'
                    nl_builder: name of class implemented in `prior_nls.py` which will be used to collect
                                atom groups associated with the prior term.
        save_nls:
            If true, will save an output of the molecule's neighbourlist.
        kwargs:
            save_dir:
                If save_nls = True, the neighbourlist will be saved to this directory.
            prior_tag:
                String identifying the specific combination of prior terms.

        Returns
        -------
        Dictionary of prior terms with specific index mapping for the given molecule.

        Example
        -------
        To build neighbour lists for a system with priors for bonds, angles, nonbonded pairs, and phi and
        psi dihedral angles:

            - class_path: input_generator.Bonds
              init_args:
                name: bonds
                separate_termini: true
                nl_builder: input_generator.StandardBonds
            - class_path: input_generator.Angles
              init_args:
                name: angles
                separate_termini: true
                nl_builder: input_generator.StandardAngles
            - class_path: input_generator.NonBonded
              init_args:
                name: non_bonded
                min_pair: 6
                res_exclusion: 1
                separate_termini: false
                nl_builder: input_generator.Non_Bonded
            - class_path: input_generator.Dihedrals
              init_args:
                name: phi
                nl_builder: input_generator.Phi
            - class_path: input_generator.Dihedrals
              init_args:
                name: psi
                nl_builder: input_generator.Psi
        """

        for prior_builder in prior_builders:
            if getattr(prior_builder, "separate_termini", False):
                prior_builder = get_terminal_atoms(
                    prior_builder,
                    cg_dataframe=self.cg_dataframe,
                    N_term=self.N_term,
                    C_term=self.C_term,
                )

        # get atom groups for edges and orders for all prior terms
        cg_top = self.input_traj.atom_slice(self.cg_atom_indices).topology

        # we need to add an extra step for CA case: in this situation, the bonds
        atoms = list(cg_top.atoms)
        unique_atom_types = set([atom.name for atom in atoms])
        if unique_atom_types == set(["CA"]):
            # iterate over chains
            for chain in cg_top.chains:
                ch_atoms = list(chain.atoms)
                # iterate over CA atoms in each chain and add bonds between them 
                for i, _ in enumerate(ch_atoms[:-1]):
                    cg_top.add_bond(ch_atoms[i], ch_atoms[i + 1])

        all_edges_and_orders = get_edges_and_orders(
            prior_builders,
            topology=cg_top,
        )
        tags = [x[0] for x in all_edges_and_orders]
        orders = [x[1] for x in all_edges_and_orders]
        edges = [
            (
                torch.tensor(x[2]).type(torch.LongTensor)
                if isinstance(x[2], np.ndarray)
                else x[2].type(torch.LongTensor)
            )
            for x in all_edges_and_orders
        ]
        prior_nls = {}
        for tag, order, edge in zip(tags, orders, edges):
            nl = make_neighbor_list(tag, order, edge)
            prior_nls[tag] = nl

        if save_nls:
            ofile = os.path.join(
                kwargs["save_dir"],
                f"{get_output_tag([self.tag, self.name], placement='before')}prior_nls_{kwargs['prior_tag']}.pkl",
            )
            with open(ofile, "wb") as pfile:
                pickle.dump(prior_nls, pfile)

        return prior_nls
    
    def has_saved_cg_output(self, save_dir: str, prior_tag: str = "") -> bool:
        """
        Returns True if cg data exists for this SampleCollection

        Used to skip processing of molecules where all frames have been removed by cis conformation filtering
        
        Parameters
        ----------
        save_dir:
            Location of saved cg data
        prior_tag:
            String identifying the specific combination of prior terms

        Returns
        -------
        True if cg output for the sample corresponding to prior_tag is present in save_dir
        False otherwise
        """
        save_templ = os.path.join(save_dir, get_output_tag([self.tag, self.name], placement="before"))
        if not os.path.exists(f"{save_templ}cg_coords.npy"):
            return False 
        elif not os.path.exists(f"{save_templ}cg_forces.npy"):
            return False
        else:
            return True

    def load_cg_output(self, save_dir: str, prior_tag: str = "") -> Tuple:
        """
        Loads all cg data produced by `save_cg_output` and `get_prior_nls`

        Parameters
        ----------
        save_dir:
            Location of saved cg data
        prior_tag:
            String identifying the specific combination of prior terms

        Returns
        -------
        Tuple of np.ndarrays containing coarse grained coordinates, forces, embeddings,
        structure, and prior neighbour list
        """
        save_templ = os.path.join(save_dir, get_output_tag([self.tag, self.name], placement="before"))
        if os.path.exists(f"{save_templ}cg_coords.npy"):
            cg_coords = np.load(f"{save_templ}cg_coords.npy")
        else:
            cg_coords = None
        if os.path.exists(f"{save_templ}cg_forces.npy"):
            cg_forces = np.load(f"{save_templ}cg_forces.npy")
        else:
            cg_forces = None
        cg_embeds = np.load(f"{save_templ}cg_embeds.npy")
        cg_pdb = md.load(f"{save_templ}cg_structure.pdb")
        # load NLs
        ofile =  f"{save_templ}prior_nls{get_output_tag(prior_tag, placement='after')}.pkl"

        with open(ofile, "rb") as f:
            cg_prior_nls = pickle.load(f)
        return cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls

    def load_cg_output_into_batches(
        self,
        save_dir: str,
        prior_tag: str,
        batch_size: int,
        stride: int,
        weights_template_fn: Optional[str],
    ):
        """
        Loads saved CG data nad splits these into batches for further processing

        Parameters
        ----------
        save_dir:
            Location of saved cg data
        prior_tag:
            String identifying the specific combination of prior terms
        batch_size:
            Number of frames to use in each batch
        stride:
            Integer by which to stride frames

        Returns
        -------
        Loaded CG data split into list of batches
        """
        cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls = self.load_cg_output(
            save_dir, prior_tag
        )
        #load weights if given
        if weights_template_fn != None:
            weights = np.load(
                os.path.join(save_dir, weights_template_fn.format(self.name))
                ) 
        else:
            weights = None
        batch_list = CGDataBatch(
            cg_coords, cg_forces, cg_embeds, cg_prior_nls, batch_size, stride, weights
        )
        return batch_list
    
    def load_training_inputs(self, training_data_dir: str, force_tag: str = "", stride: int = 1) -> Tuple:
        """
        Loads all cg data produced by `save_cg_output` and `get_prior_nls`

        Parameters
        ----------
        training_data:
            Location of saved cg data including delta forces
        force_tag:
            String identifying the produced delta forces

        Returns
        -------
        Tuple of np.ndarrays containing coarse grained coordinates, delta forces, and embeddings,
        """
        save_templ = os.path.join(training_data_dir, get_output_tag([self.tag, self.name], placement="before"))
        cg_coords = np.load(f"{save_templ}cg_coords.npy")[::stride]
        cg_embeds = np.load(f"{save_templ}cg_embeds.npy")

        save_templ_forces = os.path.join(training_data_dir, get_output_tag([self.tag, self.name, force_tag], placement="before"))
        cg_forces = np.load(f"{save_templ_forces}delta_forces.npy")[::stride]
        
        return cg_coords, cg_forces, cg_embeds


class RawDataset:
    """
    Generates a list of data samples for a specified dataset

    Attributes
    ----------
    dataset_name:
        Name given to dataset
    names:
        List of sample names
    tag:
        Label given to all output files produced from dataset
    dataset:
        List of SampleCollection objects for all samples in dataset
    """

    def __init__(self, dataset_name: str, names: List[str], tag: str) -> None:
        self.dataset_name = dataset_name
        self.names = names
        self.tag = tag
        self.dataset = []

        for name in names:
            data_samples = SampleCollection(
                name=name,
                tag=tag,
            )
            self.dataset.append(data_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class SimInput:
    """
    Generates a list of samples from pdb structures to be used in simulation

    Attributes
    ----------
    dataset_name:
        Name given to dataset
    tag:
        Label given to all output files produced from dataset
    pdb_fns:
        List of pdb filenames from which samples will be generated
    dataset:
        List of SampleCollection objects for all structures
    """

    def __init__(self, dataset_name: str, tag: str, pdb_fns: List[str]) -> None:
        self.dataset_name = dataset_name
        self.names = [fn[:-4] for fn in pdb_fns]
        self.dataset = []

        for name in self.names:
            data_samples = SampleCollection(
                name=name,
                tag=tag,
            )
            self.dataset.append(data_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
