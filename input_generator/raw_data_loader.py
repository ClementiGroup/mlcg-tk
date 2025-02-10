import numpy as np
import os
from natsort import natsorted
from glob import glob
import h5py
from typing import Tuple
import mdtraj as md
import warnings
from pathlib import Path

class DatasetLoader:
    r"""
    Base loader object for a dataset. Defines the interface that all loader should have.

    As there is no standard way of saving the output of molecular dynamics simulations, 
    the objective of this is class is to be an interface which will load the MD data
    from a given dataset stored in a local path. It will further process the data to 
    remove thing that are not relevant to our system (for example, solvent coordinates
    and forces) and returned the clean coordinate and force as np.ndarrays and an mdtraj
    object that carries the topology
    
    The loader should be flexible enough to handle datasets with different molecules, 
    multiple independent trajectories of different length and other things.


    """
    def get_traj_top(self, name: str, pdb_fn: str):
        r"""
        Function to get the topology associated with a trajectory in the dataset.

        PDB files represent molecules by putting the position of every atom in cartesian space.
        From the distances between atoms, covalent bonds and other things can be deduced. 
        
        Note that, by convention, raw PDB files use angstroms as the unit of the coordinates.
        Engines such as Pymol and mdtraj convert this values to nanometers

        Parameters
        ----------
        name:
            Name of input sample (i.e. the molecule or object in the data)
        pdb_fn:
            String representing a path to a PDB file which has the topolgy for
        """
        raise NotImplementedError(f"Base class {self.__class__} has no implementation")
    
    def load_coords_forces(
        self, base_dir: str, name: str, stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Method to load the coordinate and force data

        This requires the base directory where files can be saved, the name of the molecule
        we are trying to load and a stride parameter. The stride is useful for cases where
        the data is saved too frequently.

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files.
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        """
        raise NotImplementedError(f"Base class {self.__class__} has no implementation")


class CATH_loader(DatasetLoader):
    """
    Loader object for original 50 CATH domain proteins
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str, stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        """
        # return sorted(name, key=alphanum_key)
        outputs_fns = natsorted(glob(os.path.join(base_dir, f"output/{name}/*_part_*")))
        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for fn in outputs_fns:
            output = np.load(fn)
            coord = output["coords"]
            coord = 10.0 * coord  # convert nm to angstroms
            force = output["Fs"]
            force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)[::stride]
        aa_forces = np.concatenate(aa_force_list)[::stride]
        return aa_coords, aa_forces


class DIMER_loader(DatasetLoader):
    """
    Loader object for original dataset of mono- and dipeptide pairwise umbrella sampling simulations
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given DIMER pair name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str, stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given DIMER pair name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        """
        with h5py.File(os.path.join(base_dir, "allatom.h5"), "r") as data:
            coord = data["MINI"][name]["aa_coords"][:][::stride]
            force = data["MINI"][name]["aa_forces"][:][::stride]

        # convert to kcal/mol/angstrom and angstrom
        # from kJ/mol/nm and nm
        coord = coord * 10
        force = force / 41.84

        return coord, force


class Trpcage_loader(DatasetLoader):
    """
    Loader object for Trpcage simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str,  stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        coords_fns = natsorted(
            glob(
                os.path.join(base_dir, f"coords_nowater/trp_coor_folding-trpcage_*.npy")
            )
        )

        forces_fns = [
            fn.replace(
                "coords_nowater/trp_coor_folding", "forces_nowater/trp_force_folding"
            )
            for fn in coords_fns
        ]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in zip(coords_fns, forces_fns):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class Cln_loader(DatasetLoader):
    def get_traj_top(self, name: str, pdb_fn: str):
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str, stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords_fns = natsorted(
            glob(os.path.join(base_dir, f"coords_nowater/chig_coor_*.npy"))
        )

        forces_fns = [
            fn.replace("coords_nowater/chig_coor_", "forces_nowater/chig_force_")
            for fn in coords_fns
        ]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in zip(coords_fns, forces_fns):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class Villin_loader(DatasetLoader):
    def get_traj_top(self, name: str, pdb_fn: str):
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe
    
    def load_coords_forces(
            self, base_dir: str, name: str, stride: int =1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords_fns = sorted(
            glob(
                os.path.join(base_dir, f"{name}/*_coords.npy")
            )
        ) # combining all trajectories from single starting structure

        forces_fns = sorted(
            glob(
                os.path.join(base_dir, f"{name}/*_forces.npy")
            )
        )

        aa_coord_list = []
        aa_force_list = []
        for c, f in zip(coords_fns, forces_fns):
            coords = np.load(c)
            forces = np.load(f)
            assert coords.shape == forces.shape

            aa_coord_list.append(coords)
            aa_force_list.append(forces)
        
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class OPEP_loader(DatasetLoader):
    """
    Loader for octapeptides dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str, stride: int =1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        coord_files = sorted(glob(f"{base_dir}coords_nowater/opep_{name}/*.npy"))
        if len(coord_files) == 0:
            coord_files = sorted(
                glob(f"{base_dir}coords_nowater/coor_opep_{name}_*.npy")
            )
        force_files = sorted(glob(f"{base_dir}forces_nowater/opep_{name}/*.npy"))
        if len(force_files) == 0:
            force_files = sorted(
                glob(f"{base_dir}forces_nowater/force_opep_{name}_*.npy")
            )

        aa_coord_list = []
        aa_force_list = []
        for c, f in zip(coord_files, force_files):
            coords = np.load(c)
            forces = np.load(f)
            aa_coord_list.append(coords)
            aa_force_list.append(forces)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class SimInput_loader(DatasetLoader):
    """
    Loader for protein structures to be used in CG simulations
    """

    def get_traj_top(self, name: str, raw_data_dir: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        raw_data_dir:
            Path to pdb structure file
        """
        pdb = md.load(f"{raw_data_dir}{name}.pdb")
        input_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = input_traj.topology.to_dataframe()[0]
        return input_traj, top_dataframe
