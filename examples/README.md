# Input Training Data Generation Pipeline
-----------------------------------------

### How To

To prepare simulation data for eventual use in training a transferable coarse grained (CG) force field, these data must first be mapped to the CG resolution and then processed to incorporate prior energy terms. This is done as follows: first, in `gen_input_data.py` the atomistic simulations are loaded and mapped to the lower resolution, followed by the construction of the neighbor lists specific to the chosen prior energy model. Next, in the case that a prior model has not already been generated, this can be done by accumulating the statistics of the saved CG data and fitting these statistics to a set of predefined energy functions. Finally, the outputs are fed to `produce_delta_forces.py` along with the prior model so that delta forces can be calculated. The result is a set of CG coordinates, embeddings, and delta forces which can be used to train a neural network as well as the neighbor lists associated with each molecule which can be fitted to produce new prior models, if needed. In the details below, input files are provided as an example.

## Example: 1LY2 toy dataset

To exemplify the usage of this code, we provide an example on how to load a tiny dataset of
All-Atom (AA) simulations of a trpcage-variant, 1L2Y.

**This example is just to demonstrate the usage of the package to transport from AA simulations to the input for training an MLCG model.**
**This dataset does not contain enough points, nor is it in the right distribution, so that it would create a good CG model.**

The dataset is provided under the folder `./demo_raw_data/`. It contains a readme with details of the simulation.

#### 0) Create a directory to output all the intermediate files

The commands will output a lot of data and its better to save it in a separate directory to ensure hygenic file management.

`mkdir ./demo_processed_data`

#### 1) Loading and processing all-atom simulation data

Command:

`python ../scripts/gen_input_data.py process_raw_dataset --config configuration_files/trpcage.yaml`

`python ../scripts/gen_input_data.py build_neighborlists --config configuration_files/trpcage.yaml --config configuration_files/trpcage_priors.yaml`

This procedure will loop over all of the sample names specified by the `names` option. For each instance, it will load the atomistic coordinates, forces, and structures and map these to a lower resolution specified in the input file (this allows for various resolutions and CG embeddings to be used). Then, using the PriorBuilders listed in `prior_builders`, the script will generate a neighbourlist for each molecule, so long as the prior builders are implemented in `prior_gen.py` and their specific neighbour list builders are implemented in `prior_nls.py`.

Keep in mind that the priors are assumed to be in [kcal/mol] at the fitting stage so raw forces should be transformed to [kcal/mol/angstrom].

Note if you are using a custom dataset:

If your program gets killed after the loading of the all-atom data succeeded (tqdm bar finished) but before `process_raw_dataset` saved the CG output, try to set `batch_size` in your `trpcage.yaml` file. This will batch the matrix multiplication between atomistic coordinates/forces, which is the most memory-consuming part of the coarse-graining at this stage.

##### Batch processing for large molecules:

If the dataset loads into memory successfully (the tqdm bar completes), but the program fails before saving the CG output, consider setting atoms_batch_size in your trpcage.yaml file. This optional parameter specifies the batch size for processing atoms in large molecules. When set, constraints among atoms for coordinate and force mappings will be computed in batches of this size to reduce memory usage. To improve computational efficiency, it is assumed that the molecular structures have ordered residues. If atoms_batch_size is larger than the total number of atoms in the molecule, all atoms will be processed at once (the default behavior).

##### Batch processing for large datasets:

Should your dataset be too big to be loaded into memory at once (the tqdm bar doesn't finish before it fails), you can set the `mol_num_batches` in your `trpcage.yaml` file as well as your `trpcage_stats.yaml`, `trpcage_delta_forces.yaml` and `trpcage_packaging.yaml` file. This will seperate the trajectories in your dataset into `mol_num_batches` chunks that will be treated as separate molecules for the coarse-graining and statistics computing stages (see 2 below) and the statistics of the different batches will be automatically accumulated to get only one prior object in the end. Note that in this case, the force map will be only computed on the first batch and re-used for all subsequent batches to ensure consistency in the case of optimized force maps.

#### 2) Computing statistics and fitting priors

Command:

`python ../scripts/fit_priors.py compute_statistics --config configuration_files/trpcage_stats.yaml --config configuration_files/trpcage_priors.yaml`

`python ../scripts/fit_priors.py fit_priors --config configuration_files/trpcage_fit.yaml`

If a prior model has not already been created for a given set of samples, this can be generated by first computing features defined in the prior terms and then collecting statistics of these features from the input data, shown in the example. Finally, potential energy estimates are fitted to these statistics and saved as a new prior model. It is also possible to save statistics for individual samples of a dataset by specifying `save_sample_statistics=True` in the configuration file, in which case statics for the entire dataset will NOT be accumulated. Individual sample statistics can be merged as outlined below. 

Note: For situations where priors are to be fitted using simulation data from multiple datasets, statistics are computed individually for each dataset. These statistics can then be combined before fitting priors using the following:

`python ../scripts/merge_statistics.py --config configuration_files/trpcage_priors.yaml --save_dir path_to_output_directory --names '[dataset_tag_1, dataset_tag_2, etc]'`

The above code will merge statistics from multiple datasets. If, however, individual sample statistics have been computed by specifying `save_sample_statistics=True` as detailed above, these can be merged by providing sample names (same as previous `names` options in Step 1) and including a dataset tag in the configuration file or by passing the dataset name using `--tag dataset_tag`. This option allows for more control and debugging capabilities in case individual samples in the dataset produce problematic statistics. 

#### 3) Producing delta forces

Command:

`python ../scripts/produce_delta_forces.py produce_delta_forces --config configuration_files/trpcage_delta_forces.yaml`

This procedure will load the prior model specified by `prior_fn` and then once again loop over all sample names provided. It will then calculate and remove the baseline forces using the coordinates, forces, embeddings, and neighbourlists created in the previous step. It will then save the delta forces which can then be used for training.

The following example script shows how delta forces can be computed on a computing cluster using GPU acceleration:

```
#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=trpcage_delta_forces_gpu.log
#SBATCH --job-name=test_job

python ../scripts/produce_delta_forces.py produce_delta_forces --config configuration_files/trpcage_produce.yaml
```
Here, make sure to specify `cuda` for the `device` option in the configuration file.
Note that depending on the GPU being used and its available memory, it may be necessary to adjust the `batch_size`.

#### 4) Package Training Data

Command:

`python ../scripts/package_training_data.py package_training_data --config configuration_files/trpcage_packaging.yaml`

Once all training data has been produced, these data must be packaged in a form that can be passed to the MLCG library for model training. In this step, CG coordinates, delta forces, and embeddings are loaded for all provided sample names in a raw dataset and saved as an HDF5 file. In the same step, molecules are split into training and validation sets and a saved in a partition file, which also stores information about the batch sizes to use for training and any striding that should be applied. In the case that multiple dataset are used to train a model, these can be merged into combined HDF5 and partition files using the following:

`python ../scripts/package_training_data.py combine_datasets --dataset_names '[dataset_1, dataset_2, etc]' --save_dir /path/to/saved/files/ --force_tag tag`

The optional force tag specifies a label given to produced delta forces and saved packaged data.

#### 4) bis Add Decoys To Training Data

It is possible to add so-called decoys to the training data, distorted (unphysical) configurations with a zero delta-force label, such that the network has examples of configurations on which it should rely on the prior. 

The script `add_decoys.py` enables the addition of decoys to a previously constructed HDF5 dataset. It can be used the following way to append decoys to the existing HDF5 dataset (or the config file can be modified for the script to copy the original HDF5 dataset before adding the decoys):

`python ../scripts/add_decoys.py add_decoy --config configuration_files/trpcage_decoys_dataset.yaml`

The same script can also be used to add these decoys to an existing partition file in order to incorporate them during training:

`python ../scripts/add_decoys.py update_partition_file --config configuration_files/trpcage_decoys_partition.yaml`

Note that an arbitrary number of decoys with different noise levels and strides can be appended to a dataset, only the decoys present in the partition file will effectively be taken into account for training.

#### 5) Generate simulation input

Command:

`python ../scripts/gen_sim_input.py process_sim_input --config configuration_files/trpcage_sim.yaml --config configuration_files/trpcage_priors.yaml`

A trained MLCG model serves as a forcefield for conducting protein simulations. To run simulations of a particular system, the command above will process each structure file indicated by the `pdb_fns` option, map these to the specified CG resolution, generate neighbor lists corresponding the the given `prior_builders`, and save the specified number of copies of `AtomicData` objects storing this information.

In contrast to traditional MD forcefields, machine-learned force fields are designed to process data efficiently in batches, making it advantageous to run multiple simulations in parallel in order to maximize resource utilization and minimize the cost per trajectory. The `copies` option in the configuration file should be carefully selected based on the size of the system to ensure efficient memory usage, and may require some testing to achieve optimal simulation performance. 

For generating a simulation input for the pretrained transferable model provided with the manuscript `Navigating protein landscapes with a machine-learned coarse-grained model`, use the provided `transferable_priors.yaml` in `configuration_files` to build an input configuration using consistent priors with the ones used in the manuscript. Adapt the `trpcage_sim.yaml` file to the protein sequence to be simulated.
