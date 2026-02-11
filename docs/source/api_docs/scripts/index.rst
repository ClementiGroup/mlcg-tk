Scripts
=============

All of the scripts and their methods can be accessed directly on the command line
by appending the ``mlcg-tk-`` prefix and removing the ``.py``. You can call the methods
inside each script and pass all the arguments via a yaml file that is parsed
using `jsonargparse <https://jsonargparse.readthedocs.io/en/v4.46.0/>`_.

For example

::
    mlcg-tk-gen_input_data process_raw_dataset --config ./trpcage.yaml



.. toctree::

    gen_input_data
    fit_priors
    merge_statistics
    produce_delta_forces
    package_training_data
    add_decoys
    gen_sim_input
