import subprocess
import shutil
from pathlib import Path
import pytest
import os
from os.path import join as osp

_here = Path(__file__).parent
_examples_dir = Path(osp(_here.parent.parent, "examples"))

_command_dict = {
    "5 beads": [
        "mlcg-tk-gen_input_data process_raw_dataset --config configuration_files/trpcage.yaml",
        "mlcg-tk-gen_input_data build_neighborlists --config configuration_files/trpcage.yaml --config configuration_files/trpcage_priors.yaml",
        "mlcg-tk-fit_priors compute_statistics --config configuration_files/trpcage_stats.yaml --config configuration_files/trpcage_priors.yaml",
        "mlcg-tk-fit_priors fit_priors --config configuration_files/trpcage_fit.yaml",
        "mlcg-tk-produce_delta_forces produce_delta_forces --config configuration_files/trpcage_delta_forces.yaml",
        "mlcg-tk-package_training_data package_training_data --config configuration_files/trpcage_packaging.yaml",
        "mlcg-tk-add_decoys add_decoy --config configuration_files/trpcage_decoys_dataset.yaml",
        "mlcg-tk-add_decoys update_partition_file --config configuration_files/trpcage_decoys_partition.yaml",
        "mlcg-tk-gen_sim_input process_sim_input --config configuration_files/trpcage_sim.yaml --config configuration_files/trpcage_priors.yaml",
    ],
    "CA": [
        "mlcg-tk-gen_input_data process_raw_dataset --config configuration_files/carbon_alpha_configs/ca_trpcage.yaml",
        "mlcg-tk-gen_input_data build_neighborlists --config configuration_files/carbon_alpha_configs/ca_trpcage.yaml --config configuration_files/carbon_alpha_configs/ca_trpcage_priors.yaml",
        "mlcg-tk-fit_priors compute_statistics --config configuration_files/carbon_alpha_configs/ca_trpcage_stats.yaml --config configuration_files/carbon_alpha_configs/ca_trpcage_priors.yaml",
        "mlcg-tk-fit_priors fit_priors --config configuration_files/carbon_alpha_configs/ca_trpcage_fit.yaml",
        "mlcg-tk-produce_delta_forces produce_delta_forces --config configuration_files/carbon_alpha_configs/ca_trpcage_delta_forces.yaml",
        "mlcg-tk-package_training_data package_training_data --config configuration_files/carbon_alpha_configs/ca_trpcage_packaging.yaml",
        "mlcg-tk-add_decoys add_decoy --config configuration_files/carbon_alpha_configs/ca_trpcage_decoys_dataset.yaml",
        "mlcg-tk-add_decoys update_partition_file --config configuration_files/carbon_alpha_configs/ca_trpcage_decoys_partition.yaml",
        "mlcg-tk-gen_sim_input process_sim_input --config configuration_files/carbon_alpha_configs/ca_trpcage_sim.yaml --config configuration_files/carbon_alpha_configs/ca_trpcage_priors.yaml",
    ],
}


# All yield fixtures in pytest are executed untill the
# yield in the order they are provided to the test function and
# after the function is finished the code after the yeld is executed
# for all the fixtures in reverse order
@pytest.fixture
def test_dir(tmp_path):
    dir_path = Path(tmp_path)
    # Copy configuration files folder in test folder
    shutil.copytree(
        osp(_examples_dir, "configuration_files"), osp(dir_path, "configuration_files")
    )
    # Copy raw data folder in test folder
    shutil.copytree(osp(_examples_dir, "demo_raw_data"), osp(dir_path, "demo_raw_data"))
    yield dir_path
    # Teardown: always run also if the test fails
    if dir_path.exists():
        shutil.rmtree(dir_path)


@pytest.mark.parametrize(
    "pipeline_tag",
    [
        "5 beads",
        "CA",
    ],
)
def test_pipeline(pipeline_tag, test_dir):

    for command in _command_dict[pipeline_tag]:
        result = subprocess.run(
            command,
            cwd=test_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, (
            f"\nIn pipeline {pipeline_tag} command failed:\n"
            f"{command}\n\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}\n"
        )
