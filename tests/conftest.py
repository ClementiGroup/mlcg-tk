import pytest
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--runner_idx",
        default=0,
        type=int,
        help="Index of the current runner, i.e. container (relevant for test parallelization)",
    )

    parser.addoption(
        "--num_containers",
        default=1,
        type=int,
        help="Number of container (relevant for test parallelization)",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        # Convert nodeid to a path-safe string
        path = Path(str(item.fspath))

        if "tests/integration" in path.as_posix():
            item.add_marker(pytest.mark.integration)
        elif "tests/unit" in path.as_posix():
            item.add_marker(pytest.mark.unit)
