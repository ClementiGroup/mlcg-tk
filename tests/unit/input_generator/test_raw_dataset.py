import numpy as np
import pytest

from mlcg_tk.input_generator.raw_dataset import get_strides  # adjust import if needed


@pytest.mark.parametrize(
    "n_structure,batch_size,expected",
    [
        (10, 5, np.array([[0, 5], [5, 10]])),
        (12, 5, np.array([[0, 5], [5, 10], [10, 12]])),
        (3, 10, np.array([[0, 3]])),
    ],
)
def test_get_strides(n_structure, batch_size, expected):
    strides = get_strides(n_structure, batch_size)

    # exact match
    assert np.array_equal(strides, expected)

    # sanity checks
    assert strides[0, 0] == 0
    assert strides[-1, 1] == n_structure
    assert np.all(strides[1:, 0] == strides[:-1, 1])
