import pytest

from mlcg_tk.input_generator.utils import get_output_tag


@pytest.mark.parametrize(
    "tag_label, placement, expected_out, expected_err",
    [
        (["prior", "ca"], "before", "prior_ca_", False),
        (["prior", "ca"], "after", "_prior_ca", False),
        (["prior", "ca"], "later", "_prior_ca", True),
    ],
)
def test_output_tag(tag_label, placement, expected_out, expected_err):
    if expected_err:
        with pytest.raises(ValueError):
            get_output_tag(tag_label, placement)
    else:
        assert get_output_tag(tag_label, placement) == expected_out
