from os.path import join

import pytest

from jaxtypingdoc.main import reformat_string


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            join("styles", "google"),
            marks=pytest.mark.skip(
                reason="Currently unsupported due to hard to avoid ':' in Returns"
            ),
        ),
        join("styles", "numpy"),
        join("edge_cases", "incomplete_docstring"),
        join("edge_cases", "misleading_hint"),
        join("edge_cases", "indirect_array_type"),
        # join("edge_cases", "mixed_style"),
        join("edge_cases", "classes"),
    ],
)
def test_file(path: str) -> None:
    with open(join("tests", "res", path, "raw.py"), "r") as file:
        py_source = file.read()
    with open(join("tests", "res", path, "true.py"), "r") as file:
        correct_source = file.read()

    assert reformat_string(py_source) == correct_source
    assert reformat_string(correct_source) is None
