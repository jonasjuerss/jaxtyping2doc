import os.path

import pytest

from jaxtypingdoc.main import reformat_string


def _test_full_file(file_name: str, correct_file_name: str) -> None:
    with open(os.path.join("tests", "res", file_name), "r") as file:
        py_source = file.read()
    with open(os.path.join("tests", "res", correct_file_name), "r") as file:
        correct_source = file.read()

    assert reformat_string(py_source) == correct_source
    assert reformat_string(correct_source) is None


@pytest.mark.skip(reason="Test")
def test_google() -> None:
    _test_full_file("google.py", "google_true.py")


def test_numpy() -> None:
    _test_full_file("numpy.py", "numpy_true.py")
