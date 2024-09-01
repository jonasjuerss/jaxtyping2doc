import subprocess
import sys
from os.path import join
from subprocess import Popen

import pytest


@pytest.mark.parametrize(
    ("files", "correct"),
    [
        ([join("edge_cases", "classes", "true.py")], 1),
        ([join("edge_cases", "classes", "raw.py")], 0),
        (
            [
                join("edge_cases", "classes", "true.py"),
                join("edge_cases", "classes", "raw.py"),
            ],
            1,
        ),
    ],
)
def test_main_script_success(files: list[str], correct: int) -> None:
    files_str = " ".join(join("tests", "res", f) for f in files)
    process = Popen(
        f"jaxtype2doc {files_str} --no-rewrite-files --no-color-diff",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    out, err = process.communicate()
    errcode = process.returncode
    last_line = out.decode(sys.stdout.encoding).splitlines()[-1]

    if correct == len(files):
        assert errcode == 0
        assert (
            last_line
            == f"Congrats, the docstrings were consistent with the jaxtyping hints in "
            f"all {len(files)} files!"
        )
    else:
        assert errcode == 1
        assert (
            last_line
            == f"{len(files) - correct}/{len(files)} files would be reformatted!"
        )
