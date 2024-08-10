import argparse
import difflib
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import libcst as cst

from jaxtypingdoc.jax_typing_transformer import JaxTypingTransformer
from jaxtypingdoc.utils import get_edits_string


def reformat_string(py_source: str) -> Optional[str]:
    """

    Returns:
        None if no reformatting was necessary and the reformatted source code otherwise.
    """
    tree = cst.parse_module(py_source)
    wrapper = cst.MetadataWrapper(tree)
    transformer = JaxTypingTransformer()
    modified_tree = wrapper.visit(transformer)
    if not transformer.modified or ((modified_code := modified_tree.code) == py_source):
        return None
    return modified_code


def reformat_file(file_name: str, color_diff: bool, rewrite_file: bool) -> bool:
    script = Path(file_name)

    with open(script, "r") as py_source_file:
        py_source = py_source_file.read()

    if (modified_code := reformat_string(py_source)) is not None:
        print(f"File {file_name} needed to be updated: ")
        if color_diff:
            print(get_edits_string(py_source, modified_code))
        else:
            print(
                "".join(
                    difflib.unified_diff(
                        py_source.splitlines(True), modified_code.splitlines(True)
                    )
                )
            )

        if rewrite_file:
            with open(script, "w") as py_source_file:
                py_source_file.write(modified_code)
        return False

    return True


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        default=[],
        help="The paths to the files that should be checked.",
    )
    parser.add_argument(
        "--rewrite-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If False, files will not be updated (default: True)",
    )
    parser.add_argument(
        "--color-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If False, a classic diff (with +/- lines) will be printed rather than a "
        "colored version with green insertions and red deletions. This can be "
        "especially useful on systems where colored console output is not "
        "supported. Defaults to True",
    )
    args = parser.parse_args()

    fails = sum(
        1
        for path in args.files
        if not reformat_file(path, args.color_diff, args.rewrite_files)
    )
    if fails == 0:
        print(
            f"Congrats, the docstrings were consistent with the jaxtyping hints in all "
            f"{len(args.files)} files!"
        )
    else:
        print(f"{fails}/{len(args.files)} files were reformatted!")
        exit(1)


if __name__ == "__main__":
    main()
