import warnings
from typing import TypeVar, Sequence, Optional

from libcst import (
    SimpleString,
    BaseCompoundStatement,
    SimpleStatementLine,
    BaseSuite,
    Expr,
    ConcatenatedString,
    CSTNode,
)
import libcst as cst

# https://stackoverflow.com/questions/77173549/replacing-a-simplestring-inside-a-libcst-function-definition-dataclasses-froze
# No idea why mypy doesn't seem to identify the libcst classes here as types (despite
# libcst being installed)
T = TypeVar("T", BaseSuite, Sequence[SimpleStatementLine | BaseCompoundStatement])  # type: ignore # noqa E501
S = TypeVar("S", bound=CSTNode)


def _rec_set_docstring(expr: S, new_str: str) -> S:
    # While loop
    if isinstance(expr, (BaseSuite, SimpleStatementLine)):
        if len(expr.body) == 0:
            raise ValueError
        return expr.with_changes(  # type: ignore
            body=(_rec_set_docstring(expr.body[0], new_str),) + tuple(expr.body[1:])
        )

    # TODO handle bytes

    if not isinstance(expr, Expr):
        raise ValueError
    if isinstance(expr.value, (SimpleString, ConcatenatedString)):
        return expr.with_changes(value=expr.value.with_changes(value=new_str))  # type: ignore # noqa E501
    raise ValueError


def set_docstring(body: T, new_str: str) -> T:
    if isinstance(body, Sequence):
        # actually, don't do recursion here
        # TODO probably need to set this in a different way
        # body[0] = set_docstring(body[0], val)
        return body

    return _rec_set_docstring(body, new_str)


def parse_annotation(
    annotation: Optional[cst.Annotation], jaxtyping_imports: set[str]
) -> Optional[str]:
    if annotation is None or not hasattr(annotation, "annotation"):
        return None
    subscr = annotation.annotation
    if not isinstance(subscr, cst.Subscript):
        return None

    if isinstance(subscr.value, cst.Name):
        # Float[...]
        if subscr.value.value in jaxtyping_imports:
            dtype = subscr.value.value
        else:
            return None
    elif isinstance(subscr.value, cst.Attribute):
        # jaxtyping.Float[...] (here, we skip any checking of whether this was
        # actually imported. If I wrote jaxtyping.Float[...], as an annotation,
        # this is probably what I meant)
        if (
            isinstance(subscr.value.value, cst.Name)
            and subscr.value.value.value == "jaxtyping"
        ):
            dtype = subscr.value.attr.value
        else:
            return None
    else:
        return None

    if len(subscr.slice) != 2:
        warnings.warn(
            f"Detected valid pattern {dtype}[...] but found an invalid number of "
            f"slices {len(subscr.slice)} != 2.",
            stacklevel=4,
        )
        return None
    arr_type_subs_el, dims_subs_el = subscr.slice
    # Instead of ignoring types we could assert hasattr(..., "value"), but that
    # wouldn't be the pythonic way
    arr_type = arr_type_subs_el.slice.value.value  # type: ignore
    dims = dims_subs_el.slice.value.value  # type: ignore
    return (
        f"[{dims[1:-1].strip().replace(' ', ', ')}] {dtype.lower()} {arr_type.lower()}"
    )
