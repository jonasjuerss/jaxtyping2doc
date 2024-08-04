import docstring_parser
import libcst as cst
from libcst import (
    FunctionDef,
    ImportFrom,
)
from libcst.metadata import ScopeProvider

from jaxtypingdoc.libcst_utils import set_docstring, parse_annotation
from jaxtypingdoc.utils import update_annotated_desc, apply_indent, min_indent


class JaxTypingTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.modified: bool = False

    def leave_FunctionDef(
        self, node: FunctionDef, updated_node: FunctionDef
    ) -> FunctionDef:
        docstr = node.get_docstring(clean=False)
        if docstr is None or docstr.count("\n") <= 1:
            return updated_node

        scope = self.get_metadata(ScopeProvider, node)
        # It would probably be cleaner to generate this only once per scope using some
        # custom visitor. Maybe there's even a better way provided by libcst
        jaxtyping_imports = (
            set()
            if scope is None
            else {
                a.name
                for a in scope.assignments
                if hasattr(a, "node")
                and isinstance(a.node, ImportFrom)
                and a.node.module is not None
                and a.node.module.value == "jaxtyping"
            }
        )

        # TODO check where imported from
        param_annotations = {
            param.name.value: parsed_annotation
            for param in node.params.params
            if (
                parsed_annotation := parse_annotation(
                    param.annotation, jaxtyping_imports
                )
            )
            is not None
        }

        return_annotation = parse_annotation(node.returns, jaxtyping_imports)

        if not param_annotations and not return_annotation:
            return updated_node

        # We could definitely be more precise and check if we actually updated
        # individual docstrings. However, this is just an efficiency measure as we
        # check if the overall string changed in the end anyway.
        self.modified = True

        parsed_docstring = docstring_parser.parse(docstr)
        for param in parsed_docstring.params:
            if param.arg_name in param_annotations:
                param.description = update_annotated_desc(
                    param.description or "", param_annotations[param.arg_name]
                )
        if return_annotation is not None and parsed_docstring.returns is not None:
            parsed_docstring.returns.description = update_annotated_desc(
                parsed_docstring.returns.description or "", return_annotation
            )

        new_docstr = f'\n{docstring_parser.compose(parsed_docstring)}\n"""'
        new_docstr = f'"""{apply_indent(new_docstr, min_indent(docstr))}'
        return updated_node.with_changes(
            body=set_docstring(updated_node.body, new_docstr)
        )

    # def leave_SimpleString(
    #         self, original_node: SimpleString, updated_node: SimpleString
    # ) -> BaseExpression:
    #     """Trying to find the node defining the docstring
    #     of your function, and update the docstring"""
    #     # if original_node.value == self._docstring:
    #     #     return updated_node.with_changes(value='"""My new docstring"""')
    #
    #     return updated_node
