import difflib


def red(text: str) -> str:
    return f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"  #


def green(text: str) -> str:
    return f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"


def blue(text: str) -> str:
    return f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"


def white(text: str) -> str:
    return f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"


def get_edits_string(old: str, new: str) -> str:
    """
    Source: https://stackoverflow.com/a/64404008/5130715
    """
    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal":
            result += white(old[code[1] : code[2]])
        elif code[0] == "delete":
            result += red(old[code[1] : code[2]])
        elif code[0] == "insert":
            result += green(new[code[3] : code[4]])
        elif code[0] == "replace":
            result += red(old[code[1] : code[2]]) + green(new[code[3] : code[4]])
    return result


def min_indent(s: str) -> int:
    """
    Determines the indent of a multiline string. Based on ``inspect.cleandoc()``.
    :param s: A (potentially multi-line) string
    :return: The minimum number of white-spaces before any line
    """
    return min(
        len(line) - content
        for line in s.expandtabs().split("\n")
        if (content := len(line.lstrip()))
    )


def apply_indent(s: str, indent: int) -> str:
    prefix = indent * " "
    return "\n".join("" if len(line) == 0 else prefix + line for line in s.split("\n"))


def update_annotated_desc(cur_desc: str, annotation: str) -> str:
    if cur_desc.startswith(annotation):
        return cur_desc
    return f"{annotation} {cur_desc}"
