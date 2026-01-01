"""Text that came from a file the user supplied, made safe to print on a terminal."""
from __future__ import annotations

NAME_LIMIT = 120  # file names
CELL_LIMIT = 40  # header values and keys


def escape_control(text, keep_newline: bool = True) -> str:
    """``text`` with every control, non-printable and format character written as a Python escape.

    A terminal can be driven by such characters (an escape sequence can set the window title), so nothing of
    the sort reaches the console. Ordinary printable characters, including non-ASCII letters, are kept.
    """
    out = []
    for ch in str(text):
        if ch == "\n" and keep_newline:
            out.append(ch)
        elif ch == "\\":
            out.append(ch)
        elif ch.isprintable():
            out.append(ch)
        else:
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}" if code < 0x10000 else f"\\U{code:08x}")
    return "".join(out)


def clip(text, limit: int) -> str:
    """Escaped ``text`` cut to ``limit`` characters (a cut is marked with ``...``)."""
    escaped = escape_control(text, keep_newline=False)
    return escaped if len(escaped) <= limit else escaped[:limit] + "..."


def shown(value, limit: int = CELL_LIMIT) -> str:
    """``repr`` of a value as it may be echoed in a message: control characters escaped, at most ``limit``
    characters."""
    return repr(value)[:limit]


FULL_EDITION_URL = "https://labworkspace.gumroad.com/l/labelfence"


def full_edition_only(subject: str) -> str:
    """The one sentence the lite edition uses for anything that belongs to the full edition."""
    return f"{subject} is part of the full edition of Labelfence. See {FULL_EDITION_URL}"
