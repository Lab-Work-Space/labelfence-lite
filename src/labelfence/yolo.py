"""The YOLO reader and writer (Ultralytics, split-first and flat layouts) and a minimal ``data.yaml`` reader."""
from __future__ import annotations

import math
import os
import pathlib
import re
import string
import sys

from .errors import UnsupportedInput
from . import images as _images
from .images import (IMAGE_SUFFIXES, MAX_PIXELS_DEFAULT, ImageProblem, image_header, read_text_bounded,
                     text_cap_reason, walk)
from ._plan import ImageCopier, plan_images
from .model import Box, Dataset, Finding, ImageRecord, Split, finding
from .text import clip

LAYOUTS = ("ultralytics", "split-first", "flat")
SPLITS = ("train", "val", "valid", "test")  # "valid" is the Roboflow spelling of the validation split
UNSPLIT = "unsplit"  # the folder of the unnamed split when it sits next to named ones
SPLIT_DIRS = (*SPLITS, UNSPLIT)
ROLE_KEYS = ("train", "val", "test")  # the keys of data.yaml that name the role of a folder
MAX_ROLE_VALUES = 1000  # values read from one role key
MAX_LIST_LINES = 200_000  # lines read from a list file named by a role key
_STANDARD_SPLIT = re.compile(r"^(train|val|valid|test)[0-9]*$")
_SPLIT_RANK = {"train": 0, "val": 1, "valid": 2, "test": 3}
TOLERANCE = 1e-9
MAX_CLASS = 100_000  # the largest class id or class count read
MAX_DIGITS = 9  # a longer whole number is refused before int() sees it
MAX_LINE = 4096  # characters in one line of data.yaml or of a label file; a longer line is a finding, never parsed
MAX_LABEL_LINES = 100_000  # lines read from one label file; a longer file is one finding (memory stays bounded)
_KEY_START = frozenset(string.ascii_letters + "_")
_KEY_REST = frozenset(string.ascii_letters + string.digits + "_-. ")
_INT = re.compile(r"^[+-]?[0-9]+$")
_UINT = re.compile(r"^[0-9]+$")
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "r": "\r", "0": "\0", "/": "/"}


# ---------------------------------------------------------------- data.yaml (a minimal subset)

class _Bad(Exception):
    pass


def _match_key(content: str):
    """``(key, value)`` of a ``key: value`` line, or ``None``; one pass, no backtracking.

    The key starts with a letter or ``_`` and holds letters, digits, ``_ - .`` and spaces; the colon is followed
    by the end of the line or by white space.
    """
    if not content or content[0] not in _KEY_START:
        return None
    end = 1
    while end < len(content) and content[end] in _KEY_REST:
        end += 1
    colon = end
    while colon < len(content) and content[colon].isspace():
        colon += 1
    if colon >= len(content) or content[colon] != ":":
        return None
    rest = content[colon + 1:]
    if rest and not rest[0].isspace():
        return None
    return content[:end].rstrip(" "), rest.strip()


def _strip_comment(line: str) -> str:
    out_to = len(line)
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        prev = line[i - 1] if i else " "
        if quote == "'":
            if ch == "'":
                if line[i + 1:i + 2] == "'":
                    i += 1
                else:
                    quote = None
        elif quote == '"':
            if ch == "\\":
                i += 1
            elif ch == '"':
                quote = None
        elif ch in "'\"" and prev in " \t[{,:":
            quote = ch
        elif ch == "#" and prev in " \t":
            out_to = i
            break
        i += 1
    return line[:out_to].rstrip()


def _split_top(text: str, sep: str) -> list[str]:
    parts, cur, quote, i = [], [], None, 0
    blank = True  # nothing but white space in ``cur`` so far (kept as a flag: joining ``cur`` again and again is quadratic)
    while i < len(text):
        ch = text[i]
        if quote == "'":
            cur.append(ch)
            if ch == "'":
                if text[i + 1:i + 2] == "'":
                    cur.append("'")
                    i += 1
                else:
                    quote = None
        elif quote == '"':
            cur.append(ch)
            if ch == "\\" and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == '"':
                quote = None
        elif ch in "'\"" and blank:
            quote = ch
            blank = False
            cur.append(ch)
        elif ch == sep:
            parts.append("".join(cur))
            cur = []
            blank = True
        else:
            cur.append(ch)
            if not ch.isspace():
                blank = False
        i += 1
    if quote:
        raise _Bad("an unclosed quote")
    parts.append("".join(cur))
    return parts


def _scalar(token: str) -> str:
    token = token.strip()
    if not token:
        raise _Bad("an empty value")
    if token[0] == '"':
        if len(token) < 2 or token[-1] != '"':
            raise _Bad("an unclosed quote")
        body, out, i = token[1:-1], [], 0
        while i < len(body):
            ch = body[i]
            if ch == '"':
                raise _Bad("an unescaped quote inside a quoted value")
            if ch == "\\":
                i += 1
                nxt = body[i:i + 1]
                if nxt in _ESCAPES:
                    out.append(_ESCAPES[nxt])
                elif nxt in ("x", "u"):
                    width = 2 if nxt == "x" else 4
                    digits = body[i + 1:i + 1 + width]
                    if len(digits) != width or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                        raise _Bad("an unknown escape")
                    out.append(chr(int(digits, 16)))
                    i += width
                else:
                    raise _Bad("an unknown escape")
            else:
                out.append(ch)
            i += 1
        return "".join(out)
    if token[0] == "'":
        if len(token) < 2 or token[-1] != "'":
            raise _Bad("an unclosed quote")
        body = token[1:-1].replace("''", "\0")
        if "'" in body:
            raise _Bad("an unescaped quote inside a quoted value")
        return body.replace("\0", "'")
    if token[0] in "[]{}*&!|>%@`,?":
        raise _Bad("a value this reader does not support")
    if ": " in token or token.endswith(":"):
        raise _Bad("a nested mapping where a name was expected")
    return token


def _flow_list(text: str) -> list[str]:
    if not text.endswith("]"):
        raise _Bad("an unclosed list")
    inner = text[1:-1]
    if not inner.strip():
        return []
    parts = _split_top(inner, ",")
    if parts and not parts[-1].strip():
        parts.pop()
    return [_scalar(p) for p in parts]


def _key_value(text: str, what: str) -> tuple[str, str]:
    pieces = _split_top(text, ":")
    if len(pieces) < 2:
        raise _Bad(f"{what} without a colon")
    key = pieces[0].strip()
    value = ":".join(pieces[1:])
    if value and not value[0].isspace():
        raise _Bad(f"{what} without a space after the colon")
    return key, value.strip()


def _names_from_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    """Class names from ``number: name`` pairs; a number that is not given becomes ``class_<n>``."""
    numbered = {}
    for key, value in pairs:
        if not _UINT.match(key):
            raise _Bad("a class key that is not a number")
        if len(key) > MAX_DIGITS or int(key) >= MAX_CLASS:
            raise _Bad(f"a class number above the supported maximum of {MAX_CLASS - 1}")
        if int(key) in numbered:
            raise _Bad("a class number given twice")
        numbered[int(key)] = _scalar(value)
    count = max(numbered) + 1 if numbered else 0
    return [numbered.get(i, f"class_{i}") for i in range(count)]


def _role_values(value: str, block) -> list[str]:
    """The strings of a ``path``, ``train``, ``val`` or ``test`` key: a scalar, a flow list or a block list. What
    cannot be read gives an empty list (these keys only name folders; they are never a finding by themselves)."""
    try:
        if value:
            if block:
                return []
            items = _flow_list(value) if value.startswith("[") else [_scalar(value)]
        else:
            items = []
            for _, _, content in block:
                if not (content == "-" or content.startswith("- ")):
                    return []
                items.append(_scalar(content[1:]))
    except _Bad:
        return []
    return [item for item in items[:MAX_ROLE_VALUES] if item and item not in ("null", "~")]


def read_data_yaml(path, roles=None) -> tuple[list[str] | None, int | None, list[Finding]]:
    """``(names, nc, findings)`` from the ``names`` and ``nc`` keys of a ``data.yaml``.

    When ``roles`` is a dict it is filled with the strings of the ``path``, ``train``, ``val`` and ``test`` keys
    (``{"train": ["images/train"], ...}``).

    A subset of YAML: top-level keys; ``names`` as a flow list, a block list or a mapping of numbers to names
    (block or flow); quoted or bare scalars; comments. Other top-level keys are ignored together with their
    nested lines. What the subset cannot read is a finding (``E_LABEL_MALFORMED``) with the line number.
    """
    path = pathlib.Path(path)
    findings: list[Finding] = []

    def bad(line, why):
        findings.append(finding("E_LABEL_MALFORMED", "", path.name,
                                f"data.yaml: line {line}: {why}. Expected a list or a numbered mapping of "
                                "class names under 'names' and a whole number under 'nc'.", line))

    try:
        text = read_text_bounded(path)
    except UnsupportedInput:
        bad(None, text_cap_reason())
        return None, None, findings
    except UnicodeDecodeError:
        bad(None, "the file is not UTF-8 text")
        return None, None, findings
    except OSError:
        bad(None, "the file cannot be read")
        return None, None, findings

    items = []
    for number, raw in enumerate(text.split("\n"), 1):
        if len(raw) > MAX_LINE:
            findings.append(finding("E_LABEL_MALFORMED", "", path.name,
                                    f"data.yaml: line {number} is longer than {MAX_LINE} characters; it was not read.",
                                    number))
            continue
        content = _strip_comment(raw)
        if content.strip():
            items.append((number, len(content) - len(content.lstrip(" ")), content.strip(" ")))
    names: list[str] | None = None
    nc: int | None = None
    i = 0
    while i < len(items):
        number, indent, content = items[i]
        i += 1
        block = []
        while i < len(items) and (items[i][1] > 0 or items[i][2] == "-" or items[i][2].startswith("- ")):
            block.append(items[i])
            i += 1
        if indent > 0 and number == items[0][0]:
            bad(number, "unexpected indentation")
            continue
        if content in ("---", "..."):
            continue
        if content.startswith("\t") or content.startswith("%"):
            bad(number, "unsupported syntax")
            continue
        match = _match_key(content)
        if match is None:
            bad(number, "this line is not 'key: value'")
            continue
        key, value = match
        if key == "nc":
            if block or not _UINT.match(value):
                bad(number, "'nc' must be a whole number, 0 or more")
            elif len(value) > MAX_DIGITS or int(value) > MAX_CLASS:
                bad(number, f"'nc' is above the supported maximum of {MAX_CLASS}")
            else:
                nc = int(value)
        elif key == "names":
            names = _read_names(value, block, number, bad)
        elif roles is not None and key in ("path", *ROLE_KEYS):
            values = _role_values(value, block)
            if values:
                roles[key] = values
    return names, nc, findings


def _read_names(value, block, number, bad):
    try:
        if value:
            if block:
                bad(block[0][0], "unexpected indented line under a one-line value")
                return None
            if value.startswith("["):
                return _flow_list(value)
            if value.startswith("{"):
                if not value.endswith("}"):
                    raise _Bad("an unclosed mapping")
                inner = value[1:-1]
                if not inner.strip():
                    return []
                parts = _split_top(inner, ",")
                if not parts[-1].strip():
                    parts.pop()
                return _names_from_pairs([_key_value(p, "an entry") for p in parts])
            raise _Bad("'names' must be a list or a numbered mapping")
        if not block:
            raise _Bad("'names' has no value")
    except _Bad as why:
        bad(number, str(why))
        return None
    base = block[0][1]
    first = block[0][2]
    if first == "-" or first.startswith("- "):
        out = []
        for line, indent, content in block:
            if indent != base or not (content == "-" or content.startswith("- ")):
                bad(line, "a list item was expected here")
                return None
            try:
                out.append(_scalar(content[1:]))
            except _Bad as why:
                bad(line, str(why))
                return None
        return out
    pairs, nested = [], False
    for line, indent, content in block:
        if indent > base and nested:
            continue
        if indent != base:
            bad(line, "unexpected indentation")
            return None
        nested = False
        try:
            key, val = _key_value(content, "an entry")
            if not val:
                nested = True
                raise _Bad("a nested mapping where a class name was expected")
            if not _UINT.match(key):
                raise _Bad("a class key that is not a number")
            pairs.append((key, val))
        except _Bad as why:
            bad(line, str(why))
            return None
    try:
        return _names_from_pairs(pairs)
    except _Bad as why:
        bad(block[0][0], str(why))
        return None


# ---------------------------------------------------------------- layouts

def _subdirs(folder: pathlib.Path) -> set[str]:
    try:
        return {e.name for e in os.scandir(folder) if not e.name.startswith(".") and e.is_dir()}
    except OSError:
        return set()


def split_folders(root, layout: str) -> set[str]:
    """The names of the folders that hold splits: the sub-folders of ``images`` and ``labels``
    (``ultralytics``) or the top-level folders that have an ``images`` or ``labels`` folder (``split-first``)."""
    root = pathlib.Path(root)
    top = _subdirs(root)
    if layout == "ultralytics":
        return {name for kind in ("images", "labels") if kind in top for name in _subdirs(root / kind)}
    if layout == "split-first":
        return {name for name in top - {"images", "labels"} if {"images", "labels"} & _subdirs(root / name)}
    return set()


def detect_layout(root) -> str | None:
    """``"ultralytics"`` (``images/<split>`` and ``labels/<split>``, a split being any sub-folder),
    ``"split-first"`` (``<split>/images`` and ``<split>/labels``), ``"flat"`` (``images`` and ``labels``), or
    ``None``."""
    root = pathlib.Path(root)
    top = _subdirs(root)
    if any(_subdirs(root / kind) for kind in ("images", "labels") if kind in top):
        return "ultralytics"
    if split_folders(root, "split-first"):
        return "split-first"
    if {"images", "labels"} & top:
        return "flat"
    return None


def canonical_split(folder: str) -> str:
    """The split name a folder stands for when nothing else names it: ``unsplit`` and no folder at all are the
    unnamed split ``""``; ``train2017`` is ``train`` (the way the COCO reader names ``instances_train2017.json``);
    every other folder keeps its name."""
    if folder in ("", UNSPLIT):
        return ""
    match = _STANDARD_SPLIT.match(folder)
    return match.group(1) if match else folder


def split_order(name: str) -> tuple:
    """The sort key of split names: train, val, valid, test, then every other name alphabetically, then ``""``."""
    return (_SPLIT_RANK.get(name, 4 if name else 5), name)


def _place(layout: str, parts: tuple[str, ...]):
    """``(folder, kind, sub-parts)`` for a file's relative path, or ``None`` when it is not part of the data.

    ``folder`` is the split folder (``""`` for files directly in ``images`` or ``labels``)."""
    if layout == "ultralytics":
        if parts[0] in ("images", "labels"):
            if len(parts) >= 3:
                return parts[1], parts[0], parts[2:]
            if len(parts) == 2:
                return "", parts[0], parts[1:]
    elif layout == "split-first":
        if len(parts) >= 3 and parts[0] not in ("images", "labels") and parts[1] in ("images", "labels"):
            return parts[0], parts[1], parts[2:]
    elif len(parts) >= 2 and parts[0] in ("images", "labels"):
        return "", parts[0], parts[1:]
    return None


def _inside(value: str, base: tuple[str, ...]):
    """The parts of the path ``value`` (relative to the folder ``base``, itself relative to the dataset root) as
    a tuple relative to the root, or ``None`` when it is absolute or leaves the root."""
    text = value.replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return None
    parts = list(base)
    for segment in text.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(segment)
    return tuple(parts)


def _folder_of(parts: tuple[str, ...], layout: str, folders: set[str]):
    """The split folder a path (to a folder or an image) lies in, or ``None``."""
    if layout == "ultralytics":
        if len(parts) >= 2 and parts[0] in ("images", "labels") and parts[1] in folders:
            return parts[1]
    elif layout == "split-first":
        if parts and parts[0] in folders and (len(parts) == 1 or parts[1] in ("images", "labels")):
            return parts[0]
    return None


def _role_folders(raw: dict, layout: str, folders: set[str], files: dict, findings: list) -> dict[str, str]:
    """``{folder: role}``: which of ``train``, ``val`` and ``test`` of ``data.yaml`` names which folder.

    Values are read relative to ``path:`` when that stays inside the dataset root, else relative to the folder of
    ``data.yaml`` (the root); a ``path:`` that leaves the root is the author's location, not this one, and is
    ignored. A value that leaves the root is tried once more without its leading ``../`` (the Roboflow keys are
    written ``../train/images`` for a folder that sits at ``train/images``); when that finds nothing, the key is
    ignored with ``W_PATH_OUTSIDE_ROOT``. A value ending in ``.txt`` is a list of image paths: the folders its
    lines lie in get the role. Nothing outside the root is ever read."""
    bases: list[tuple[str, ...]] = [()]
    given = raw.get("path")
    if given:
        inner = _inside(given[0], ())
        if inner:
            bases.insert(0, inner)
    roles: dict[str, str] = {}

    def assign(folder, role):
        roles.setdefault(folder, role)

    for role in ROLE_KEYS:
        for value in raw.get(role, []):
            matched = False
            escaped = True
            for base in bases:
                parts = _inside(value, base)
                if parts is None:
                    continue
                escaped = False
                if value.lower().endswith(".txt"):
                    target = files.get("/".join(parts))
                    if target is None:
                        continue
                    try:
                        text = read_text_bounded(target)
                    except (UnsupportedInput, UnicodeDecodeError, OSError):
                        continue
                    for line in text.split("\n", MAX_LIST_LINES)[:MAX_LIST_LINES]:
                        for line_base in bases:
                            image = _inside(line.strip(), line_base) if line.strip() else None
                            folder = _folder_of(image, layout, folders) if image else None
                            if folder:
                                assign(folder, role)
                                matched = True
                                break
                    if matched:
                        break
                    continue
                folder = _folder_of(parts, layout, folders)
                if folder:
                    assign(folder, role)
                    matched = True
                    break
            if matched or not escaped:
                continue
            stripped = value.replace("\\", "/")
            while stripped.startswith("../"):
                stripped = stripped[3:]
            parts = _inside(stripped, ()) if not stripped.startswith("..") else None
            folder = _folder_of(parts, layout, folders) if parts else None
            if folder:
                assign(folder, role)
            else:
                findings.append(finding("W_PATH_OUTSIDE_ROOT", "", "data.yaml",
                                        f"the data.yaml key '{role}' points outside the dataset folder; "
                                        "it was ignored."))
    return roles


# ---------------------------------------------------------------- labels

def _number(text: str) -> float:
    if "_" in text:
        raise ValueError(text)
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


class _TooManyLines(Exception):
    pass


def _label_lines(path):
    """``(line number, text)`` of a label file, one line at a time; the text is ``None`` for a line longer than
    ``MAX_LINE`` characters (the rest of that line is skipped, never held). Blank lines are not returned. Line
    endings are normalised, a BOM is dropped. Raises ``UnsupportedInput`` (over the size cap), ``_TooManyLines``, ``UnicodeDecodeError`` or
    ``OSError``; nothing is read beyond the cap."""
    with open(path, "r", encoding="utf-8-sig", newline=None) as handle:
        if os.fstat(handle.fileno()).st_size > _images.MAX_TEXT_BYTES:
            raise UnsupportedInput(text_cap_reason())
        number = read = counted = 0
        while True:
            line = handle.readline(MAX_LINE + 1)
            if not line:
                return
            number += 1
            read += len(line)
            if read > _images.MAX_TEXT_BYTES:  # the file grew while it was read
                raise UnsupportedInput(text_cap_reason())
            if not line.strip():
                continue  # a blank line costs nothing to keep and is not counted
            counted += 1
            if counted > MAX_LABEL_LINES:
                raise _TooManyLines
            too_long = len(line) > MAX_LINE and not line.endswith("\n")
            while too_long and line and not line.endswith("\n"):  # skip the rest of the line
                line = handle.readline(MAX_LINE + 1)
                read += len(line)
            if read > _images.MAX_TEXT_BYTES:
                raise UnsupportedInput(text_cap_reason())
            yield number, None if too_long else line


def _parse_label(lines, split, rel, width, height, nc, allow_polygons, seen_ids):
    """``(boxes, findings)`` of one label file given as ``(number, text)`` pairs. A box is kept only when its
    line has no finding."""
    boxes: list[Box] = []
    findings: list[Finding] = []
    scale_x, scale_y = (float(width), float(height)) if width else (1.0, 1.0)
    nonblank = 0
    for number, raw in lines:
        if raw is None:
            nonblank += 1
            findings.append(finding("E_LABEL_MALFORMED", split, rel,
                                    f"line {number} is longer than {MAX_LINE} characters; it was not read.", number))
            continue
        fields = raw.split()
        if not fields:
            continue
        nonblank += 1

        def add(code, message, _n=number):
            findings.append(finding(code, split, rel, f"line {_n}: {message}", _n))

        count = len(fields)
        if count < 5:
            add("E_LABEL_MALFORMED", f"found {count} fields, expected 5 (class x_center y_center width height).")
            continue
        if count > 5 and not allow_polygons:
            add("E_LABEL_MALFORMED", f"polygon labels are not supported (found {count} fields, expected 5). "
                                     "Use --allow-polygons to take their bounding box.")
            continue
        ok = True
        class_id = None
        token = fields[0]
        if _INT.match(token) and len(token.lstrip("+-")) > MAX_DIGITS:
            add("E_CLASS_ID_INVALID", f"field 1 (the class id) is above the supported maximum of {MAX_CLASS}.")
            ok = False
        elif _INT.match(token):
            class_id = int(token)
            if class_id > MAX_CLASS:
                add("E_CLASS_ID_INVALID", f"field 1 (the class id) is above the supported maximum of {MAX_CLASS}.")
                ok = False
            elif class_id < 0:
                add("E_CLASS_ID_INVALID", "field 1 (the class id) is negative; expected 0 or more.")
                ok = False
            elif nc is not None and class_id >= nc:
                add("E_CLASS_ID_INVALID", f"field 1 (the class id) is not below the class count {nc}.")
                ok = False
            if 0 <= class_id <= MAX_CLASS:
                seen_ids.append(class_id)
        else:
            add("E_CLASS_ID_INVALID", "field 1 (the class id) is not a whole number.")
            ok = False
        try:
            values = [_number(f) for f in fields[1:]]
        except ValueError:
            add("E_LABEL_MALFORMED", "a coordinate is not a number; expected decimal numbers from 0 to 1.")
            continue
        if count == 5:
            xc, yc, w, h = values
            out_of_range = [i for i, v in enumerate(values, 2) if v < -TOLERANCE or v > 1 + TOLERANCE]
        else:
            if (count - 1) % 2 or count - 1 < 6:
                add("E_LABEL_MALFORMED", f"a polygon needs an even number of at least 6 coordinates, "
                                         f"found {count - 1}.")
                continue
            xs, ys = values[0::2], values[1::2]
            out_of_range = [i for i, v in enumerate(values, 2) if v < -TOLERANCE or v > 1 + TOLERANCE]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if out_of_range:
            add("E_BOX_OUT_OF_RANGE", f"field {out_of_range[0]} is outside 0 to 1.")
            continue
        if w <= 0 or h <= 0:
            add("E_BOX_DEGENERATE", "width and height must both be above 0.")
            continue
        if ok:
            boxes.append(Box(class_id, (xc - w / 2) * scale_x, (yc - h / 2) * scale_y,
                             (xc + w / 2) * scale_x, (yc + h / 2) * scale_y, number))
    if not nonblank:
        findings.append(finding("W_LABEL_EMPTY", split, rel, "the label file has no boxes."))
    return boxes, findings


def _read_label(path, split, rel, width, height, nc, allow_polygons, seen_ids):
    """``(boxes, findings, problem)`` of a label file, read line by line. ``problem`` is a finding when the file
    cannot be read as a whole (not UTF-8, over a cap, unreadable); the boxes and findings are then empty."""
    ids: list[int] = []
    try:
        boxes, findings = _parse_label(_label_lines(path), split, rel, width, height, nc, allow_polygons, ids)
    except UnsupportedInput:
        why = text_cap_reason()
    except _TooManyLines:
        why = (f"the file has more than {MAX_LABEL_LINES} non-blank lines, which is the most Labelfence reads from one label "
               "file; it was not read.")
    except UnicodeDecodeError:
        why = "the file is not UTF-8 text."
    except OSError:
        why = "the file cannot be read."
    else:
        seen_ids.extend(ids)
        return boxes, findings, None
    return [], [], finding("E_LABEL_MALFORMED", split, rel, why)


def _class_list(path):
    """The names in a ``classes.txt`` (one per line, blank lines skipped), or ``None`` when it cannot be read."""
    try:
        lines = [line.strip() for line in read_text_bounded(path).split("\n")]
    except (UnsupportedInput, UnicodeDecodeError, OSError):
        return None
    names = [line for line in lines if line]
    return names if 0 < len(names) <= MAX_CLASS else None


def read_yolo(root, *, allow_polygons: bool = False, max_pixels: int = MAX_PIXELS_DEFAULT,
              follow_links: bool = False) -> Dataset:
    """Read a YOLO dataset into the model. Bad files become findings; only a root that is not a folder, or
    that has no recognised layout, raises ``UnsupportedInput``.

    Every sub-folder of ``images`` (or every ``<split>/images`` folder) is a split. A split is named after its
    folder (``train2017`` is ``train``), or after the role that ``train``, ``val`` or ``test`` of ``data.yaml``
    gives it. ``classes.txt`` is a class list, never a label file; it names the classes when ``data.yaml``
    does not. With ``follow_links`` a link to an image outside the folder is read (read only)."""
    given = pathlib.Path(root)
    if not given.is_dir():
        raise UnsupportedInput(f"{clip(given, 200)}: not a folder. Expected the root folder of a YOLO dataset.")
    base = given.resolve()
    layout = detect_layout(base)
    if layout is None:
        raise UnsupportedInput(f"{clip(given, 200)}: no YOLO layout found. Expected images/<split> and "
                               "labels/<split>, <split>/images and <split>/labels, or images and labels.")
    walked = walk(base, follow_images=follow_links)
    findings: list[Finding] = []
    entries: dict[str, dict] = {}  # folder -> {"images": [...], "labels": {...}, "unusual": {...}, "outside": set()}
    unusual_files: list[tuple[str, str]] = []  # (folder, relative path)
    class_files: list[tuple[str, pathlib.Path]] = []
    yaml_path = None

    def bucket_of(folder):
        return entries.setdefault(folder, {"images": [], "labels": {}, "unusual": {}, "outside": set()})

    def key_of(sub):
        return "/".join(sub[:-1] + (pathlib.PurePosixPath(sub[-1]).with_suffix("").name,))

    for file in walked.files:
        rel = file.relative_to(base)
        if rel.as_posix() == "data.yaml":
            yaml_path = file
            continue
        placed = _place(layout, rel.parts)
        if rel.name == "classes.txt" and (len(rel.parts) == 1 or (placed and placed[1] == "labels"
                                                                  and len(placed[2]) == 1)):
            class_files.append((rel.as_posix(), file))  # LabelImg and Label Studio write the class list here
            continue
        if placed is None:
            continue
        folder, kind, sub = placed
        suffix = pathlib.PurePosixPath(sub[-1]).suffix.lower()
        key = key_of(sub)
        bucket = bucket_of(folder)
        if kind == "images" and suffix in IMAGE_SUFFIXES:
            bucket["images"].append((key, rel.as_posix(), file))
        elif kind == "labels" and suffix == ".txt":
            bucket["labels"].setdefault(key, (rel.as_posix(), file))
        elif kind == "images" and suffix != ".txt":
            bucket["unusual"][key] = rel.as_posix()
            unusual_files.append((folder, rel.as_posix()))
    for link in walked.outside:  # an image that is a link out of the folder explains its label: no orphan
        placed = _place(layout, link.parts)
        if placed and placed[1] == "images":
            bucket_of(placed[0])["outside"].add(key_of(placed[2]))
    class_files.sort()

    # data.yaml: names, nc and the roles of the folders
    names = declared = None
    raw_roles: dict = {}
    if yaml_path is not None:
        names, declared, yfound = read_data_yaml(yaml_path, raw_roles)
        findings.extend(yfound)
    on_disk = split_folders(base, layout)
    roles = _role_folders(raw_roles, layout, on_disk | set(entries), {f.relative_to(base).as_posix(): f
                                                                    for f in walked.files}, findings) \
        if raw_roles else {}

    def name_of(folder: str) -> str:
        reported = canonical_split(folder)
        return reported if reported in _SPLIT_RANK or not folder or folder == UNSPLIT else roles.get(folder, reported)

    blind: set = set()  # (split folder, "images" | "labels") of a folder that could not be read
    for folder_rel in walked.unreadable:
        placed = _place(layout, folder_rel.parts + ("_",))
        if not placed and folder_rel.parts[0] not in on_disk | {"images", "labels"}:
            continue  # a folder that is no part of the data
        blind.add(placed[:2] if placed else None)
        findings.append(finding("E_LABEL_MALFORMED", name_of(placed[0]) if placed else "", folder_rel.as_posix(),
                                "the folder cannot be read (permission denied)"))
    for link in walked.outside:
        split = next((name_of(part) for part in link.parts[:2] if part in on_disk), "")
        findings.append(finding("W_PATH_OUTSIDE_ROOT", split, link.as_posix(),
                                "a symbolic link that resolves outside the dataset folder; it was skipped."))
    left_out: list[pathlib.PurePosixPath] = []
    for folder, rel in unusual_files:
        findings.append(finding("W_EXTENSION_UNUSUAL", name_of(folder), rel,
                                "the extension is not one of the image extensions Labelfence reads "
                                f"({', '.join(sorted(IMAGE_SUFFIXES))}); the file is not audited as an image."))
    if names is None and class_files:
        names = _class_list(class_files[0][1])
    nc = declared if declared is not None else (len(names) if names is not None else None)
    if yaml_path is not None and names is not None and declared is not None and len(names) != declared:
        findings.append(finding("E_NAMES_MISMATCH", "", "data.yaml",
                                f"'nc' says {declared} classes but 'names' lists {len(names)}."))
    if yaml_path is not None and names is not None:
        first_entry: dict[str, int] = {}
        for index, name in enumerate(names, 1):  # a name listed twice: two class ids for one class
            if name in first_entry:
                findings.append(finding("E_NAMES_MISMATCH", "", "data.yaml",
                                        f"'names' lists a class name twice (entry {index} repeats entry "
                                        f"{first_entry[name]}); a name must belong to one class."))
            else:
                first_entry[name] = index
    seen_ids: list[int] = []
    by_name: dict[str, Split] = {}
    for folder in sorted(entries, key=lambda f: (split_order(name_of(f)), f)):
        split = name_of(folder)
        bucket = entries[folder]
        groups: dict[str, int] = {}
        for key, _, _ in bucket["images"]:
            groups[key] = groups.get(key, 0) + 1
        attached: set[str] = set()
        records: list[ImageRecord] = []
        for key, rel, file in bucket["images"]:
            if groups[key] > 1:
                findings.append(finding("W_IMAGE_NAME_COLLISION", split, rel,
                                        "another image in this split has the same base name; "
                                        "their labels cannot be told apart. The label is used for the first."))
            orientation = 1
            try:
                width, height, orientation = image_header(file, max_pixels)
            except ImageProblem as problem:
                findings.append(finding(problem.code, split, rel, problem.message))
                width = height = None
            later_duplicate = key in attached  # its label went to the first image of that name
            label = None if later_duplicate else bucket["labels"].get(key)
            attached.add(key)
            boxes: list[Box] = []
            label_rel = None
            unreadable = False
            if label is None and later_duplicate:
                pass  # already reported as a name collision
            elif label is None:
                if (folder, "labels") not in blind:  # a folder nobody could read has no labels to miss
                    findings.append(finding("W_IMAGE_WITHOUT_LABEL", split, rel, "no label file for this image."))
            else:
                label_rel, label_file = label
                boxes, found, problem_finding = _read_label(label_file, split, label_rel, width, height, nc,
                                                            allow_polygons, seen_ids)
                findings.extend(found)
                if problem_finding:
                    findings.append(problem_finding)
                    unreadable = True
            records.append(ImageRecord(pathlib.PurePosixPath(rel), width, height, boxes,
                                       pathlib.PurePosixPath(label_rel) if label_rel else None, unreadable,
                                       orientation))
        for key, (label_rel, _) in sorted(bucket["labels"].items(), key=lambda kv: kv[1][0]):
            if key in bucket["unusual"] and key not in attached:  # an unusual image is reported already
                left_out.append(pathlib.PurePosixPath(bucket["unusual"][key]))
            elif key not in attached and key not in bucket["outside"] and (folder, "images") not in blind:
                findings.append(finding("E_LABEL_MISSING_IMAGE", split, label_rel, "no image for this label file."))
        by_name.setdefault(split, Split(split, [])).images.extend(records)

    if names is not None:
        classes = list(names)
    elif nc is not None:
        classes = [f"class_{i}" for i in range(nc)]
    else:
        classes = [f"class_{i}" for i in range(max(seen_ids) + 1)] if seen_ids else []
    left_out.sort(key=lambda p: p.as_posix())
    return Dataset(given, "yolo", layout, classes, list(by_name.values()), findings, walked.hidden, walked.outside,
                   walked.followed, left_out)


# ---------------------------------------------------------------- writer

def _quote(name: str) -> str:
    out = ['"']
    for ch in name:
        if ch in '\\"':
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif not ch.isprintable():
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}" if code < 0x10000 else "?")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def write_yolo(dataset: Dataset, out_dir, *, link: bool = False, rename=None) -> None:
    """Write ``dataset`` as ``data.yaml``, ``images/<split>/`` and ``labels/<split>/`` into ``out_dir``, copying
    the images (hard-linking with ``link``, which falls back to a copy). An unnamed split alone is written to
    ``images/`` and ``labels/``; next to named splits it goes to ``images/unsplit`` and ``labels/unsplit`` (the
    reader takes that folder as the unnamed split). Nothing existing is replaced: an existing file raises
    ``FileExistsError``. Everything is checked before the first file is written. ``rename`` maps an image path to the file
    name it is written under. The caller owns ``out_dir``."""
    out = pathlib.Path(out_dir)
    plan = []
    mixed = any(s.name and s.images for s in dataset.splits)
    for split_name, rec, name, stem, source in plan_images(dataset, by_stem=True, annotation="label file", rename=rename):
        lines = None
        if rec.boxes or rec.label_path is not None:
            if rec.boxes and not (rec.width and rec.height):
                raise UnsupportedInput(f"{clip(rec.path.as_posix(), 120)}: the image size is unknown, so its "
                                       "boxes cannot be written as normalised values.")
            lines = []
            for b in rec.boxes:
                w, h = rec.width, rec.height
                lines.append(f"{b.class_id} {_clamp((b.x_min + b.x_max) / 2 / w):.6f} "
                             f"{_clamp((b.y_min + b.y_max) / 2 / h):.6f} "
                             f"{_clamp((b.x_max - b.x_min) / w):.6f} {_clamp((b.y_max - b.y_min) / h):.6f}\n")
        folder = UNSPLIT if (not split_name and mixed) else split_name
        plan.append((folder, name, stem, source, lines))

    out.mkdir(parents=True, exist_ok=True)
    yaml = [f"nc: {len(dataset.classes)}\n", "names:\n" if dataset.classes else "names: []\n"]
    yaml += [f"  {i}: {_quote(n)}\n" for i, n in enumerate(dataset.classes)]
    # only the role keys are written, and only for folders that carry a role name: any other split is found by
    # its folder (a folder called ``names`` or ``nc`` must never become a second ``names`` or ``nc`` key)
    named = {split.name for split in dataset.splits}
    for split in dataset.splits:
        if split.name in ROLE_KEYS:
            yaml.append(f"{split.name}: images/{split.name}\n")
        elif split.name == "valid" and "val" not in named:
            yaml.append("val: images/valid\n")
        elif not split.name:
            yaml.append(f"{UNSPLIT}: images/{UNSPLIT}\n" if mixed else "train: images\n")
    with open(out / "data.yaml", "x", encoding="utf-8", newline="\n") as handle:
        handle.writelines(yaml)
    copier = ImageCopier(link)
    for split_name, name, stem, source, lines in plan:
        copier.put(source, out / "images" / split_name / name if split_name else out / "images" / name)
        if lines is not None:
            label_dir = out / "labels" / split_name if split_name else out / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)
            with open(label_dir / f"{stem}.txt", "x", encoding="utf-8", newline="\n") as handle:
                handle.writelines(lines)
