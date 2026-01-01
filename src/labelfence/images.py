"""Bounded reading: image headers without decoding, a walk of the dataset root, and size-capped text files."""
from __future__ import annotations

import os
import pathlib
import stat
import warnings
from dataclasses import dataclass, field

from PIL import Image

from .errors import UnsupportedInput
from .text import clip

MAX_PIXELS_DEFAULT = 50_000_000
MAX_TEXT_MB_DEFAULT = 64
MAX_TEXT_BYTES = MAX_TEXT_MB_DEFAULT * 1024 * 1024  # the cap for a label, JSON or XML file; --max-file-mb sets it
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# The only formats Pillow is allowed to open: nothing that hands a file to another program (EPS runs Ghostscript).
# Every ``Image.open`` of both packages passes this as ``formats=``; a file of another format is unreadable.
IMAGE_FORMATS = ("JPEG", "PNG", "BMP", "WEBP", "TIFF")


class ImageProblem(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


EXIF_ORIENTATION = 0x0112


def _orientation(img) -> int:
    """The EXIF orientation (1 to 8) from what the header already holds, never decoding; 1 when there is none or
    the block cannot be read."""
    try:
        if img.format == "TIFF":
            value = img.tag_v2.get(EXIF_ORIENTATION)
        else:
            raw = img.info.get("exif")  # JPEG, PNG (before the pixels) and WebP keep it here after the open
            if not raw:
                return 1
            exif = Image.Exif()
            exif.load(raw)
            value = exif.get(EXIF_ORIENTATION)
        return value if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 8 else 1
    except Exception:  # a malformed block means no orientation is known, not an unreadable image
        return 1


def image_header(path, max_pixels: int = MAX_PIXELS_DEFAULT) -> tuple[int, int, int]:
    """``(width, height, EXIF orientation)`` from the file header; the pixels are never decoded.

    The width and height are those of the stored pixels; the orientation (1 when there is none) says how a viewer
    turns them. The cap is checked explicitly on the header's ``width * height`` (the command line silences
    Pillow's warnings, so it cannot rely on them); ``Image.MAX_IMAGE_PIXELS`` is set to the same value meanwhile
    and a ``DecompressionBombError`` or ``DecompressionBombWarning`` counts as too large as well.
    """
    path = pathlib.Path(path)
    name = clip(path.name, 120)
    try:
        info = os.stat(path)
    except OSError:
        raise ImageProblem("E_IMAGE_UNREADABLE", f"{name}: the file cannot be read.") from None
    if not stat.S_ISREG(info.st_mode):
        raise ImageProblem("E_IMAGE_UNREADABLE", f"{name}: not a regular file.")
    if info.st_size == 0:
        raise ImageProblem("E_IMAGE_UNREADABLE", f"{name}: the file is empty (zero size).")
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=IMAGE_FORMATS) as img:
                width, height = img.size  # header only: never .load()
                orientation = _orientation(img)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageProblem("E_IMAGE_TOO_LARGE",
                           f"{name}: the image declares more than {max_pixels} pixels, the cap.") from None
    except Exception:  # not an image, truncated header, unsupported format
        raise ImageProblem("E_IMAGE_UNREADABLE", f"{name}: not a readable image (header missing or truncated).") from None
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    if width <= 0 or height <= 0:
        raise ImageProblem("E_IMAGE_UNREADABLE", f"{name}: the image has zero size.")
    if width * height > max_pixels:
        raise ImageProblem("E_IMAGE_TOO_LARGE",
                           f"{name}: the image declares {width}x{height} pixels, more than the cap of {max_pixels}.")
    return width, height, orientation


def image_size(path, max_pixels: int = MAX_PIXELS_DEFAULT) -> tuple[int, int]:
    """``(width, height)`` of the stored pixels from the file header (see ``image_header``)."""
    return image_header(path, max_pixels)[:2]


@dataclass
class WalkResult:
    files: list[pathlib.Path]
    hidden: int = 0
    outside: list[pathlib.PurePosixPath] = field(default_factory=list)
    followed: list[pathlib.PurePosixPath] = field(default_factory=list)  # image links out of the root that were read
    unreadable: list[pathlib.PurePosixPath] = field(default_factory=list)  # folders refused by permission


def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    return path == root or root in path.parents


def walk(root, follow_outside: bool = False, follow_images: bool = False) -> WalkResult:
    """Every regular file under ``root``, sorted by POSIX relative path (paths start at ``root.resolve()``).

    Names starting with ``.`` are skipped and counted. A symbolic link that resolves outside the resolved root
    is recorded in ``outside`` (its path relative to the root) and skipped, unless ``follow_outside``. With
    ``follow_images`` (what ``--follow-links`` uses) a link to an image file (an image suffix) is listed as a file
    instead and recorded in ``followed``; a link to a folder, or to any other file, is never followed. A link to
    a folder inside the root is not entered again;
    links to files inside the root are listed as files. A dangling link is recorded in ``outside`` too when its
    target would lie outside the root. Directories, FIFOs, sockets and broken links are not returned. A folder that cannot be read (permission denied)
    is recorded in ``unreadable`` (its path relative to the root) and its files are not listed.
    """
    base = pathlib.Path(root).resolve()
    files: list[tuple[str, pathlib.Path]] = []
    outside: list[pathlib.PurePosixPath] = []
    followed: list[pathlib.PurePosixPath] = []
    unreadable: list[pathlib.PurePosixPath] = []
    hidden = 0
    stack = [(base, pathlib.PurePosixPath())]
    seen = {base}
    while stack:
        folder, rel_folder = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except PermissionError:
            if rel_folder.parts:  # the root itself is checked by its callers
                unreadable.append(rel_folder)
            continue
        except OSError:
            continue
        for entry in entries:
            rel = rel_folder / entry.name
            if entry.name.startswith("."):
                hidden += 1
                continue
            full = folder / entry.name
            try:
                is_link = entry.is_symlink()
                if is_link:
                    try:
                        target = full.resolve(strict=True)
                    except FileNotFoundError:
                        # a dangling link: outside the root when its target would be (the target may not exist)
                        if not follow_outside and not _under(full.resolve(strict=False), base):
                            outside.append(rel)
                        continue
                    except (OSError, RuntimeError):  # a link loop: OSError on 3.13+, RuntimeError before
                        outside.append(rel)
                        continue
                    if not _under(target, base) and not follow_outside:
                        if follow_images and entry.is_file() and \
                                pathlib.PurePosixPath(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                            followed.append(rel)  # read only: listed like any file, never written to
                        else:
                            outside.append(rel)
                            continue
                    elif entry.is_dir():
                        if _under(target, base):
                            continue  # its contents are walked under their own names
                        if target in seen:
                            continue
                        seen.add(target)
                        stack.append((target, rel))
                        continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append((full, rel))
                elif entry.is_file():  # follows a link; False for FIFOs, sockets, broken links
                    files.append((rel.as_posix(), full))
            except OSError:
                continue
    files.sort(key=lambda item: item[0])
    outside.sort(key=lambda p: p.as_posix())
    followed.sort(key=lambda p: p.as_posix())
    unreadable.sort(key=lambda p: p.as_posix())
    return WalkResult([f for _, f in files], hidden, outside, followed, unreadable)


def text_cap_mib() -> int:
    """The current cap on a text file, in MiB."""
    return MAX_TEXT_BYTES // (1024 * 1024)


def text_cap_reason() -> str:
    """The sentence every reader gives for a text file over the cap; it names the option that raises the cap."""
    return (f"the file is larger than {text_cap_mib()} MiB, which is the most Labelfence reads "
            "from one text file. Raise the cap with --max-file-mb N if the file is genuine.")


def read_text_bounded(path) -> str:
    """The text of a file: UTF-8, BOM stripped, line endings normalised to LF.

    More than ``MAX_TEXT_BYTES`` raises ``UnsupportedInput``; invalid UTF-8 raises ``UnicodeDecodeError`` and a
    file that cannot be read raises ``OSError`` (the caller decides what that means).
    """
    path = pathlib.Path(path)
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        too_big = size > MAX_TEXT_BYTES
        # at most one byte past the size seen (the file may grow meanwhile), never past the cap + 1; a read of
        # the full cap would allocate its whole buffer for every small label
        data = b"" if too_big else handle.read(min(size, MAX_TEXT_BYTES) + 1)
    if too_big or len(data) > MAX_TEXT_BYTES:
        raise UnsupportedInput(f"{clip(path.name, 120)}: {text_cap_reason()}")
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")
