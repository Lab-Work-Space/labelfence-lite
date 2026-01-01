"""What every writer does before it writes: check the images to copy, and copy them."""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

from .errors import UnsupportedInput
from .text import clip


_COPY_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


class ImageCopier:
    """Copies image files byte for byte (exclusive create, never re-encoded); hard-links with ``link``.

    When a hard link is not possible (another file system, a system without links) the file is copied and one
    line on standard error says so, once.
    """

    def __init__(self, link: bool = False):
        self.link = link
        self.noted = False

    def put(self, source: pathlib.Path, target: pathlib.Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise FileExistsError(f"{target.name}: exists")
        if self.link:
            try:
                os.link(source, target)
                return
            except FileExistsError:
                raise
            except (OSError, NotImplementedError):
                if not self.noted:
                    self.noted = True
                    print("labelfence: note: hard links are not possible here; the images are copied instead.",
                          file=sys.stderr)
        # the copy is created exclusively and never through a link: whatever appeared at the name meanwhile is kept
        try:
            reader = open(source, "rb")
        except OSError as exc:  # a read problem, not a write problem
            raise UnsupportedInput(f"{clip(source.name, 120)}: the image cannot be read "
                                   f"({exc.strerror or type(exc).__name__}), so it cannot be copied.") from None
        with reader:
            fd = os.open(target, _COPY_FLAGS, 0o666)
            with os.fdopen(fd, "wb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)


def plan_images(dataset, *, by_stem: bool, need_size: bool = False, annotation: str = "annotation file",
                rename=None):
    """``[(split name, record, file name, stem, source path)]`` for every image, checked before anything is written.

    Refuses (``UnsupportedInput``, naming the image by its relative path) an image outside the dataset root, an
    image that is missing or not a regular file, an image whose size is unknown (``need_size``), a box whose
    class is not in the class list, and two images of one split that would share one annotation file
    (``by_stem``) or one name. ``rename`` maps an image's relative path to the file name to write it under
    (the source is still read from its own path).
    """
    root = pathlib.Path(dataset.root).resolve()
    followed = set(getattr(dataset, "followed", ()))  # image links out of the root that were read, read only
    plan = []
    for split in dataset.splits:
        name = split.name
        if name and (not name.isprintable() or name.startswith(".") or "/" in name or "\\" in name):
            raise UnsupportedInput(f"the split name {clip(name, 40)!r} cannot be written as a folder name.")
        taken: dict[str, str] = {}
        for rec in split.images:
            name = rename.get(rec.path, rec.path.name) if rename else rec.path.name
            stem = pathlib.PurePosixPath(name).stem
            shown = clip(rec.path.as_posix(), 120)
            key = (stem if by_stem else name).casefold()
            if key in taken:
                raise UnsupportedInput(
                    f"{shown}: the images {clip(taken[key], 60)} and {clip(name, 60)} in split "
                    f"{clip(split.name or '(unnamed)', 20)} would share one {annotation if by_stem else 'name'}. "
                    "Rename one of them.")
            taken[key] = name
            source = (root / rec.path).resolve()
            if root not in source.parents and rec.path not in followed:
                raise UnsupportedInput(f"{shown}: the image is outside the dataset root.")
            if not source.is_file():
                raise UnsupportedInput(f"{shown}: the image file is missing (or is not a regular file), so it "
                                       "cannot be copied.")
            if need_size and not (rec.width and rec.height):
                raise UnsupportedInput(f"{shown}: the image size is unknown (unreadable image), so it cannot "
                                       "be written with its size.")
            for box in rec.boxes:
                if not 0 <= box.class_id < len(dataset.classes):
                    raise UnsupportedInput(f"{shown}: a box has class {box.class_id}, which is not in the "
                                           f"class list of {len(dataset.classes)} classes.")
            plan.append((split.name, rec, name, stem, source))
    return plan
