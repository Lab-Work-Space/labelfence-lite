"""Synthetic datasets with ground truth: what the generator injects is what the audit must report."""
from __future__ import annotations

import os
import pathlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw

from .errors import UsageError
from .text import clip

# The cases of the full edition (it registers them); the lite edition names them in its refusal.
FULL_CASES = ("coco", "voc", "leaky", "imbalanced", "polygons", "flat-layout", "nested-layout", "big-names",
              "coco-faulty", "voc-faulty")

CLASS_COLOURS = ((220, 40, 40), (40, 200, 60), (50, 80, 230), (230, 210, 40), (200, 60, 200), (40, 200, 200))
JPEG_QUALITY = 90
IMBALANCE_BOXES = 60  # boxes in the imbalance file: far more than 50 times the one box of the rarest class


@dataclass(frozen=True)
class CaseSpec:
    name: str
    classes: tuple[str, ...] = ("cat", "dog", "bird")
    images_per_split: dict = field(default_factory=lambda: {"train": 12, "val": 4})
    image_size: tuple[int, int] = (96, 64)  # width, height; some cases use odd sizes
    boxes_per_image: tuple[int, int] = (1, 3)
    layout: str = "ultralytics"
    faults: tuple[str, ...] = ()  # finding codes to inject, each once, in a known file
    description: str = ""
    # an own way to build the case: ``builder(spec, out_dir, seed) -> Sample`` (the full edition builds the
    # COCO and VOC cases by converting the clean one); ``out_dir`` is absent-or-empty and already created
    builder: Optional[Callable] = None


@dataclass(frozen=True)
class Sample:
    root: pathlib.Path
    spec: CaseSpec
    # code -> [(split, relative path)] the audit must report, exactly (the strings a Finding carries)
    expected: dict
    # image path (relative, POSIX) -> [(class id, x0, y0, x1, y1)] the integer pixel rectangles that were drawn
    drawn: dict = field(default_factory=dict)
    # one-line remarks for the person who generated the sample (a skipped fault, a deliberate symbolic link)
    notes: tuple = ()


CASES: dict[str, CaseSpec] = {}


def register_case(spec: CaseSpec) -> None:
    """Add (or replace) a case; the full edition registers its catalogue through this."""
    CASES[spec.name] = spec


LITE_FAULTS = (
    "E_IMAGE_UNREADABLE", "E_IMAGE_TOO_LARGE", "E_LABEL_MISSING_IMAGE", "E_LABEL_MALFORMED", "E_CLASS_ID_INVALID",
    "E_BOX_OUT_OF_RANGE", "E_BOX_DEGENERATE", "E_DUPLICATE_BOX", "E_NAMES_MISMATCH", "W_IMAGE_WITHOUT_LABEL", "W_LABEL_EMPTY",
    "W_BOX_TINY", "W_BOX_NEAR_DUPLICATE", "W_CLASS_UNUSED", "W_CLASS_IMBALANCE", "W_SPLIT_MISSING",
    "W_IMAGE_NAME_COLLISION", "W_EXTENSION_UNUSUAL", "W_PATH_OUTSIDE_ROOT", "W_IMAGE_EXIF_ORIENTATION",
)

register_case(CaseSpec(
    "clean", description="3 classes, train and val, no finding: the audit reports nothing."))
register_case(CaseSpec(
    "faulty", classes=("cat", "dog", "bird", "fish"), images_per_split={"train": 8}, image_size=(97, 63),
    faults=LITE_FAULTS,
    description="one of every error and warning code of the lite edition, each in a known file (odd image size)."))


# ---------------------------------------------------------------- drawing

def _png_declaring(width: int, height: int) -> bytes:
    """A valid PNG whose header declares ``width`` x ``height`` and whose data is one tiny row."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00" * 16))
            + chunk(b"IEND", b""))


def _darker(colour):
    return tuple(max(0, c - 90) for c in colour)


def _render(path: pathlib.Path, size, rects, rng, orientation: int = 1) -> None:
    width, height = size
    noise = rng.integers(40, 120, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(noise)
    draw = ImageDraw.Draw(image)
    for cls, x0, y0, x1, y1 in rects:
        colour = CLASS_COLOURS[cls % len(CLASS_COLOURS)]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=colour, outline=_darker(colour), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        image.save(path, format="PNG")
    elif orientation > 1:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(path, format="JPEG", quality=JPEG_QUALITY, exif=exif)
    else:
        image.save(path, format="JPEG", quality=JPEG_QUALITY)


def _line(cls: int, rect, size) -> str:
    x0, y0, x1, y1 = rect
    width, height = size
    return (f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")


def _place_boxes(rng, size, classes, count):
    """Up to ``count`` non-overlapping rectangles (at least one), each side 12 px or more, inside the image."""
    width, height = size
    rects = []
    tries = 0
    while len(rects) < count and tries < 200:
        tries += 1
        w = int(rng.integers(12, max(13, min(36, width // 2))))
        h = int(rng.integers(12, max(13, min(36, height // 2))))
        x0 = int(rng.integers(1, max(2, width - w - 1)))
        y0 = int(rng.integers(1, max(2, height - h - 1)))
        x1, y1 = x0 + w, y0 + h
        if x1 > width - 1 or y1 > height - 1:
            continue
        if any(x0 < b[3] + 2 and b[1] < x1 + 2 and y0 < b[4] + 2 and b[2] < y1 + 2 for b in rects):
            continue
        rects.append((classes[len(rects) % len(classes)], x0, y0, x1, y1))
    if not rects:
        rects.append((classes[0], 4, 4, 4 + 12, 4 + 12))
    return rects


# ---------------------------------------------------------------- the generator

class _Builder:
    def __init__(self, spec: CaseSpec, root: pathlib.Path, seed: int):
        self.spec, self.root, self.seed = spec, root, seed
        self.size = spec.image_size
        self.expected: dict[str, list] = {}
        self.drawn: dict[str, list] = {}
        self.notes: list[str] = []
        self.counter = 0
        faults = set(spec.faults)
        n = len(spec.classes)
        # classes that ordinary images never use: the last one stays unused, the one before it is the rarest
        reserved = set()
        if "W_CLASS_UNUSED" in faults:
            reserved.add(n - 1)
        self.rare = n - 2 if "W_CLASS_IMBALANCE" in faults else None
        if self.rare is not None:
            reserved.add(self.rare)
        self.ordinary = [c for c in range(n) if c not in reserved] or [0]

    def rng(self):
        self.counter += 1
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.counter]))

    def expect(self, code, split, path):
        self.expected.setdefault(code, []).append((split, path))

    def next_classes(self, count):
        start = self.counter
        return [self.ordinary[(start + i) % len(self.ordinary)] for i in range(count)]

    def write_label(self, split, stem, lines):
        path = self.root / "labels" / split / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="")
        return f"labels/{split}/{stem}.txt"

    def write_image(self, split, stem, ext, rects, rng, orientation=1):
        rel = f"images/{split}/{stem}{ext}"
        _render(self.root / rel, self.size, rects, rng, orientation)
        self.drawn[rel] = list(rects)
        return rel

    def image_with_label(self, split, stem, ext, rects, rng, extra=()):
        """An image with the given rectangles and a label of their lines followed by ``extra`` lines."""
        rel_image = self.write_image(split, stem, ext, rects, rng)
        rel_label = self.write_label(split, stem, [_line(c, (x0, y0, x1, y1), self.size)
                                                   for c, x0, y0, x1, y1 in rects] + list(extra))
        return rel_image, rel_label

    def ordinary_image(self, split, stem, ext):
        rng = self.rng()
        lo, hi = self.spec.boxes_per_image
        count = int(rng.integers(lo, hi + 1))
        rects = _place_boxes(rng, self.size, self.next_classes(count), count)
        self.image_with_label(split, stem, ext, rects, rng)

    def one_box(self, rng):
        return _place_boxes(rng, self.size, self.next_classes(1), 1)

    def build(self):
        spec, size = self.spec, self.size
        if spec.layout != "ultralytics":
            raise UsageError(f"The sample generator of this edition writes the 'ultralytics' layout, "
                             f"not {clip(spec.layout, 40)!r}.")
        if size[0] < 90 or size[1] < 60:
            raise UsageError("The sample images must be at least 90x60 pixels.")
        faults = list(spec.faults)
        unknown = [code for code in faults if code not in LITE_FAULTS]
        if unknown:
            raise UsageError(f"The sample generator cannot inject {clip(unknown[0], 40)}.")
        for split, count in spec.images_per_split.items():
            for i in range(count):
                self.ordinary_image(split, f"img_{i:03d}", ".png" if i % 4 == 3 else ".jpg")
        split = next(iter(spec.images_per_split), None)
        if faults and split is None:
            raise UsageError("A case with faults needs at least one split.")
        for code in faults:
            getattr(self, "fault_" + code)(split)
        self.write_yaml()

    def write_yaml(self):
        names = list(self.spec.classes)
        if "E_NAMES_MISMATCH" in self.expected:
            names[1] = names[0]  # E_NAMES_MISMATCH: one name listed twice (two class ids, one class)
        lines = [f"nc: {len(names)}\n", f"names: [{', '.join(names)}]\n"]
        lines += [f"{s}: images/{s}\n" for s in self.spec.images_per_split]
        (self.root / "data.yaml").write_text("".join(lines), encoding="utf-8", newline="")

    # -- faults: each writes its own file(s) and records what the audit must report

    def fault_E_IMAGE_UNREADABLE(self, split):
        rng = self.rng()
        rel = f"images/{split}/unreadable.jpg"
        (self.root / rel).write_bytes(b"this is not an image\n")
        self.write_label(split, "unreadable", [_line(r[0], r[1:], self.size) for r in self.one_box(rng)])
        self.expect("E_IMAGE_UNREADABLE", split, rel)

    def fault_E_IMAGE_TOO_LARGE(self, split):
        rng = self.rng()
        rel = f"images/{split}/toolarge.png"
        (self.root / rel).write_bytes(_png_declaring(30000, 30000))
        self.write_label(split, "toolarge", [_line(r[0], r[1:], self.size) for r in self.one_box(rng)])
        self.expect("E_IMAGE_TOO_LARGE", split, rel)

    def fault_E_LABEL_MISSING_IMAGE(self, split):
        rng = self.rng()
        label = self.write_label(split, "orphan_label", [_line(r[0], r[1:], self.size) for r in self.one_box(rng)])
        self.expect("E_LABEL_MISSING_IMAGE", split, label)

    def _with_extra(self, code, split, stem, extra):
        rng = self.rng()
        rects = self.one_box(rng)
        _, label = self.image_with_label(split, stem, ".jpg", rects, rng, extra)
        self.expect(code, split, label)

    def fault_E_LABEL_MALFORMED(self, split):
        self._with_extra("E_LABEL_MALFORMED", split, "malformed", ["0 0.5 0.5 0.2"])

    def fault_E_CLASS_ID_INVALID(self, split):
        self._with_extra("E_CLASS_ID_INVALID", split, "badclass",
                         [f"{len(self.spec.classes) + 3} 0.5 0.5 0.2 0.2"])

    def fault_E_NAMES_MISMATCH(self, split):
        self.expect("E_NAMES_MISMATCH", "", "data.yaml")  # written by write_yaml, which lists a name twice

    def fault_E_BOX_OUT_OF_RANGE(self, split):
        self._with_extra("E_BOX_OUT_OF_RANGE", split, "outofrange", ["0 0.5 0.5 1.2 0.2"])

    def fault_E_BOX_DEGENERATE(self, split):
        self._with_extra("E_BOX_DEGENERATE", split, "degenerate", ["0 0.5 0.5 0.0 0.2"])

    def fault_E_DUPLICATE_BOX(self, split):
        rng = self.rng()
        rects = self.one_box(rng)
        extra = [_line(rects[0][0], rects[0][1:], self.size)]
        _, label = self.image_with_label(split, "duplicate", ".jpg", rects, rng, extra)
        self.expect("E_DUPLICATE_BOX", split, label)

    def fault_W_IMAGE_WITHOUT_LABEL(self, split):
        rng = self.rng()
        rel = self.write_image(split, "unlabelled", ".jpg", self.one_box(rng), rng)
        self.expect("W_IMAGE_WITHOUT_LABEL", split, rel)

    def fault_W_LABEL_EMPTY(self, split):
        rng = self.rng()
        self.write_image(split, "emptylabel", ".jpg", self.one_box(rng), rng)
        label = self.write_label(split, "emptylabel", [])
        self.expect("W_LABEL_EMPTY", split, label)

    def fault_W_BOX_TINY(self, split):
        rng = self.rng()
        cls = self.next_classes(1)[0]
        tiny = (cls, 10, 10, 11, 22)  # 1 px wide
        _, label = self.image_with_label(split, "tiny", ".jpg", [tiny], rng)
        self.expect("W_BOX_TINY", split, label)

    def fault_W_BOX_NEAR_DUPLICATE(self, split):
        rng = self.rng()
        cls = self.next_classes(1)[0]
        first, second = (cls, 20, 20, 60, 50), (cls, 20, 20, 61, 50)  # IoU 0.976
        _, label = self.image_with_label(split, "neardup", ".jpg", [first, second], rng)
        self.expect("W_BOX_NEAR_DUPLICATE", split, label)

    def fault_W_CLASS_UNUSED(self, split):
        self.expect("W_CLASS_UNUSED", "", "")  # the last class is kept out of every ordinary image

    def fault_W_CLASS_IMBALANCE(self, split):
        rng = self.rng()
        width, height = self.size
        cols, rows = (width - 1) // 9, (height - 1) // 10
        cells = [(1 + 9 * c, 1 + 10 * r) for r in range(rows) for c in range(cols)][:IMBALANCE_BOXES]
        rects = [(0, x, y, x + 5, y + 5) for x, y in cells]
        rects[-1] = (self.rare, *rects[-1][1:])  # exactly one box of the rarest class
        self.image_with_label(split, "crowded", ".jpg", rects, rng)
        self.expect("W_CLASS_IMBALANCE", "", "")

    def fault_W_SPLIT_MISSING(self, split):
        if "val" in self.spec.images_per_split or "valid" in self.spec.images_per_split:
            raise UsageError("W_SPLIT_MISSING needs a case without a val split.")
        self.expect("W_SPLIT_MISSING", "", "")

    def fault_W_IMAGE_NAME_COLLISION(self, split):
        rng = self.rng()
        rects = self.one_box(rng)
        first, _ = self.image_with_label(split, "collision", ".jpg", rects, rng)
        second = self.write_image(split, "collision", ".png", self.one_box(rng), rng)
        self.expect("W_IMAGE_NAME_COLLISION", split, first)
        self.expect("W_IMAGE_NAME_COLLISION", split, second)

    def fault_W_EXTENSION_UNUSUAL(self, split):
        rel = f"images/{split}/extra.xyz"
        (self.root / rel).write_bytes(b"not an image either\n")
        self.expect("W_EXTENSION_UNUSUAL", split, rel)

    def fault_W_IMAGE_EXIF_ORIENTATION(self, split):
        rng = self.rng()
        rects = self.one_box(rng)
        rel = self.write_image(split, "exifrot", ".jpg", rects, rng, orientation=6)
        self.write_label(split, "exifrot", [_line(c, (x0, y0, x1, y1), self.size) for c, x0, y0, x1, y1 in rects])
        self.expect("W_IMAGE_EXIF_ORIENTATION", split, rel)

    def fault_W_PATH_OUTSIDE_ROOT(self, split):
        rel = f"images/{split}/outside_link"
        # a relative link to a file that does not exist, in the folder that holds the sample root: outside it
        try:
            os.symlink("../../../labelfence-sample-outside-target.txt", self.root / rel)
        except (OSError, NotImplementedError):
            self.notes.append("W_PATH_OUTSIDE_ROOT was skipped: this system did not allow a symbolic link.")
            return
        self.notes.append(f"note: {rel} is a deliberate symbolic link to a file outside this folder (it does not "
                          "exist); it is how the faulty case shows W_PATH_OUTSIDE_ROOT. Do not copy this folder "
                          "with a tool that follows links.")
        self.expect("W_PATH_OUTSIDE_ROOT", split, rel)


def check_out_dir(out: pathlib.Path) -> None:
    """``UsageError`` unless ``out`` is absent or an empty folder (never a link or a file)."""
    if os.path.lexists(out) and (out.is_symlink() or not out.is_dir() or any(out.iterdir())):
        raise UsageError(f"{clip(out, 200)}: the output folder is not empty. Choose a folder that does not "
                         "exist or is empty.")


def build_case(spec: CaseSpec, out_dir, seed: int = 0) -> Sample:
    """Write ``spec`` into ``out_dir`` (absent or empty, else ``UsageError``) and return it with its ground truth.

    The bytes are the same for the same spec and seed on one machine. They are not promised across Pillow
    (JPEG encoder) or numpy versions; the labels, ``expected`` and ``drawn`` do not depend on the encoder.
    """
    out = pathlib.Path(out_dir)
    check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    if spec.builder is not None:
        return spec.builder(spec, out, seed)
    builder = _Builder(spec, out, seed)
    builder.build()
    return Sample(out, spec, builder.expected, builder.drawn, tuple(builder.notes))
