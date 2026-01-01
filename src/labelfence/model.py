"""The one in-memory model every format is read into and written out of, and the closed list of finding codes."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


MAX_COORDINATE = 1e9  # a coordinate, width, height or image size above this (or not finite) is refused by every reader


@dataclass(frozen=True, slots=True)
class Box:
    """One box in absolute pixels (floats). When the image size is unknown the values are normalised (0 to 1)."""
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    line: int | None = field(default=None, compare=False)  # label line the box came from; not part of equality


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # "error" or "warning"
    split: str
    path: str  # relative to the dataset root, POSIX separators
    line: int | None
    message: str


@dataclass
class ImageRecord:
    path: pathlib.PurePosixPath  # relative to the dataset root, POSIX separators
    width: int | None  # None when the image is unreadable
    height: int | None
    boxes: list[Box]
    label_path: pathlib.PurePosixPath | None
    # the label file exists but could not be read at all (not the same as an empty label: a background image)
    label_unreadable: bool = field(default=False, compare=False)
    # the EXIF orientation of the image (1 to 8; 1 when there is none): 2 to 8 mean that viewers turn the picture
    # while the header size and the labels are in the stored frame
    orientation: int = field(default=1, compare=False)


@dataclass
class Split:
    name: str  # "train", "val", "test" or "" (unnamed)
    images: list[ImageRecord]


@dataclass
class Dataset:
    root: pathlib.Path
    fmt: str  # "yolo" | "coco" | "voc"
    layout: str  # free text naming the layout detected
    classes: list[str]
    splits: list[Split]
    findings: list[Finding] = field(default_factory=list)  # reader-level findings (malformed files etc.)
    skipped_hidden: int = 0
    skipped_outside_root: list[pathlib.PurePosixPath] = field(default_factory=list)
    # image links that leave the root and were read, read only (``--follow-links``)
    followed: list[pathlib.PurePosixPath] = field(default_factory=list)
    # labelled images left out because their extension is not one Labelfence reads (relative paths)
    left_out_by_extension: list[pathlib.PurePosixPath] = field(default_factory=list)


# code -> (severity, one-line meaning). Exactly the codes of the design spec, section 5.3; the list is closed.
CODES: dict[str, tuple[str, str]] = {
    "E_IMAGE_UNREADABLE": ("error", "the image is not an image, is truncated, or has zero size"),
    "E_IMAGE_TOO_LARGE": ("error", "the image header declares more pixels than the cap"),
    "E_LABEL_MISSING_IMAGE": ("error", "a label file has no image"),
    "E_LABEL_MALFORMED": ("error", "a label line has the wrong field count or a non-numeric field"),
    "E_CLASS_ID_INVALID": ("error", "a class id is not an integer, is negative, or is at least the class count"),
    "E_BOX_OUT_OF_RANGE": ("error", "a box value is outside the image or outside 0 to 1"),
    "E_BOX_DEGENERATE": ("error", "a box has no positive width or height"),
    "E_DUPLICATE_BOX": ("error", "a file holds two boxes with identical class and coordinates"),
    "E_NAMES_MISMATCH": ("error", "the category names disagree with the class list"),
    "E_SIZE_MISMATCH": ("error", "the image size declared by the format differs from the file"),
    "E_DUPLICATE_IMAGE_ID": ("error", "two images share one id"),
    "E_ANNOTATION_ORPHAN": ("error", "an annotation refers to an image id that does not exist"),
    "W_IMAGE_WITHOUT_LABEL": ("warning", "an image has no label file (a background image)"),
    "W_LABEL_EMPTY": ("warning", "a label file has no boxes"),
    "W_BOX_TINY": ("warning", "a box is smaller than the minimum size"),
    "W_BOX_NEAR_DUPLICATE": ("warning", "two boxes of one class overlap almost completely"),
    "W_CLASS_UNUSED": ("warning", "a class has no box"),
    "W_CLASS_IMBALANCE": ("warning", "the most frequent class has far more boxes than the least frequent used class"),
    "W_SPLIT_MISSING": ("warning", "there is no validation split"),
    "W_IMAGE_NAME_COLLISION": ("warning", "two images share a base name"),
    "W_EXTENSION_UNUSUAL": ("warning", "an image has an extension outside the common set"),
    "W_LEAKAGE_NEAR_DUPLICATE": ("warning", "a group of near-duplicate or look-alike images spans two splits"),
    "W_IMAGE_DUPLICATE_EXACT": ("warning", "two image files are byte-identical"),
    "W_PATH_OUTSIDE_ROOT": ("warning", "a symbolic link resolves outside the dataset root; it was skipped"),
    "E_IMAGE_IN_TWO_SPLITS": ("error", "the same image file is listed in two splits"),
    "W_IMAGE_EXIF_ORIENTATION": ("warning", "the image has an EXIF orientation, so tools may disagree about the "
                                            "frame its labels use"),
}


def finding(code: str, split: str, path, message: str, line: int | None = None) -> Finding:
    """Build a finding; a code outside ``CODES`` raises ``KeyError`` (a reader never invents a code)."""
    severity, _ = CODES[code]
    return Finding(code, severity, split, str(path), line, message)
