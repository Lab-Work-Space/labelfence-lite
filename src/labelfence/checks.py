"""Dataset-level checks on the in-memory model: duplicates, tiny boxes, class balance, splits, extensions."""
from __future__ import annotations

import pathlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .images import IMAGE_SUFFIXES
from .model import Box, Dataset, Finding, ImageRecord, finding

PIXEL_TOLERANCE = 0.5  # a box edge may leave the image by this many pixels before it is an error
VALIDATION_SPLITS = ("val", "valid")


def is_validation_split(name: str) -> bool:
    """Whether a split name stands for the validation split: ``val``, ``valid``, any name that starts with
    ``val`` (``validation``, ``val2017``) or ``dev``, whatever the case."""
    lowered = name.lower()
    return lowered.startswith("val") or lowered == "dev"
IOU_SLACK = 1e-9  # so that an IoU of exactly the limit counts whatever the floating-point noise
MAX_PAIRWISE_BOXES = 5000  # distinct boxes in one label file up to which near-duplicate boxes are looked for
_SCALAR_BELOW = 48  # boxes of one class up to which the pairs are compared in plain Python


@dataclass(frozen=True)
class CheckOptions:
    min_box_px: float = 2.0
    imbalance_ratio: float = 50.0
    near_duplicate_iou: float = 0.95


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0.0 when they do not overlap or the union is empty."""
    width = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = ((a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter)
    return inter / union if union > 0 else 0.0


def _sort_key(f: Finding):
    return f.split, f.path, -1 if f.line is None else f.line, f.code, f.message


def sort_findings(findings) -> list[Finding]:
    """The findings sorted by (split, path, line, code, message): the order of every report."""
    return sorted(findings, key=_sort_key)


def _where(line: int | None) -> str:
    return f"line {line}: " if line is not None else ""


def _earlier(line: int | None) -> str:
    return f"the box on line {line}" if line is not None else "an earlier box"


def _check_record(split: str, rec: ImageRecord, options: CheckOptions, out: list[Finding]) -> None:
    path = (rec.label_path or rec.path).as_posix()
    boxes = rec.boxes
    sized = bool(rec.width and rec.height)  # without a size the boxes stay normalised: no pixel checks
    for box in boxes:
        if sized and (box.x_min < -PIXEL_TOLERANCE or box.y_min < -PIXEL_TOLERANCE
                      or box.x_max > rec.width + PIXEL_TOLERANCE or box.y_max > rec.height + PIXEL_TOLERANCE):
            out.append(finding("E_BOX_OUT_OF_RANGE", split, path,
                               f"{_where(box.line)}the box leaves the {rec.width}x{rec.height} image by more "
                               f"than {PIXEL_TOLERANCE} px.", box.line))
        small = min(box.x_max - box.x_min, box.y_max - box.y_min)
        if sized and small < options.min_box_px:
            out.append(finding("W_BOX_TINY", split, path,
                               f"{_where(box.line)}the box is smaller than {options.min_box_px:g} px "
                               f"(its smaller side is {small:.2f} px).", box.line))
    first_seen: dict[tuple, Box] = {}
    distinct: list[Box] = []
    for box in boxes:
        key = (box.class_id, box.x_min, box.y_min, box.x_max, box.y_max)
        if key in first_seen:
            out.append(finding("E_DUPLICATE_BOX", split, path,
                               f"{_where(box.line)}the box is identical in class and coordinates to "
                               f"{_earlier(first_seen[key].line)}.", box.line))
        else:
            first_seen[key] = box
            distinct.append(box)
    # near duplicates: same class, IoU at least the limit. A box is reported once, against its first earlier
    # near-duplicate (so N near-identical boxes give N - 1 findings).
    if len(distinct) > MAX_PAIRWISE_BOXES:
        out.append(finding("W_BOX_NEAR_DUPLICATE", split, path,
                           f"the check for near-duplicate boxes was skipped for this file: it has {len(distinct)} "
                           f"distinct boxes, more than {MAX_PAIRWISE_BOXES}."))
        return
    first_earlier = near_duplicate_pairs(distinct, options.near_duplicate_iou)
    for late in sorted(first_earlier):
        box, early = distinct[late], distinct[first_earlier[late]]
        out.append(finding("W_BOX_NEAR_DUPLICATE", split, path,
                           f"{_where(box.line)}the box overlaps {_earlier(early.line)} "
                           f"(same class, IoU at least {options.near_duplicate_iou:g}).", box.line))


def near_duplicate_pairs(boxes, limit: float) -> dict[int, int]:
    """``{later index: index of its first earlier near-duplicate}`` among ``boxes``: same class, IoU at least
    ``limit``. Only boxes of one class are compared, in order of ``x_min``, and a box is compared only with the
    boxes whose ``x_min`` lies within ``(1 - limit)`` of its own width to the right: an IoU of ``limit`` needs
    an overlap of at least ``limit`` times the width, so no pair outside that window can qualify. Large groups
    compare a whole window at once with numpy; every candidate is confirmed by ``iou``."""
    need = limit - IOU_SLACK
    by_class: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        by_class.setdefault(box.class_id, []).append(index)
    found: dict[int, int] = {}
    for indices in by_class.values():
        if len(indices) > 1:
            _pairs_of_one_class(boxes, indices, need, found)
    return found


def _record(found, first: int, second: int) -> None:
    early, late = (first, second) if first < second else (second, first)
    if late not in found or early < found[late]:
        found[late] = early


def _reach(box: Box, need: float) -> float:
    """The largest ``x_min`` of a box that can still be a near-duplicate of ``box`` when ``box.x_min`` is not
    larger (with a margin for rounding)."""
    if need <= 0:
        return float("inf")
    margin = 1e-7 * max(1.0, abs(box.x_min), abs(box.x_max))
    return box.x_min + max(0.0, 1.0 - need) * (box.x_max - box.x_min) + margin


def _pairs_of_one_class(boxes, indices, need, found) -> None:
    order = sorted(indices, key=lambda i: (boxes[i].x_min, boxes[i].y_min, boxes[i].x_max, boxes[i].y_max, i))
    if len(order) < _SCALAR_BELOW:
        for position, first in enumerate(order):
            a = boxes[first]
            reach = _reach(a, need)
            for second in order[position + 1:]:
                b = boxes[second]
                if b.x_min > reach:
                    break
                if iou(a, b) >= need:
                    _record(found, first, second)
        return
    x0 = np.array([boxes[i].x_min for i in order], dtype=np.float64)
    y0 = np.array([boxes[i].y_min for i in order], dtype=np.float64)
    x1 = np.array([boxes[i].x_max for i in order], dtype=np.float64)
    y1 = np.array([boxes[i].y_max for i in order], dtype=np.float64)
    area = (x1 - x0) * (y1 - y0)
    reach = np.array([_reach(boxes[i], need) for i in order], dtype=np.float64)
    ends = np.searchsorted(x0, reach, side="right")
    for position in range(len(order) - 1):
        stop = int(ends[position])
        if stop <= position + 1:
            continue
        window = slice(position + 1, stop)
        width = np.minimum(x1[window], x1[position]) - np.maximum(x0[window], x0[position])
        height = np.minimum(y1[window], y1[position]) - np.maximum(y0[window], y0[position])
        inter = width * height
        union = area[position] + area[window] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((width > 0) & (height > 0) & (union > 0), inter / union, 0.0)
        for offset in np.nonzero(ratio >= need - 1e-6)[0].tolist():
            second = position + 1 + offset
            if iou(boxes[order[position]], boxes[order[second]]) >= need:
                _record(found, order[position], order[second])


def run_checks(dataset: Dataset, options: CheckOptions = CheckOptions()) -> list[Finding]:
    """The reader's findings plus the dataset-level ones, sorted by (split, path, line, code).

    The dataset's own finding list is not changed. Messages hold counts, line numbers and class numbers, never
    text taken from a file.
    """
    out: list[Finding] = list(dataset.findings)
    known_unusual = {(f.split, f.path) for f in out if f.code == "W_EXTENSION_UNUSUAL"}
    counts: Counter = Counter()
    for split in dataset.splits:
        for rec in split.images:
            if pathlib.PurePosixPath(rec.path.name).suffix.lower() not in IMAGE_SUFFIXES \
                    and (split.name, rec.path.as_posix()) not in known_unusual:
                out.append(finding("W_EXTENSION_UNUSUAL", split.name, rec.path.as_posix(),
                                   "the extension is not one of the image extensions Labelfence reads."))
            if rec.orientation > 1:
                out.append(finding("W_IMAGE_EXIF_ORIENTATION", split.name, rec.path.as_posix(),
                                   f"the image has the EXIF orientation {rec.orientation}: viewers turn it, while "
                                   "its size and its labels are in the stored frame, so tools disagree about which "
                                   "frame the labels use; bake the rotation into the pixels."))
            _check_record(split.name, rec, options, out)
            counts.update(box.class_id for box in rec.boxes)
    for class_id in range(len(dataset.classes)):
        if counts[class_id] == 0:
            out.append(finding("W_CLASS_UNUSED", "", "", f"class {class_id} has no box in any split."))
    used = {c: n for c, n in counts.items() if n > 0}
    if len(used) >= 2:
        top = max(sorted(used), key=lambda c: used[c])
        low = min(sorted(used), key=lambda c: used[c])
        if used[top] > options.imbalance_ratio * used[low]:
            out.append(finding("W_CLASS_IMBALANCE", "", "",
                               f"class {top} has {used[top]} boxes and class {low} has {used[low]}: "
                               f"more than {options.imbalance_ratio:g} times as many."))
    if not any(is_validation_split(split.name) for split in dataset.splits):
        out.append(finding("W_SPLIT_MISSING", "", "",
                           "there is no validation split (a split named val, valid, validation or dev)."))
    out.sort(key=_sort_key)
    return out
