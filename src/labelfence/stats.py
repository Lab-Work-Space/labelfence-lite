"""Dataset statistics (design spec 5.4), computed from the in-memory model so every format shares them."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .model import Dataset, ImageRecord

QUANTILE_POINTS = [0, 25, 50, 75, 100]
SIZES_LISTED = 20
DECIMALS = 6


def _quantiles(values) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]  # a value that overflowed to inf or nan would make the report unserialisable
    if array.size == 0:
        return None
    result = np.percentile(array, QUANTILE_POINTS, method="linear")
    return [round(float(v), DECIMALS) for v in result]


def _split_stats(records: list[ImageRecord], classes: list[str]) -> dict:
    labelled = sum(1 for r in records if r.boxes)
    counts: Counter = Counter()
    sizes: Counter = Counter()
    unknown = 0
    width_px, height_px, area_px, aspect = [], [], [], []
    width_n, height_n, area_n = [], [], []
    for rec in records:
        if rec.width and rec.height:
            sizes[(rec.width, rec.height)] += 1
        else:
            unknown += 1
        for box in rec.boxes:
            counts[box.class_id] += 1
            w, h = box.x_max - box.x_min, box.y_max - box.y_min
            if rec.width and rec.height:
                width_px.append(w)
                height_px.append(h)
                area_px.append(w * h)
                aspect.append(w / h)
                w, h = w / rec.width, h / rec.height
            width_n.append(w)  # a box without an image size is already normalised
            height_n.append(h)
            area_n.append(w * h)
    total_boxes = sum(counts.values())
    ids = sorted(set(range(len(classes))) | set(counts))
    listed = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:SIZES_LISTED]
    return {
        "images": len(records),
        "labelled_images": labelled,
        "background_images": len(records) - labelled,
        "boxes": total_boxes,
        "classes": [{"class_id": i, "name": classes[i] if i < len(classes) else f"class_{i}", "count": counts[i],
                     "share": round(counts[i] / total_boxes, DECIMALS) if total_boxes else 0.0} for i in ids],
        "classes_present": sorted(counts),
        "box_width_px": _quantiles(width_px),
        "box_height_px": _quantiles(height_px),
        "box_area_px": _quantiles(area_px),
        "aspect_ratio": _quantiles(aspect),
        "box_width_norm": _quantiles(width_n),
        "box_height_norm": _quantiles(height_n),
        "box_area_norm": _quantiles(area_n),
        "image_sizes": {"distinct": len(sizes), "unknown": unknown,
                        "listed": [{"width": w, "height": h, "count": n} for (w, h), n in listed]},
    }


def compute_stats(dataset: Dataset) -> dict:
    """Per split and total: image and box counts, per-class counts and shares, box size and aspect-ratio
    quantiles (at 0, 25, 50, 75 and 100 percent, linear interpolation) in pixels and normalised, and the image
    size histogram (at most 20 sizes listed). A quantile list is ``None`` when there is nothing to measure.
    Shares are the share of all boxes in the split (0.0 when it has none)."""
    classes = list(dataset.classes)
    everything = [rec for split in dataset.splits for rec in split.images]
    return {
        "quantile_points": list(QUANTILE_POINTS),
        "total": _split_stats(everything, classes),
        "splits": [{"name": split.name, **_split_stats(split.images, classes)} for split in dataset.splits],
    }
