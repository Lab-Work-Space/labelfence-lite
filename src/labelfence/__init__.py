"""Labelfence: audit object-detection datasets."""

__version__ = "1.0.0"

NOTICE = ("Labelfence reports what it finds in your files; you decide what to change. "
          "It makes no claim about training results.")

# Reader registry: format name -> ``reader(root, *, allow_polygons, max_pixels) -> Dataset``. The full edition
# adds "coco" and "voc". The CLI looks the reader up here when a command runs.
READERS: dict = {}

# Writer registry: format name -> ``writer(dataset, out_dir, *, link) -> None``. The full edition adds "coco" and
# "voc"; its conversion command looks the writers up here.
WRITERS: dict = {}

# Audit extensions: objects the full edition appends here so that ``audit`` can do more without the lite
# package importing it. Each has ``add_arguments(parser)`` (extra options of ``audit``) and
# ``run(dataset, args) -> (findings, groups, options)``: findings to add, entries for the report's
# ``near_duplicate_groups`` and a dict of options to record in the report. Empty in the lite edition.
AUDIT_EXTENSIONS: list = []

from .yolo import read_yolo, write_yolo  # noqa: E402  (after the names above, so the modules may import them)

READERS["yolo"] = read_yolo
WRITERS["yolo"] = write_yolo
