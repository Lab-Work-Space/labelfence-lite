# Labelfence (lite)

Labelfence reports what it finds in your files; you decide what to change. It makes no claim about training results.

Labelfence is a command-line tool that audits object-detection datasets before you train on them. The lite edition reads a dataset in YOLO format and checks every image and every label file against the rules of the format. It prints what it found, writes a JSON report, and exits with a code a script can test. It runs offline and never changes your dataset.

## What it does

- Reads YOLO datasets in three layouts: `images/<split>/` with `labels/<split>/` (the Ultralytics layout), `<split>/images/` with `<split>/labels/`, and a flat `images/` with `labels/`. Every sub-folder of `images/` is a split, whatever its name, and files directly in `images/` form one unnamed split. A `data.yaml` with `names` and `nc` is read when it exists, and its `train`, `val` and `test` keys name the split of a folder. A `classes.txt` at the root or in a labels folder gives the class names when there is no `data.yaml`. A folder named `valid` is accepted as the validation split, and `train2017` is read as `train`.
- Checks every label line (field count, numbers, class id, box inside the image, box with a size, repeated boxes) and every image (readable, size within a cap, extension, name, label present).
- Reports 20 kinds of findings, each with a stable code: `E_IMAGE_UNREADABLE`, `E_IMAGE_TOO_LARGE`, `E_LABEL_MISSING_IMAGE`, `E_LABEL_MALFORMED`, `E_CLASS_ID_INVALID`, `E_BOX_OUT_OF_RANGE`, `E_BOX_DEGENERATE`, `E_DUPLICATE_BOX`, `E_NAMES_MISMATCH`, `W_IMAGE_WITHOUT_LABEL`, `W_LABEL_EMPTY`, `W_BOX_TINY`, `W_BOX_NEAR_DUPLICATE`, `W_CLASS_UNUSED`, `W_CLASS_IMBALANCE`, `W_SPLIT_MISSING`, `W_IMAGE_NAME_COLLISION`, `W_EXTENSION_UNUSUAL`, `W_PATH_OUTSIDE_ROOT` and `W_IMAGE_EXIF_ORIENTATION`. Codes starting with `E_` are errors; codes starting with `W_` are warnings.
- Prints counts per split and per class (`labelfence stats`); the JSON report also holds box-size, aspect-ratio and image-size statistics.
- Writes a JSON report (`--report FILE.json`, or `--json -` to print it).
- Writes a small synthetic dataset to try it on: `labelfence sample --out DIR --case clean` has no finding, `--case faulty` has one of every code above.

## What it does not do

- It reads the headers of images and does not decode them. A JPEG that is cut off after its header is not found by the lite edition.
- It reads YOLO datasets only. COCO and Pascal VOC datasets, conversion between formats, near-duplicate images, splits and the HTML report are part of the full edition.
- Labels are boxes only. A polygon label is a finding unless you pass `--allow-polygons`, which takes the bounding box of the polygon.
- It never modifies your dataset, never writes inside it, and never replaces an existing file unless you pass `--overwrite`.
- It leaves no temporary file behind when a command fails or is interrupted (a process that is killed can leave a `.labelfence-tmp-*` file; delete it).

## Install

Requires Python 3.10 or newer. Check with `python3 --version` first. On an older Python, `pip` stops with an error such as `Package 'labelfence' requires a different Python: 3.9.6 not in '>=3.10'`, or, with an older `pip`, `No matching distribution found for numpy<2.6,>=2.2`. From the unzipped folder that contains this README:

```text
python3 --version
python3 -m venv .venv
. .venv/bin/activate
pip install .
labelfence --version
```

On Windows the two commands after the version check differ: `py -m venv .venv`, then `.venv\Scripts\activate`. Windows is not tested.

`pip` may leave a folder `build/` next to this README and a folder ending in `.egg-info` under `src/`. They are harmless; you can delete them after the install.

To remove Labelfence later: `pip uninstall -y labelfence`. The libraries it installed stay until you delete the virtual environment (the folder `.venv`).

## Quick start

```text
labelfence sample --out faulty --case faulty
labelfence audit faulty --report faulty-audit.json
```

The second command prints the summary and the findings grouped by code, writes `faulty-audit.json`, and exits with code 3 because the dataset has errors. Run `labelfence audit` on your own dataset in the same way: `labelfence audit path/to/dataset`. Add `--strict` to exit 3 on warnings as well, and `--quiet` to print the summary only.

Where the report goes: without `--report FILE` the report is written to `<dataset folder name>-audit.json` in the current folder, for example `clean-audit.json` for a folder `clean`. If that file already exists, or the current folder is inside the dataset, the audit still runs and prints its summary, then one line: `Report not written: the default report clean-audit.json cannot be used here (it already exists). Pass --report FILE (and --overwrite to replace).` The exit code is the audit's own, so a second run of `labelfence audit clean` on a clean dataset exits 0. Pass `--report FILE --overwrite` to replace a report. An explicit `--report FILE` that already exists is a usage error (exit code 2) before the audit runs.

## Exit codes

| Code | Meaning |
| :--- | :--- |
| 0 | No finding, or warnings only |
| 1 | Unexpected error (the message names the error type only; add `--debug` for a traceback) |
| 2 | Usage error, a command that belongs to the full edition, or an output that cannot be written (`cannot write <path>: <reason>`) |
| 3 | The audit found errors (or warnings with `--strict`) |
| 5 | Unsupported or invalid input (not a dataset, or a layout that cannot be read) |
| 130 | Interrupted (Ctrl-C or SIGTERM); temporary files are removed |

## Limits

- Boxes only; YOLO only in the lite edition.
- An image is read for its header; the size cap is 50,000,000 pixels (`--max-pixels`), and a label, JSON or XML file over 64 MiB is refused (`--max-file-mb`).
- A line of a label file or of `data.yaml` is read up to 4096 characters, and a label file with more than 100,000 non-blank lines is not read; each is a finding. The check for boxes of one class that overlap almost completely (`W_BOX_NEAR_DUPLICATE`) is skipped for a file with more than 5000 distinct boxes, and one finding says so.
- A folder where the images and the label files lie side by side (as LabelImg writes them) is not read; put the images in `images/` and the label files in `labels/`.
- An output that cannot be written (for example, a report in a folder without write permission) is reported as `cannot write <path>` with exit code 2.
- Tested on macOS with Python 3.10 and 3.14. Linux and Windows are not tested.

## Full edition

The full edition adds COCO and Pascal VOC, conversion between the three formats with a round-trip check, near-duplicate or look-alike images and leakage between splits, split lists, an HTML report, a catalogue of synthetic datasets and pytest fixtures: https://labworkspace.gumroad.com/l/labelfence

`--follow-links` (before or after the command) reads images that are links leaving the dataset folder, read only; without it such a link is skipped and reported as `W_PATH_OUTSIDE_ROOT`.

In the lite edition the commands `dupes`, `split`, `convert` and `html-report` appear in `labelfence --help`. Each one prints a message that it is part of the full edition, and exits with code 2.

## Licence

MIT, see `LICENSE`. Copyright (c) 2026 Labworkspace.
