"""Command-line entry point and plugin loading."""

import argparse
import json
import logging
import math
import os
import pathlib
import re
import signal
import sys
import traceback
import warnings
from importlib import import_module
from importlib.metadata import entry_points

import labelfence
from labelfence import NOTICE, __version__, checks, images, report as reports, sample as samples, stats as statistics
from labelfence._files import fold_key, make_temp_dir, publish_dir, remove_tree, writing_to
from labelfence.errors import FindingsError, LabelfenceError, UnsupportedInput, UsageError
from labelfence.images import MAX_PIXELS_DEFAULT, read_text_bounded
from labelfence.text import clip, escape_control, full_edition_only
from labelfence.yolo import SPLIT_DIRS

PLUGIN_GROUP = "labelfence.commands"
FULL_ONLY_COMMANDS = ("dupes", "split", "convert", "html-report")


def _plugin_entry_points():
    return list(entry_points(group=PLUGIN_GROUP))


def _version_text() -> str:
    lines = [f"labelfence {__version__}", NOTICE]
    try:
        lines.append(f"labelfence-pro {import_module('labelfence_pro').__version__}")
    except Exception:  # a full package that is missing or broken must not stop the lite edition
        pass
    return "\n".join(lines)


class _VersionAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(_version_text())
        parser.exit(0)


def _full_edition_only(name):
    def handler(args):
        raise UsageError(full_edition_only(f"'{name}'"))
    handler.full_edition_only = True
    return handler


FORMATS = ("yolo", "coco", "voc")
FORMAT_NAMES = {"yolo": "YOLO", "coco": "COCO", "voc": "Pascal VOC"}


def _non_negative_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("expected a number, 0 or more")
    return value


def _positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a number above 0")
    return value


MAX_INTEGER_OPTION = 10 ** 15  # no option needs more; a larger number is a usage error, never a surprise later


def _count(text):
    value = int(text)
    if not 0 <= value <= MAX_INTEGER_OPTION:
        raise argparse.ArgumentTypeError(f"expected a whole number from 0 to {MAX_INTEGER_OPTION}")
    return value


def _positive_int(text):
    value = int(text)
    if not 1 <= value <= MAX_INTEGER_OPTION:
        raise argparse.ArgumentTypeError(f"expected a whole number from 1 to {MAX_INTEGER_OPTION}")
    return value


def _dataset_arguments(parser) -> None:
    parser.add_argument("dataset", metavar="DATASET", help="the root folder of the dataset")
    parser.add_argument("--format", choices=("auto", *FORMATS), default="auto",
                        help="dataset format (default: detect from the files)")
    parser.add_argument("--report", metavar="R.json", help="write the JSON report here")
    parser.add_argument("--json", choices=("-",), metavar="-", help="print the JSON report to standard output")
    parser.add_argument("--allow-polygons", action="store_true", help="take the bounding box of polygon labels")
    parser.add_argument("--max-pixels", type=_positive_int, default=MAX_PIXELS_DEFAULT, metavar="N",
                        help=f"largest image read, in pixels (default {MAX_PIXELS_DEFAULT})")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")


def _add_commands(subparsers) -> None:
    audit = subparsers.add_parser("audit", help="check a dataset and report what is found", allow_abbrev=False)
    _dataset_arguments(audit)
    audit.add_argument("--html", metavar="R.html", help="write an HTML report (full edition)")
    audit.add_argument("--strict", action="store_true", help="exit 3 on warnings as well as errors")
    audit.add_argument("--min-box-px", type=_non_negative_float, default=checks.CheckOptions.min_box_px, metavar="N",
                       help="a box with a side below this many pixels is tiny (default 2)")
    audit.add_argument("--imbalance-ratio", type=_positive_float, default=checks.CheckOptions.imbalance_ratio,
                       metavar="R", help="class imbalance warning above this ratio (default 50)")
    audit.add_argument("--max-per-code", type=_count, default=20, metavar="N",
                       help="findings shown per code on the console (default 20)")
    audit.add_argument("--quiet", action="store_true", help="print the summary only")
    audit.set_defaults(func=_audit)  # the full edition adds its options after the plugins load, see build_parser
    stats = subparsers.add_parser("stats", help="print dataset statistics, without checks", allow_abbrev=False)
    _dataset_arguments(stats)
    stats.set_defaults(func=_stats)
    sample = subparsers.add_parser("sample", help="write a small synthetic dataset with known findings",
                                   allow_abbrev=False)
    sample.add_argument("--out", metavar="DIR", help="the folder to write (must not exist or must be empty)")
    sample.add_argument("--case", default="clean", metavar="NAME", help="the sample case (default: clean)")
    sample.add_argument("--seed", type=_count, default=0, metavar="N", help="seed of the generator (default 0)")
    sample.add_argument("--list", action="store_true", help="list the sample cases and exit")
    sample.set_defaults(func=_sample)


def _looks_like_coco(path: pathlib.Path) -> bool:
    try:
        data = json.loads(read_text_bounded(path))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError, MemoryError, UnsupportedInput):
        return False
    return isinstance(data, dict) and "images" in data and "annotations" in data


def _detect_formats(root: pathlib.Path) -> list[str]:
    found = []
    if root.is_file():
        return ["coco"] if root.suffix.lower() == ".json" and _looks_like_coco(root) else []
    # YOLO: a data.yaml or a labels folder (a COCO or VOC dataset may have an images folder of its own)
    labels = [root / "labels", *(root / name / "labels" for name in SPLIT_DIRS)]
    if (root / "data.yaml").is_file() or any(path.is_dir() for path in labels):
        found.append("yolo")
    jsons = []
    for folder in (root, root / "annotations"):
        try:
            jsons += sorted(p for p in folder.iterdir() if p.suffix.lower() == ".json" and not p.name.startswith(".")
                            and p.is_file())
        except OSError:
            pass
    for name in SPLIT_DIRS:  # the Roboflow export: <split>/_annotations.coco.json
        candidate = root / name / "_annotations.coco.json"
        if candidate.is_file():
            jsons.append(candidate)
    if any(_looks_like_coco(p) for p in jsons):
        found.append("coco")
    # VOC: an xml file in Annotations, or in a sub-folder of it (one folder per split)
    if any(name.lower().endswith(".xml") for _, _, names in os.walk(root / "Annotations") for name in names):
        found.append("voc")
    return found


def _choose_format(args, root: pathlib.Path) -> str:
    if args.format != "auto":
        chosen = args.format
    else:
        if not root.exists():
            raise UnsupportedInput(f"{clip(root, 200)}: not found. Expected the root folder of a dataset.")
        found = _detect_formats(root)
        if not found:
            hint = (" It has an images/ folder but no labels/ folder."
                    if (root / "images").is_dir() and not (root / "labels").is_dir() else "")
            raise UnsupportedInput(f"{clip(root, 200)}: no dataset found.{hint} Expected a YOLO layout (data.yaml, "
                                   "images/ and labels/), a COCO json or a VOC Annotations folder.")
        if len(found) > 1:
            raise UsageError(f"{clip(root, 200)}: this folder looks like more than one format "
                             f"({', '.join(found)}). Pass --format to choose.")
        chosen = found[0]
        print(f"labelfence: detected format {chosen}", file=sys.stderr)
    if chosen not in labelfence.READERS:
        raise UsageError(full_edition_only(f"Reading {FORMAT_NAMES[chosen]} datasets"))
    return chosen


def _output_paths(args, root: pathlib.Path):
    """The report path (or ``None``), checked before the dataset is read.

    An explicit ``--report`` that cannot be used (it exists, it lies in the dataset) is a usage error. The DEFAULT
    path of ``audit`` that cannot be used (it exists without ``--overwrite``, or the current folder is inside the
    dataset) is not: the audit runs, and ``args.report_note`` holds the line that says why no report was written.
    """
    path = args.report
    default = path is None and args.command == "audit" and args.json is None
    if default:
        # made of safe characters only: the name of a folder may hold anything
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", root.resolve().name or "dataset")[:80]
        path = pathlib.Path.cwd() / f"{stem}-audit.json"
    if path is None:
        return None
    try:
        path = reports.check_output_path(path, root)
        if os.path.lexists(path) and not args.overwrite:
            raise UsageError(f"{clip(path.name, 120)}: the output already exists. Pass --overwrite to replace it.")
    except UsageError as exc:
        if not default:
            raise
        name = clip(path.name, 120)
        why = ("it already exists" if "already exists" in str(exc)
               else "the current folder is inside the dataset folder" if "inside the dataset" in str(exc)
               else clip(exc, 200))
        args.report_note = f"Report not written: the default report {name} cannot be used here ({why}). " \
                           "Pass --report FILE (and --overwrite to replace)."
        return None
    return path


def _html_renderer():
    try:
        return import_module("labelfence_pro.html_report").render_html
    except Exception:  # a missing or broken full package: no HTML renderer
        return None


def _read_dataset(args, root, fmt):
    reader = labelfence.READERS[fmt]
    return reader(root, allow_polygons=args.allow_polygons, max_pixels=args.max_pixels,
                  follow_links=getattr(args, "follow_links", False))


def link_options(args, dataset) -> dict:
    """The options entries of the report that ``--follow-links`` adds (none when it is off)."""
    if not getattr(args, "follow_links", False):
        return {}
    return {"follow_links": True, "followed_links": len(dataset.followed)}


def _announce(path) -> None:
    print(f"Report written: {clip(path, 200)}")


def _audit(args) -> None:
    root = pathlib.Path(args.dataset)
    renderer = None
    if args.html:
        renderer = _html_renderer()
        if renderer is None:
            raise UsageError(full_edition_only("'--html'"))
    fmt = _choose_format(args, root)
    out_path = _output_paths(args, root)
    html_path = reports.check_output_path(args.html, root) if args.html else None
    if html_path is not None and out_path is not None and fold_key(html_path) == fold_key(out_path):
        raise UsageError(f"{clip(html_path.name, 120)}: --report and --html name the same file. Choose two paths.")
    options = checks.CheckOptions(args.min_box_px, args.imbalance_ratio)
    dataset = _read_dataset(args, root, fmt)
    findings = checks.run_checks(dataset, options)
    groups: list = []
    extra_options: dict = {}
    for extension in labelfence.AUDIT_EXTENSIONS:  # the full edition's near-duplicate scan, when installed
        more, found_groups, found_options = extension.run(dataset, args)
        findings = findings + list(more)
        groups += found_groups
        extra_options.update(found_options)
    findings = checks.sort_findings(findings)
    report = reports.build_report(
        dataset, findings, statistics.compute_stats(dataset), tool_versions=reports.tool_versions(),
        options={**vars(options), "strict": args.strict, "allow_polygons": args.allow_polygons,
                 "max_pixels": args.max_pixels, **link_options(args, dataset), **extra_options})
    report["near_duplicate_groups"] = groups
    if args.json == "-":
        print(reports.dump_json(report), end="")
    else:
        reports.print_summary(report, max_per_code=args.max_per_code, quiet=args.quiet, stream=sys.stdout)
    if getattr(args, "report_note", None):
        print(args.report_note)
    if out_path is not None:
        reports.write_report(report, out_path, args.overwrite)
        if args.json != "-":
            _announce(out_path)
    if html_path is not None:
        reports.write_text_output(renderer(report), html_path, root, overwrite=args.overwrite, suffix=".html")
        if args.json != "-":
            _announce(html_path)
    if report["exit_code"] == 3:
        errors, warnings = report["summary"]["errors"], report["summary"]["warnings"]
        parts = [reports.plural(errors, "error")] if errors else []
        if warnings and (args.strict or not errors):
            parts.append(reports.plural(warnings, "warning"))
        raise FindingsError(f"the audit found {' and '.join(parts)}"
                            + (" (--strict counts warnings)" if args.strict and not errors else "") + ".")


def _stats(args) -> None:
    root = pathlib.Path(args.dataset)
    fmt = _choose_format(args, root)
    out_path = _output_paths(args, root)
    dataset = _read_dataset(args, root, fmt)
    report = reports.build_report(dataset, None, statistics.compute_stats(dataset),
                                  tool_versions=reports.tool_versions(),
                                  options={"allow_polygons": args.allow_polygons, "max_pixels": args.max_pixels,
                                           **link_options(args, dataset)})
    if args.json == "-":
        print(reports.dump_json(report), end="")
    else:
        reports.print_summary(report, stream=sys.stdout)
    if out_path is not None:
        reports.write_report(report, out_path, args.overwrite)
        if args.json != "-":
            _announce(out_path)


def _sample(args) -> None:
    cases = samples.CASES
    if args.list:
        for name in sorted(cases):
            print(f"{name}  {clip(cases[name].description, 200)}")
        return
    if args.out is None:
        raise UsageError("--out DIR is needed (or --list to see the cases).")
    if args.case not in cases:
        if args.case in samples.FULL_CASES:
            raise UsageError(full_edition_only(f"The sample case '{args.case}'"))
        raise UsageError(f"Unknown sample case '{clip(args.case, 60)}'. Available: {', '.join(sorted(cases))}.")
    out = pathlib.Path(args.out)
    here = pathlib.Path.cwd().resolve()
    if not args.out or out.resolve() == here:
        raise UsageError("--out must name a folder to create or an empty folder, not the current folder. "
                         "Give a new name, for example --out sample-data.")
    samples.check_out_dir(out)
    out = out.resolve()
    if not out.parent.is_dir():
        raise UsageError(f"{clip(out.parent, 200)}: the folder to write into does not exist.")
    # built next to the final name and renamed at the end, so that a failure or an interruption leaves nothing
    temp = None
    try:
        with writing_to(out):
            temp = make_temp_dir(out.parent)
            built = samples.build_case(cases[args.case], temp, seed=args.seed)
            if out.is_dir():
                out.rmdir()  # an empty folder; publish_dir never replaces one
            publish_dir(temp, out)
    finally:
        if temp is not None:
            remove_tree(temp)
    print(f"Sample '{args.case}' written: {clip(out, 200)}")
    for note in built.notes:
        print(clip(note, 400))
    print(f"Next: labelfence audit {clip(out, 200)}")


def _load_plugins(subparsers, debug: bool) -> None:
    for entry_point in _plugin_entry_points():
        try:
            entry_point.load()(subparsers)
        except (Exception, SystemExit) as exc:  # a plugin may even try to end the process
            print(f"labelfence: warning: plugin '{escape_control(entry_point.name)}' failed to load: "
                  f"{type(exc).__name__}: {escape_control(exc)}", file=sys.stderr)
            if debug:
                traceback.print_exc()


ERROR_LIMIT = 400  # characters of an argument-parsing message


class _Parser(argparse.ArgumentParser):
    """An argument parser whose error text is escaped and clipped: it may quote what the user typed, and a shell
    glob can expand to file names with control characters. Sub-parsers use this class too."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {clip(message, ERROR_LIMIT)}\n")


def build_parser(debug: bool = False) -> argparse.ArgumentParser:
    parser = _Parser(prog="labelfence", description=NOTICE, allow_abbrev=False)
    parser.add_argument("--version", action=_VersionAction, help="show the version and exit")
    parser.add_argument("--debug", action="store_true", help="show a traceback on unexpected errors")
    parser.add_argument("--follow-links", action="store_true", default=False,
                        help="read images that are links leaving the dataset folder (read only, never written to); "
                             "also accepted after the command")
    parser.add_argument("--max-file-mb", type=_positive_int, default=None, metavar="N",
                        help=f"largest label, JSON or XML file read, in MiB (default {images.MAX_TEXT_MB_DEFAULT}); "
                             "also accepted after the command")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_commands(subparsers)
    _load_plugins(subparsers, debug)
    for extension in labelfence.AUDIT_EXTENSIONS:
        extension.add_arguments(subparsers.choices["audit"])
    for sub in subparsers.choices.values():
        if not any(action.dest == "rest" for action in sub._actions):  # not a stub that passes everything on
            sub.add_argument("--max-file-mb", type=_positive_int, default=argparse.SUPPRESS, metavar="N",
                             help="largest label, JSON or XML file read, in MiB (see the global option)")
            sub.add_argument("--follow-links", action="store_true", default=argparse.SUPPRESS,
                             help="read images that are links leaving the dataset folder (see the global option)")
    for name in FULL_ONLY_COMMANDS:
        if name not in subparsers.choices:
            stub = subparsers.add_parser(name, help="(full edition only)", add_help=False)
            stub.add_argument("rest", nargs=argparse.REMAINDER)
            stub.set_defaults(func=_full_edition_only(name))
    return parser


QUIET_LOGGERS = ("PIL", "numpy")
_SILENT = logging.CRITICAL + 1


class _QuietLoggers:
    """Raise the level of the libraries' loggers while a command runs, then put it back exactly.

    A log line can quote a value from the user's files; without a handler Python prints it on stderr.
    """

    def __enter__(self):
        self.saved = []
        for name, logger in list(logging.root.manager.loggerDict.items()):
            if isinstance(logger, logging.Logger) and any(name == p or name.startswith(p + ".") for p in QUIET_LOGGERS):
                self.saved.append((logger, logger.level))
        for name in QUIET_LOGGERS:  # also loggers that do not exist yet but are created by an import
            logger = logging.getLogger(name)
            if all(logger is not seen for seen, _ in self.saved):
                self.saved.append((logger, logger.level))
        for logger, _ in self.saved:
            logger.setLevel(_SILENT)
        return self

    def __exit__(self, *exc):
        for logger, level in self.saved:
            logger.setLevel(level)
        return False


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


class _TerminateAsInterrupt:
    """While a command runs, SIGTERM ends it like Ctrl-C does, so that temporary files are removed.

    The previous handler is put back afterwards. Outside the main thread signals cannot be set; nothing changes.
    """

    def __enter__(self):
        self.previous = None
        try:
            self.previous = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
        except (ValueError, OSError, AttributeError):
            self.previous = None
        return self

    def __exit__(self, *exc):
        if self.previous is not None:
            try:
                signal.signal(signal.SIGTERM, self.previous)
            except (ValueError, OSError):
                pass
        return False


def main(argv: list[str] | None = None) -> int:
    with _TerminateAsInterrupt():
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    debug = "--debug" in (sys.argv[1:] if argv is None else argv)
    with warnings.catch_warnings():
        if debug:
            return _run(argv, debug)
        # A library warning can quote a value from the user's files; keep it off the console.
        # Blanket on purpose (not module-scoped): any library warning may quote a value from the user's files.
        # Pillow's DecompressionBombWarning is silent too, so the pixel cap must be an own header check.
        # The filter lives only while the command runs, so users of the Python API are not affected.
        warnings.simplefilter("ignore")
        with _QuietLoggers():
            return _run(argv, debug)


def _run(argv, debug: bool) -> int:
    try:
        parser = build_parser(debug)
        try:
            # The full-only stubs take any options, so their own message is shown, not an argparse one.
            args, extra = parser.parse_known_args(argv)
            if extra and not getattr(getattr(args, "func", None), "full_edition_only", False):
                hint = ""
                if any(item in ("--debug", "--version") for item in extra):
                    hint = (". --debug and --version go before the command, for example: "
                            "labelfence --debug audit DATASET")
                parser.error("unrecognized arguments: " + " ".join(extra) + hint)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 2
        if getattr(args, "func", None) is None:
            parser.print_help()
            return 2
        cap = getattr(args, "max_file_mb", None)
        saved = images.MAX_TEXT_BYTES
        if cap is not None:
            images.MAX_TEXT_BYTES = cap * 1024 * 1024  # for this command only
        try:
            args.func(args)
        finally:
            images.MAX_TEXT_BYTES = saved
        return 0
    except LabelfenceError as exc:
        print(f"labelfence: error: {escape_control(exc)}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("labelfence: interrupted", file=sys.stderr)
        if debug:
            traceback.print_exc()
        return 130
    except Exception as exc:
        # The message of an unexpected exception can hold a value from the user's files: print the type only.
        hint = "" if debug else " Run with --debug for details."
        print(f"labelfence: unexpected error: {type(exc).__name__}.{hint}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
