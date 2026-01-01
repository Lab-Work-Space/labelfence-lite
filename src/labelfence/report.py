"""The JSON report, its exit code, a safe writer and the console summary."""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from importlib import import_module

from . import NOTICE, __version__
from ._files import finish_temp, fold_key, open_temp, publish_file, writing_to
from .errors import UsageError
from .model import CODES, Dataset, Finding
from .text import NAME_LIMIT, clip, escape_control

MESSAGE_LIMIT = 300
CLASS_LIMIT = 40


def tool_versions() -> dict:
    """Versions of Labelfence (and the full edition when installed) and of the libraries it reads with."""
    import numpy
    import PIL

    versions = {"labelfence": __version__}
    try:
        versions["labelfence-pro"] = import_module("labelfence_pro").__version__
    except Exception:  # a full package that is missing or broken must not stop the lite edition
        pass
    versions["numpy"] = numpy.__version__
    versions["pillow"] = PIL.__version__
    return versions


def exit_code_for(findings, strict: bool = False) -> int:
    """0, or 3 when any finding is an error (or any is a warning with ``strict``)."""
    for f in findings:
        if f.severity == "error" or strict:
            return 3
    return 0


def _finding_dict(f: Finding) -> dict:
    return {"code": f.code, "severity": f.severity, "split": f.split, "path": f.path, "line": f.line,
            "message": f.message}


def build_report(dataset: Dataset, findings, stats: dict, *, tool_versions: dict, options) -> dict:
    """The JSON-ready report (design spec 6.2). ``findings`` ``None`` makes a statistics-only report (no
    findings, summary or exit code). ``options`` is a dict or a dataclass; its ``strict`` entry, when present,
    decides the exit code. Paths are the relative POSIX paths of the model; the root is as the user gave it."""
    opts = dataclasses.asdict(options) if dataclasses.is_dataclass(options) else dict(options)
    report = {
        "tool_versions": dict(tool_versions),
        "dataset_root": str(dataset.root),
        "format": dataset.fmt,
        "layout": dataset.layout,
        "classes": list(dataset.classes),
        "options": opts,
        "skipped_hidden": dataset.skipped_hidden,
        "statistics": stats,
    }
    if findings is not None:
        by_code: dict[str, int] = {}
        for f in findings:
            by_code[f.code] = by_code.get(f.code, 0) + 1
        report["summary"] = {"errors": sum(f.severity == "error" for f in findings),
                             "warnings": sum(f.severity == "warning" for f in findings),
                             "by_code": dict(sorted(by_code.items()))}
        report["findings"] = [_finding_dict(f) for f in findings]
        report["near_duplicate_groups"] = []  # filled by the full edition
        report["exit_code"] = exit_code_for(findings, bool(opts.get("strict")))
    report["notice"] = NOTICE
    return report


def check_output_path(path, root, inputs=()) -> pathlib.Path:
    """The output path, or ``UsageError`` when it is inside the dataset root or is an input file."""
    path = pathlib.Path(path)
    if not path.name:
        raise UsageError(f"{clip(path, NAME_LIMIT)}: not a file name. Expected a path such as report.json.")
    key = fold_key(path)
    root = pathlib.Path(root)
    try:
        root = root.resolve()
    except (OSError, RuntimeError):
        pass
    protected = [root, *map(pathlib.Path, inputs)] if root.is_file() else [*map(pathlib.Path, inputs)]
    for item in protected:
        if fold_key(item) == key:
            raise UsageError(f"{clip(path.name, NAME_LIMIT)}: this is an input file and is never overwritten. "
                             "Choose another output path.")
    root_key = fold_key(root)
    if root.is_dir() and key.startswith(root_key.rstrip(os.sep) + os.sep):
        raise UsageError(f"{clip(path.name, NAME_LIMIT)}: the output would be inside the dataset folder, "
                         "where Labelfence never writes. Choose a path outside it.")
    if path.is_dir():
        raise UsageError(f"{clip(path.name, NAME_LIMIT)}: this is a folder. Expected a file name for the output.")
    if not path.parent.is_dir():
        raise UsageError(f"{clip(path.parent, NAME_LIMIT)}: the folder of the output does not exist.")
    return path


def write_text_output(text: str, path, root, *, overwrite: bool = False, inputs=(), suffix: str = ".json") -> None:
    """Write ``text`` to ``path`` through an exclusive temporary file in the same folder, then publish it."""
    final = check_output_path(path, root, inputs)
    data = text.encode("utf-8")
    with writing_to(path):
        tmp, handle = open_temp(final.parent, suffix)
        try:
            handle.write(data)
            finish_temp(tmp, handle)
            publish_file(tmp, final, overwrite, handle)
        finally:
            handle.close()
            if os.path.lexists(tmp):
                try:
                    tmp.unlink()
                except OSError:
                    pass


def dump_json(report: dict) -> str:
    """The report as JSON text: ASCII only, so a file name with control or odd characters is escaped."""
    return json.dumps(report, indent=2, allow_nan=False) + "\n"


def write_report(report: dict, path, overwrite: bool = False, *, inputs=()) -> None:
    """Write the report as JSON. Refuses a path inside the dataset root or equal to an input file, and an
    existing file without ``overwrite``."""
    text = dump_json(report)  # first: a report that cannot be serialised leaves nothing behind
    write_text_output(text, path, report.get("dataset_root", ""), overwrite=overwrite, inputs=inputs)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _percent(share) -> str:
    return f"{100 * share:.1f}%"


def print_summary(report: dict, *, max_per_code: int = 20, quiet: bool = False, stream) -> None:
    """The console summary (design spec 6.1). Every name is escaped and clipped; no file content is shown."""
    out = lambda text="": print(text, file=stream)  # noqa: E731
    stats = report["statistics"]
    total = stats["total"]
    out(f"Labelfence {clip(report['tool_versions'].get('labelfence', ''), 20)}: {clip(report['dataset_root'], 200)}")
    out(f"Format: {clip(report['format'], 20)} (layout: {clip(report['layout'], 40)})")
    out(f"Images: {total['images']}  Boxes: {total['boxes']}  Classes: {len(report['classes'])}")
    followed = report.get("options", {}).get("followed_links")
    if isinstance(followed, int) and followed:
        out(f"Followed {followed} link(s) to images outside the dataset folder (read only).")
    for split in stats["splits"]:
        out(f"  {clip(split['name'] or '(unnamed)', 40)}: {plural(split['images'], 'image')}, "
            f"{split['boxes']} {'box' if split['boxes'] == 1 else 'boxes'}, "
            f"{split['background_images']} without boxes")
    if "findings" not in report:
        for item in total["classes"]:
            out(f"  class {item['class_id']} {clip(item['name'], CLASS_LIMIT)}: {item['count']} boxes "
                f"({_percent(item['share'])})")
        return
    summary = report["summary"]
    out(f"Findings: {plural(summary['errors'], 'error')}, {plural(summary['warnings'], 'warning')}")
    ordered = sorted(summary["by_code"], key=lambda code: (code not in CODES or CODES[code][0] != "error", code))
    for code in ordered:
        out(f"  {code}: {summary['by_code'][code]}")
    if not quiet:
        for code in ordered:
            items = [f for f in report["findings"] if f["code"] == code]
            out()
            meaning = CODES[code][1] if code in CODES else ""
            out(f"{code} ({len(items)}): {meaning}")
            for f in items[:max_per_code]:
                where = clip(f["path"] or "(dataset)", NAME_LIMIT) + (f":{f['line']}" if f["line"] is not None else "")
                out(f"  {where}  {clip(f['message'], MESSAGE_LIMIT)}")
            if len(items) > max_per_code:
                out(f"  ... and {len(items) - max_per_code} more")
    out()
    out(escape_control(report["notice"]))
