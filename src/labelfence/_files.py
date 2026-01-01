"""Temporary files that are created exclusively and published without replacing a file that appeared meanwhile."""
from __future__ import annotations

import contextlib
import os
import pathlib
import secrets
import shutil
import unicodedata

from .errors import CheckFailed, UsageError
from .text import escape_control

TEMP_PREFIX = ".labelfence-tmp-"
TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)


def open_temp(folder, suffix: str):
    """Create ``<folder>/.labelfence-tmp-<random><suffix>`` exclusively (never through a link), mode 0o600.

    Returns ``(path, binary file object)``. The name is random and not derived from the process id.
    """
    folder = pathlib.Path(folder)
    while True:
        path = folder / f"{TEMP_PREFIX}{secrets.token_hex(6)}{suffix}"
        try:
            fd = os.open(path, TEMP_FLAGS, 0o600)
        except FileExistsError:
            continue
        return path, os.fdopen(fd, "wb")


def default_dir_mode() -> int:
    """The mode a normal ``mkdir`` would give: 0o777 without the bits of the user's umask."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o777 & ~mask


def make_temp_dir(folder) -> pathlib.Path:
    """Create ``<folder>/.labelfence-tmp-<random>`` exclusively with mode 0o700 and return it.

    The folder is private while it is built; ``publish_dir`` gives it the normal mode just before it takes
    its final name.
    """
    folder = pathlib.Path(folder)
    while True:
        path = folder / f"{TEMP_PREFIX}{secrets.token_hex(6)}"
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            continue
        return path


def fold_key(path) -> str:
    """The path with its parent folder resolved and case folded, for comparing names that may not exist yet.

    Two paths with the same key are treated as the same file on every platform (a case-insensitive file
    system, the macOS default, makes ``Out.json`` and ``out.json`` one file, and a file system that ignores
    Unicode normalisation makes the composed and decomposed spellings of ``Caf\u00e9`` one file). ``ss`` and
    the German sharp s fold to one key as well: two such names are refused as one, and no data is lost.
    """
    path = pathlib.Path(os.path.abspath(path))
    try:
        folder = path.parent.resolve()
    except (OSError, RuntimeError):  # a symbolic link loop raises RuntimeError before Python 3.13
        raise UsageError(f"{escape_control(path)}: the folder of this path cannot be resolved "
                         "(a symbolic link loop?).") from None
    return unicodedata.normalize("NFC", str(folder / path.name)).casefold()


@contextlib.contextmanager
def writing_to(path):
    """Inside the block an ``OSError`` becomes a ``UsageError`` (exit 2) that names ``path``, the output that was
    asked for, and the reason the system gives. Errors of Labelfence itself pass through unchanged."""
    try:
        yield
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise UsageError(f"cannot write {escape_control(path)}: {reason}") from None


def default_file_mode() -> int:
    """The mode a normal create would give: 0o666 without the bits of the user's umask."""
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def check_unchanged(path: pathlib.Path, handle) -> None:
    """Raise ``CheckFailed`` unless ``path`` is still the file this process created (``handle``)."""
    handle.flush()
    try:
        same = os.path.samestat(os.fstat(handle.fileno()), os.lstat(path))
    except OSError:
        same = False
    if not same:
        raise CheckFailed(f"{path.name}: the temporary file was replaced while it was being written. "
                          "Nothing was written.")


def finish_temp(path: pathlib.Path, handle) -> None:
    """Check that ``path`` is still the file this process created and set its mode through the handle.

    The mode becomes what a normal create would give (the user's umask). The handle stays open: the caller
    closes it after publishing, because the published entry is compared with it once more.
    """
    check_unchanged(path, handle)
    if hasattr(os, "fchmod"):
        os.fchmod(handle.fileno(), default_file_mode())


def _verify_published(final: pathlib.Path, handle, linked) -> None:
    """The entry now at ``final`` must be the file behind ``handle``.

    ``linked`` is the ``lstat`` of the entry this call just put at ``final``. If ``final`` is not the handle's
    file, it is removed only when it is that entry; anything else at the name was put there by someone else and
    is left alone.
    """
    if handle is None:
        return
    try:
        same = os.path.samestat(os.fstat(handle.fileno()), os.lstat(final))
    except OSError:
        same = False
    if same:
        return
    try:
        ours = os.path.samestat(linked, os.lstat(final))
    except OSError:
        ours = False
    if ours:
        try:
            os.unlink(final)
        except OSError:
            pass
        raise CheckFailed(f"{final.name}: the temporary file was replaced while it was being written. "
                          "Nothing was written.")
    raise CheckFailed(f"{final.name}: the output name was taken by another file while it was being written. "
                      "That file was left in place.")


def publish_file(tmp: pathlib.Path, final: pathlib.Path, overwrite: bool, handle=None) -> None:
    """Move ``tmp`` to ``final``; without ``overwrite`` never replace a file that appeared meanwhile.

    The link never follows a symbolic link at ``tmp``. When the still-open ``handle`` is given, ``tmp`` is
    compared with it just before publishing, and the published entry afterwards. With ``overwrite`` an existing
    output is replaced only after that first check, so a swapped temporary name never costs the old output.
    """
    if handle is not None:
        check_unchanged(tmp, handle)
    if overwrite:
        before = os.lstat(tmp)
        os.replace(tmp, final)
        _verify_published(final, handle, before)
        return
    try:
        os.link(tmp, final, follow_symlinks=False)
    except FileExistsError:
        raise UsageError(f"{final.name}: the output already exists. Pass --overwrite to replace it.") from None
    except (OSError, NotImplementedError):
        if os.path.lexists(final):
            raise UsageError(f"{final.name}: the output already exists. Pass --overwrite to replace it.") from None
        before = os.lstat(tmp)
        os.replace(tmp, final)
        _verify_published(final, handle, before)
        return
    try:
        _verify_published(final, handle, os.lstat(tmp))
    finally:
        if os.path.lexists(tmp):
            tmp.unlink()


def publish_dir(tmp_dir, final_dir) -> None:
    """Rename the temporary folder ``tmp_dir`` to ``final_dir``, only when nothing exists at ``final_dir``.

    A folder is never replaced (there is no ``overwrite`` for folders): an existing entry of any kind at the
    final name, a link or an empty folder included, is a ``UsageError`` and ``tmp_dir`` is left for the caller
    to remove. The name is claimed with ``mkdir``, which fails when anything is there, so an entry that appears
    between a check and the rename is refused as well; ``tmp_dir`` then takes the place of that empty claim. The
    folder gets the mode a normal ``mkdir`` would give (it was private while it was built).
    """
    tmp_dir, final_dir = pathlib.Path(tmp_dir), pathlib.Path(final_dir)
    exists = (f"{escape_control(final_dir.name)}: the output already exists. "
              "Choose a name that does not exist; folders are never replaced.")
    if os.path.lexists(final_dir):
        raise UsageError(exists)
    try:
        os.mkdir(final_dir)
    except FileExistsError:
        raise UsageError(exists) from None
    claim = os.lstat(final_dir)
    try:
        os.chmod(tmp_dir, default_dir_mode())
        os.rename(tmp_dir, final_dir)  # replaces the empty claim, which is ours
    except BaseException:
        try:
            if os.path.samestat(claim, os.lstat(final_dir)):
                os.rmdir(final_dir)
        except OSError:
            pass
        raise


def remove_tree(tmp_dir) -> None:
    """Remove the temporary folder and everything in it; a link is removed itself, never its target.

    Used on failure and on interruption. A folder that is already gone is not an error.
    """
    tmp_dir = pathlib.Path(tmp_dir)
    try:
        if tmp_dir.is_symlink():
            tmp_dir.unlink()
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError:
        pass
