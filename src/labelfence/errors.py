"""Exceptions that carry the process exit code."""


class LabelfenceError(Exception):
    exit_code = 1


class UsageError(LabelfenceError):
    exit_code = 2


class FindingsError(LabelfenceError):
    """Raised by the CLI when the audit result's exit code is 3 (errors found, or warnings with --strict)."""
    exit_code = 3


class CheckFailed(LabelfenceError):
    """A conversion round trip failed; nothing was written."""
    exit_code = 3


class UnsupportedInput(LabelfenceError):
    exit_code = 5
