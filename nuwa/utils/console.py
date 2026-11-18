"""Consistent command output, with optional diagnostic details."""

import argparse
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
from functools import wraps
import io
from pathlib import Path
import sys

from ..paths import PROJECT_ROOT


_verbose = ContextVar("nuwa_verbose", default=False)


class CommandParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "--verbose" not in self._option_string_actions:
            self.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS,
                              help="Show detailed checks and library output")

    def error(self, message):
        self.exit(2, "Error: {}\n".format(message))


def is_verbose():
    return _verbose.get()


def command_output(function):
    @wraps(function)
    def wrapped(args, *positional, **keywords):
        token = _verbose.set(bool(getattr(args, "verbose", is_verbose())))
        try:
            return function(args, *positional, **keywords)
        finally:
            _verbose.reset(token)
    return wrapped


def detail(*args, **kwargs):
    if is_verbose():
        print(*args, **kwargs)


def display_path(value):
    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


@contextmanager
def library_output():
    """Hide routine library chatter; retain stderr and diagnostics on failure."""
    if is_verbose():
        yield
        return
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            yield
    except BaseException:
        sys.stderr.write(captured.getvalue())
        raise
