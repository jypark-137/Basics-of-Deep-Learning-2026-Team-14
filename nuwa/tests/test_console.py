"""Command verbosity must preserve errors and stay local to each command."""

from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from types import SimpleNamespace
import unittest

import main
from ..utils.console import command_output, detail, is_verbose, library_output


class ConsoleChecks(unittest.TestCase):
    verbose = False

    def test_verbose_option_works_before_and_after_command(self):
        for arguments in (["--verbose", "prepare", "--list"],
                          ["prepare", "--list", "--verbose"]):
            with self.subTest(arguments=arguments):
                args = main.get_args(arguments)
                self.assertEqual(args.command, "prepare")
                self.assertTrue(args.verbose)

    @command_output
    def test_library_output_preserves_stderr_and_failure_diagnostics(self):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with library_output():
                print("routine library message")
                print("library warning", file=sys.stderr)
            print("command result")
            with self.assertRaisesRegex(ValueError, "dataset failed"):
                with library_output():
                    print("dataset diagnostics")
                    raise ValueError("dataset failed")
        self.assertEqual(output.getvalue(), "command result\n")
        self.assertIn("library warning", errors.getvalue())
        self.assertIn("dataset diagnostics", errors.getvalue())

    @command_output
    def test_verbose_scope_is_restored_after_failure(self):
        @command_output
        def command(args):
            with library_output():
                print("library details")
            detail("command details")
            self.assertTrue(is_verbose())
            raise ValueError("failed command")

        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(ValueError):
            command(SimpleNamespace(verbose=True))
        self.assertFalse(is_verbose())
        self.assertIn("library details", output.getvalue())
        self.assertIn("command details", output.getvalue())


if __name__ == "__main__":
    unittest.main()
