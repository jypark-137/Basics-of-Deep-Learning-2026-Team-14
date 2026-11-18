"""Run numerical and portable-checkpoint checks without datasets or pretrained weights."""

import argparse
import io
import sys
from ..utils.console import CommandParser, command_output, is_verbose, library_output


def get_args_parser():
    parser = CommandParser(description=__doc__)
    parser.add_argument("--math-only", action="store_true")
    return parser


@command_output
def run(args, parser=None):
    from . import check_anchor, check_final_steps, check_ofp
    print("Running checks...", flush=True)
    with library_output():
        check_ofp.check_math()
        check_final_steps.check_reports()
    if not args.math_only:
        with library_output():
            check_anchor.main()
            check_ofp.check_torch()
            check_final_steps.check_torch()
        from . import test_anchor_training, test_configuration, test_console, test_devices, test_species, test_training
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_training)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_configuration))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_devices))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_species))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_anchor_training))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_console))
        stream = sys.stderr if is_verbose() else io.StringIO()
        with library_output():
            result = unittest.TextTestRunner(stream=stream, verbosity=2 if is_verbose() else 1).run(suite)
            if not result.wasSuccessful():
                if not is_verbose():
                    sys.stderr.write(stream.getvalue())
                raise RuntimeError("Project checks failed")
        print("Checks passed ({} tests).".format(result.testsRun))
    else:
        print("Math checks passed.")


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
