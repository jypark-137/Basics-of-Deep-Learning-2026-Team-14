"""Run numerical and portable-checkpoint checks without datasets or pretrained weights."""

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-only", action="store_true")
    return parser


def run(args, parser=None):
    from . import check_anchor, check_final_steps, check_ofp
    check_ofp.check_math()
    check_final_steps.check_reports()
    if not args.math_only:
        check_anchor.main()
        check_ofp.check_torch()
        check_final_steps.check_torch()
        from . import test_configuration, test_devices, test_training
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_training)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_configuration))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_devices))
        if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
            raise RuntimeError("Fine-tuning checks failed")
    print("NuWa checks: PASS")


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
