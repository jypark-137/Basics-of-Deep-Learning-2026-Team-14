"""NuWa-shaped ViTPose/AP-10K project entry point.

Run `python main.py --help` from the project root.
Without a command, options are interpreted by the complete derive workflow.
"""


import argparse
import importlib
import os
import sys


COMMANDS = {
    "derive": ("nuwa.method.nuwa", "Learn/reuse an anchor, prune, and compare pose models"),
    "anchor": ("nuwa.method.get_anchor_model", "Learn SKP masks or export saved masks as an anchor"),
    "prune": ("nuwa.method.pruning", "Perform OFP from an anchor or a baseline ablation"),
    "evaluate": ("nuwa.engine.eval", "Run official AP-10K evaluation"),
    "train": ("nuwa.engine.train", "Optionally fine-tune model weights with the pose loss"),
    "profile": ("nuwa.utils.utils", "Measure matrix costs, parameters and forward latency"),
    "prepare": ("nuwa.data.prepare_species", "Prepare the selected AP-10K species subsets"),
    "check-model": ("nuwa.model.check_model", "Check model heatmaps and pose-loss gradients"),
    "compare": ("nuwa.finalize_pose", "Export, benchmark and compare baseline/anchor/OFP models"),
    "ablate": ("nuwa.run_pose_ablation", "Run the controlled pruning ablations"),
    "self-test": ("nuwa.tests.run_checks", "Run numerical and checkpoint checks without a dataset"),
}


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (module_name, help_text) in COMMANDS.items():
        module = importlib.import_module(module_name)
        stage_parser = module.get_args_parser()
        commands.add_parser(name, parents=[stage_parser], add_help=False,
                            help=help_text, description=stage_parser.description,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser


def get_args(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in ("-h", "--help")):
        arguments.insert(0, "derive")
    return get_args_parser().parse_args(arguments)


def nuwa(args=None):
    from nuwa.method.nuwa import run
    return run(get_args() if args is None else args)


def main(argv=None):
    # These affect only this CLI process and its children, before NumPy/PyTorch
    # imports. Explicit --threads controls PyTorch and the comparison protocol.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    args = get_args(argv)
    module = importlib.import_module(COMMANDS[args.command][0])
    return module.run(args)


if __name__ == "__main__":
    main()
