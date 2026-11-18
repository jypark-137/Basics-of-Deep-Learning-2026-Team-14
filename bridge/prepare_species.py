"""Prepare AP-10K species subsets through the shared NuWa implementation.

Run from the project root:
    python bridge/prepare_species.py --list
    python bridge/prepare_species.py --species dog cat
"""

if __package__:
    from .bridge_paths import PROJECT_ROOT as PROJECT_ROOT
else:
    from bridge_paths import PROJECT_ROOT as PROJECT_ROOT

from nuwa.data.prepare_species import (
    category_names, check_images, get_args_parser, labeled_animals, main,
    make_subset, prepare, read_annotations, run,
)

__all__ = [
    "category_names", "check_images", "get_args_parser", "labeled_animals",
    "main", "make_subset", "prepare", "read_annotations", "run",
]

if __name__ == "__main__":
    main()
