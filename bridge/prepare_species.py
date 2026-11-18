"""Prepare one AP-10K species for NuWa, using only Python's standard library.

Keep this file and bridge_paths.py in bridge/.
Run from the project root:

    python bridge/prepare_species.py --list
    python bridge/prepare_species.py

The default species is dog. Use --species to select another species.
The default data root is bridge/data/ap-10k.

The data root must contain annotations/ and data/. The script keeps the
official train/validation split, original IDs, keypoints, and category metadata.
It filters animal annotations, then keeps the images those annotations use.
Images are reused in place. The test set is never read.

New annotations go to annotations/species/<species>/ inside the data root.
The original annotation files are never overwritten. Rerunning the same
command replaces only the generated subset files for that species.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

if __package__:
    from .bridge_paths import DEFAULT_DATA_ROOT, DEFAULT_SPECIES, resolve_data_root
else:
    from bridge_paths import DEFAULT_DATA_ROOT, DEFAULT_SPECIES, resolve_data_root


def read_annotations(path):
    if not path.is_file():
        raise ValueError("Annotation file not found: {}".format(path))
    with path.open("r", encoding="utf-8-sig") as handle:
        dataset = json.load(handle)
    for field in ("images", "annotations", "categories"):
        if not isinstance(dataset.get(field), list):
            raise ValueError("{} must contain a '{}' list".format(path, field))
    return dataset


def category_names(dataset):
    return {category["id"]: category["name"] for category in dataset["categories"]}


def make_subset(dataset, category_id):
    annotations = [
        annotation for annotation in dataset["annotations"]
        if annotation["category_id"] == category_id
    ]
    image_ids = {annotation["image_id"] for annotation in annotations}
    images = [image for image in dataset["images"] if image["id"] in image_ids]
    if {image["id"] for image in images} != image_ids:
        raise ValueError("Some selected annotations refer to missing image entries")

    # Keep every category's metadata and every original ID for compatibility
    # with the existing AnimalAP10KDataset evaluator.
    subset = dict(dataset)
    subset["images"] = images
    subset["annotations"] = annotations
    return subset


def labeled_animals(dataset):
    return sum(
        not annotation.get("iscrowd", 0)
        and any(value > 0 for value in annotation.get("keypoints", [])[2::3])
        for annotation in dataset["annotations"]
    )


def check_images(dataset, image_root):
    missing = [
        image_root / image["file_name"] for image in dataset["images"]
        if not (image_root / image["file_name"]).is_file()
    ]
    if missing:
        raise ValueError(
            "{} selected image(s) are missing. First missing image: {}".format(
                len(missing), missing[0]
            )
        )


def prepare(args):
    data_root = resolve_data_root(args.data_root)
    source_paths = {
        split: data_root / "annotations" / "ap10k-{}-split{}.json".format(
            split, args.split
        )
        for split in ("train", "val")
    }
    datasets = {
        split: read_annotations(path) for split, path in source_paths.items()
    }
    names = category_names(datasets["train"])
    if names != category_names(datasets["val"]):
        raise ValueError("Train and validation category IDs/names do not match")

    if args.list:
        counts = {
            split: Counter(
                annotation["category_id"]
                for annotation in dataset["annotations"]
            )
            for split, dataset in datasets.items()
        }
        print("{:28s} {:>13s} {:>13s}".format(
            "Species", "Train animals", "Val animals"
        ))
        for category_id, name in sorted(names.items(), key=lambda item: item[1].casefold()):
            print("{:28s} {:13d} {:13d}".format(
                name, counts["train"][category_id], counts["val"][category_id]
            ))
        print("\nUse a species name exactly as shown (capitalization does not matter).")
        print("Counts are animal annotations, before MMPose's normal validity filtering.")
        return

    matching_ids = [
        category_id for category_id, name in names.items()
        if name.casefold() == args.species.strip().casefold()
    ]
    if len(matching_ids) != 1:
        raise ValueError(
            "Species {!r} was not found uniquely. Run with --list first.".format(
                args.species
            )
        )
    category_id = matching_ids[0]
    species = names[category_id]
    slug = re.sub(r"[^a-z0-9]+", "_", species.casefold()).strip("_")
    if not slug:
        slug = "category_{}".format(category_id)
    output_dir = data_root / "annotations" / "species" / slug
    subsets = {
        split: make_subset(dataset, category_id)
        for split, dataset in datasets.items()
    }

    # Complete validation before writing either output file.
    for split, subset in subsets.items():
        if not labeled_animals(subset):
            raise ValueError("No labeled {} animals found for {}".format(split, species))
        check_images(subset, data_root / "data")
    train_files = {image["file_name"] for image in subsets["train"]["images"]}
    val_files = {image["file_name"] for image in subsets["val"]["images"]}
    if train_files & val_files:
        raise ValueError("Train and validation contain overlapping image filenames")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Species: {} (original category ID: {})".format(species, category_id))
    for split, subset in subsets.items():
        output_path = output_dir / source_paths[split].name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(subset, handle, ensure_ascii=False)
            handle.write("\n")
        print("{}: {} images, {} animal annotations ({} labeled, non-crowd)".format(
            split, len(subset["images"]), len(subset["annotations"]),
            labeled_animals(subset)
        ))
        print("  {}".format(output_path))

    print("\nPaths to use AFTER Config.fromfile(...):")
    for split in ("train", "val"):
        annotation_path = (output_dir / source_paths[split].name).as_posix()
        image_prefix = (data_root / "data").as_posix() + "/"
        print("cfg.data.{}.ann_file = {!r}".format(split, annotation_path))
        print("cfg.data.{}.img_prefix = {!r}".format(split, image_prefix))
    print("\nEvaluate cfg.data.val to record the unpruned species baseline AP.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path,
                        help="AP-10K folder; default: bridge/data/ap10k")
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1,
                        help="Official split number; default: 1")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true",
                        help="List species and annotation counts, without writing files")
    action.add_argument("--species", default=DEFAULT_SPECIES,
                        help="One species name from --list; default: dog")
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
