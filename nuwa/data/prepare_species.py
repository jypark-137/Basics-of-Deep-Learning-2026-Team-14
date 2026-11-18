"""Prepare AP-10K species annotations while retaining official train/validation splits."""


import argparse
from collections import Counter
import json
from pathlib import Path
from numbers import Integral
from ..paths import DEFAULT_DATA_ROOT, DEFAULT_SPECIES, resolve_data_root
from ..species import add_species_argument, category_ids, normalize_species, species_slug

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


def make_subset(dataset, selected_ids):
    selected_ids = {selected_ids} if isinstance(selected_ids, Integral) else set(selected_ids)
    annotations = [
        annotation for annotation in dataset["annotations"]
        if annotation["category_id"] in selected_ids
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

    species = normalize_species(args.species)
    selected_ids = category_ids(datasets["train"]["categories"], species)
    slug = species_slug(species)
    output_dir = data_root / "annotations" / "species" / slug
    subsets = {
        split: make_subset(dataset, selected_ids)
        for split, dataset in datasets.items()
    }

    # Complete validation before writing either output file.
    for split, subset in subsets.items():
        for category_id in selected_ids:
            if not labeled_animals(make_subset(subset, category_id)):
                raise ValueError("No labeled {} animals found for {}".format(split, names[category_id]))
        check_images(subset, data_root / "data")
    train_files = {image["file_name"] for image in subsets["train"]["images"]}
    val_files = {image["file_name"] for image in subsets["val"]["images"]}
    if train_files & val_files:
        raise ValueError("Train and validation contain overlapping image filenames")

    if species == ["all"]:
        print("All species use the existing official train/validation annotations.")
        return source_paths
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Species: {} (original category IDs: {})".format(", ".join(species), selected_ids))
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
    return {split: output_dir / path.name for split, path in source_paths.items()}


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path,
                        help="AP-10K folder; default: data/ap-10k")
    parser.add_argument("--split", type=int, choices=(1, 2, 3), default=1,
                        help="Official split number; default: 1")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true",
                        help="List species and annotation counts, without writing files")
    add_species_argument(action, default=DEFAULT_SPECIES)
    return parser


def run(args, parser=None):
    parser = parser or get_args_parser()
    try:
        prepare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, "Error: {}\n".format(error))


def main(argv=None):
    parser = get_args_parser()
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
