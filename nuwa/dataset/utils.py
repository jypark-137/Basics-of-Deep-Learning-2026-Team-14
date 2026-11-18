"""AP-10K species validation, dataset construction and batch transfer."""


import copy
import json
import os
import torch
from ..paths import DEFAULT_DATA_ROOT, DEFAULT_SPECIES, resolve_data_root, resolve_input_file
from ..model.utils import pose_config
from device_utils import resolve_device
from ..species import category_ids, normalize_species, species_slug

def evaluation_paths(data_root, species, split, split_id, ann_file=None):
    """Validate the selected dataset without loading images."""
    data_root = resolve_data_root(data_root)
    species = normalize_species(species)
    all_species = species == ["all"]
    slug = species_slug(species)

    filename = "ap10k-{}-split{}.json".format(split, split_id)
    if ann_file is not None:
        annotation_path = resolve_input_file(ann_file, "--ann-file")
    elif all_species:
        annotation_path = data_root / "annotations" / filename
    else:
        annotation_path = data_root / "annotations" / "species" / slug / filename
    if not annotation_path.is_file():
        if all_species or ann_file is not None:
            raise ValueError("Annotation file not found: {}".format(annotation_path))
        if split == "test":
            raise ValueError(
                "Species test annotations not found: {}. prepare_species.py creates only "
                "train/val subsets. To reproduce the complete test benchmark, "
                "use --species all --split test.".format(annotation_path)
            )
        raise ValueError(
            "Species subset not found: {}. Run main.py prepare "
            "with this --data-root and --species first.".format(annotation_path)
        )

    image_root = data_root / "data"
    if not image_root.is_dir():
        raise ValueError("Image directory not found: {}".format(image_root))
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        annotations = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if not isinstance(annotations.get(key), list) or not annotations[key]:
            raise ValueError("{} needs a nonempty {} list".format(annotation_path, key))
    selected_ids = set(category_ids(annotations["categories"], species))
    present_ids = {annotation["category_id"] for annotation in annotations["annotations"]}
    if not present_ids <= selected_ids:
        raise ValueError("The selected annotations include species outside {}".format(species))
    if not all_species and present_ids != selected_ids:
        raise ValueError("The selected annotations must include every requested species: {}".format(species))
    image_ids = {image["id"] for image in annotations["images"]}
    if any(annotation["image_id"] not in image_ids for annotation in annotations["annotations"]):
        raise ValueError("Some annotations refer to missing image records")
    for image in annotations["images"]:
        if not (image_root / image["file_name"]).is_file():
            raise ValueError("Image not found: {}".format(image_root / image["file_name"]))
    return annotation_path, image_root, slug


def validate_subsets(data_root, species, split):
    """Validate a prepared species union and preserve the official train/val split."""
    data_root = resolve_data_root(data_root)
    species = normalize_species(species)
    slug = species_slug(species)
    paths, filenames, selected_ids = {}, {}, []
    for part in ("train", "val"):
        path, _, _ = evaluation_paths(data_root, species, part, split)
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        for key in ("images", "annotations", "categories"):
            if not isinstance(data.get(key), list) or not data[key]:
                raise ValueError("{} needs a nonempty {} list".format(path, key))
        ids = category_ids(data["categories"], species)
        for category_id in ids:
            if not any(item["category_id"] == category_id and not item.get("iscrowd", 0)
                       and any(value > 0 for value in item.get("keypoints", [])[2::3])
                       for item in data["annotations"]):
                raise ValueError("No labeled {} animals for category {}".format(part, category_id))
        image_ids = {item["id"] for item in data["images"]}
        if any(item["image_id"] not in image_ids for item in data["annotations"]):
            raise ValueError("{} contains annotations with missing image records".format(path))
        filenames[part] = {item["file_name"] for item in data["images"]}
        for filename in filenames[part]:
            if not (data_root / "data" / filename).is_file():
                raise ValueError("Image not found: {}".format(data_root / "data" / filename))
        paths[part] = path
        selected_ids.append(ids)
    if selected_ids[0] != selected_ids[1]:
        raise ValueError("Train and validation category IDs differ")
    if filenames["train"] & filenames["val"]:
        raise ValueError("Species train and validation contain overlapping image filenames")
    return slug, paths, selected_ids[0]


def make_loaders(cfg, paths, data_root, args):
    from mmpose.datasets import build_dataset, build_dataloader

    pin_memory = resolve_device(getattr(args, "device", "auto")).type == "cuda"
    for part in ("train", "val"):
        cfg.data[part].ann_file = str(paths[part])
        cfg.data[part].img_prefix = str(data_root / "data") + os.sep
        cfg.data[part].test_mode = part == "val"
    train_data = build_dataset(cfg.data.train)
    if len(train_data) == 0:
        raise ValueError("No usable training animals remain after dataset filtering")
    train_loader = build_dataloader(
        train_data, samples_per_gpu=args.batch_size, workers_per_gpu=0,
        dist=False, shuffle=True, drop_last=False, pin_memory=pin_memory,
    )
    val_loader = None
    if not args.smoke_test:
        val_data = build_dataset(cfg.data.val)
        if len(val_data) == 0:
            raise ValueError("No usable validation animals remain after dataset filtering")
        val_loader = build_dataloader(
            val_data, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False, pin_memory=pin_memory,
        )
        print("Validation animals:", len(val_data), flush=True)
    print("Training animals:", len(train_data), flush=True)
    return train_loader, val_loader


def move_batch(batch, device):
    from mmcv.parallel import scatter
    if not isinstance(device, torch.device):
        device = resolve_device(device)
    destination = -1 if device.type == "cpu" else (device.index or 0)
    return scatter(batch, [destination])[0]


def build_pose_dataset(cfg, args, split="train", ann_file=None):
    """Build the existing AP-10K pipeline for one explicitly selected split."""
    from mmpose.datasets import build_dataset
    data_root = resolve_data_root(getattr(args, "data_root", DEFAULT_DATA_ROOT))
    species = getattr(args, "species", None) or DEFAULT_SPECIES
    split_id = getattr(args, "split_id", None) or getattr(args, "split", 1)
    if isinstance(split_id, str):
        split_id = 1
    if isinstance(split_id, bool) or split_id not in (1, 2, 3):
        raise ValueError("The AP-10K split ID must be 1, 2, or 3")
    annotation, image_root, _ = evaluation_paths(data_root, species, split, split_id, ann_file)
    dataset_cfg = copy.deepcopy(cfg.data[split])
    dataset_cfg.ann_file = str(annotation)
    dataset_cfg.img_prefix = str(image_root) + os.sep
    dataset_cfg.test_mode = split != "train"
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        raise ValueError("No usable {} animals remain".format(split))
    return dataset


def get_dataset(args, cfg=None):
    cfg = cfg if cfg is not None else pose_config(getattr(args, "config", None))
    return build_pose_dataset(cfg, args, "train"), build_pose_dataset(cfg, args, "val")


def get_dataloader(args, cfg=None):
    """Return training and validation loaders; the official test split is separate."""
    from mmpose.datasets import build_dataloader
    cfg = cfg if cfg is not None else pose_config(getattr(args, "config", None))
    train_data = build_pose_dataset(cfg, args, "train")
    batch_size = getattr(args, "batch_size", getattr(args, "train_batch_size", 1))
    pin_memory = resolve_device(getattr(args, "device", "auto")).type == "cuda"
    if batch_size <= 0:
        raise ValueError("Training batch size must be positive")
    train_loader = build_dataloader(train_data, samples_per_gpu=batch_size, workers_per_gpu=0,
                                    dist=False, shuffle=True, drop_last=False, pin_memory=pin_memory)
    val_loader = None
    if not getattr(args, "smoke_test", False):
        val_data = build_pose_dataset(cfg, args, "val")
        val_loader = build_dataloader(val_data, samples_per_gpu=1, workers_per_gpu=0,
                                      dist=False, shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader
