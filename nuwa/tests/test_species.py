"""Species unions, split integrity, CLI lists and legacy checkpoint provenance."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import main
from ..data.prepare_species import prepare
from ..dataset.utils import evaluation_paths, validate_subsets
from ..species import category_ids, normalize_species, species_slug
from ..utils.pose_results import validate_provenance


class SpeciesChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nuwa_species_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "annotations").mkdir()
        (self.root / "data").mkdir()
        self.sources = {}
        for offset, split in ((0, "train"), (100, "val")):
            categories = [{"id": 8, "name": "dog"}, {"id": 24, "name": "cat"},
                          {"id": 10, "name": "wolf"}, {"id": 51, "name": "brown bear"}]
            images = [{"id": offset + i + 1, "file_name": "{}_{}.jpg".format(split, i)}
                      for i in range(3)]
            annotations = [{"id": offset + i + 1, "image_id": images[max(0, i - 1)]["id"],
                            "category_id": item["id"], "keypoints": [1, 1, 2] * 17,
                            "num_keypoints": 17, "bbox": [0, 0, 4, 4], "area": 16}
                           for i, item in enumerate(categories)]
            data = {"images": images, "annotations": annotations, "categories": categories}
            for image in images:
                (self.root / "data" / image["file_name"]).touch()
            self.sources[split] = data
            self.write_source(split, data)

    def write_source(self, split, data):
        path = self.root / "annotations" / ("ap10k-{}-split1.json".format(split))
        path.write_text(json.dumps(data), encoding="utf-8")

    def prepare(self, species):
        with redirect_stdout(io.StringIO()):
            return prepare(SimpleNamespace(data_root=self.root, split=1, list=False, species=species))

    def test_normalization_and_group_paths(self):
        self.assertEqual(normalize_species(" DOG "), ["dog"])
        self.assertEqual(normalize_species(["dog", "CAT", "Dog"]), ["cat", "dog"])
        self.assertEqual(species_slug(["dog", "cat"]), "cat__dog")
        self.assertEqual(species_slug("dog"), "dog")
        self.assertEqual(species_slug(["Brown Bear", "cat"]), "brown_bear__cat")
        many = ["species " + str(i) for i in range(54)]
        self.assertLessEqual(len(species_slug(many)), 100)
        self.assertEqual(species_slug(many), species_slug(list(reversed(many))))
        for invalid in (None, [], "", ["dog", ""], [1], ["all", "cat"]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_species(invalid)

    def test_union_keeps_original_ids_and_deduplicates_shared_images(self):
        before = {part: (self.root / "annotations" / ("ap10k-{}-split1.json".format(part))).read_bytes()
                  for part in self.sources}
        paths = self.prepare(["DOG", "cat", "dog"])
        slug, validated, ids = validate_subsets(self.root, ["cat", "dog"], 1)
        self.assertEqual((slug, ids), ("cat__dog", [8, 24]))
        self.assertEqual(paths, validated)
        for part, path in paths.items():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({item["category_id"] for item in data["annotations"]}, {8, 24})
            self.assertEqual(len(data["images"]), 1)
            self.assertEqual(data["categories"], self.sources[part]["categories"])
            self.assertEqual(data["annotations"], self.sources[part]["annotations"][:2])
            self.assertEqual((self.root / "annotations" / path.name).read_bytes(), before[part])

    def test_single_species_layout_and_multiword_names(self):
        paths = self.prepare("dog")
        self.assertEqual(paths["train"].parent.name, "dog")
        self.assertEqual(validate_subsets(self.root, ["DOG"], 1)[2], [8])
        paths = self.prepare(["brown bear", "dog"])
        self.assertEqual(paths["val"].parent.name, "brown_bear__dog")
        self.assertEqual(validate_subsets(self.root, ["dog", "brown bear"], 1)[2], [8, 51])

    def test_unknown_missing_and_outside_species_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not found uniquely"):
            self.prepare(["dog", "missing species"])
        paths = self.prepare(["dog", "cat"])
        original = json.loads(paths["val"].read_text(encoding="utf-8"))
        for ids in ({8}, {8, 24, 10}):
            altered = copy.deepcopy(original)
            altered["annotations"] = [item for item in self.sources["val"]["annotations"]
                                      if item["category_id"] in ids]
            paths["val"].write_text(json.dumps(altered), encoding="utf-8")
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_subsets(self.root, ["cat", "dog"], 1)

    def test_missing_labeled_category_is_rejected_before_writing(self):
        self.sources["val"]["annotations"][1]["keypoints"] = [0] * 51
        self.write_source("val", self.sources["val"])
        with self.assertRaisesRegex(ValueError, "No labeled val animals"):
            self.prepare(["dog", "cat"])
        self.assertFalse((self.root / "annotations/species/cat__dog").exists())

    def test_train_val_overlap_is_rejected(self):
        self.sources["val"]["images"][0]["file_name"] = "train_0.jpg"
        self.write_source("val", self.sources["val"])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.prepare(["cat", "dog"])

    def test_all_species_use_original_files(self):
        before = {part: (self.root / "annotations" / ("ap10k-{}-split1.json".format(part))).read_bytes()
                  for part in self.sources}
        paths = self.prepare(["all"])
        self.assertEqual(validate_subsets(self.root, "all", 1)[2], [8, 10, 24, 51])
        for part, path in paths.items():
            self.assertEqual(path.read_bytes(), before[part])
        self.assertFalse((self.root / "annotations/species/all").exists())

    def test_custom_evaluation_annotations_check_the_complete_selection(self):
        paths = self.prepare(["cat", "dog"])
        result = evaluation_paths(self.root, ["DOG", "CAT"], "val", 1, paths["val"])
        self.assertEqual(result[0], paths["val"])
        with self.assertRaises(ValueError):
            evaluation_paths(self.root, "dog", "val", 1, paths["val"])
        self.assertEqual(category_ids(self.sources["train"]["categories"], ["cat", "dog"]), [8, 24])

    def test_legacy_and_list_checkpoint_provenance(self):
        info = {"stage": "NuWa-SKP-anchor", "species": "DOG", "split": 1,
                "base_checkpoint_sha256": "base"}
        validate_provenance(info, "anchor", ["dog"], 1, {"base"}, {"anchor"})
        info["species"] = ["dog", "cat"]
        validate_provenance(info, "anchor", ["CAT", "DOG"], 1, {"base"}, {"anchor"})
        for wrong in (["dog"], ["cat", "wolf"]):
            with self.assertRaisesRegex(ValueError, "species differs"):
                validate_provenance(info, "anchor", wrong, 1, {"base"}, {"anchor"})

    def test_all_species_commands_accept_lists(self):
        parser = main.get_args_parser()
        required = {"prune": ["--anchor", "anchor.pth", "--pruning-rate", "0.4"],
                    "compare": ["--anchor", "anchor.pth", "--edge", "edge=edge.pth"],
                    "ablate": ["--anchor", "anchor.pth", "--rates", "0.4"]}
        for command in ("prepare", "derive", "anchor", "prune", "evaluate", "train", "compare", "ablate"):
            with self.subTest(command=command):
                args = parser.parse_args([command] + required.get(command, [])
                                         + ["--species", "dog", "cat", "brown bear"])
                self.assertEqual(args.species, ["dog", "cat", "brown bear"])


if __name__ == "__main__":
    unittest.main()
