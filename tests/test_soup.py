"""Soup builder: a uniform average of bodies written on the pinned d3-mini chassis."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from zils import d3, models, soup

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"


def t(value, n=2):
    return torch.full((n,), float(value), dtype=torch.bfloat16)


def body(root, tensors, shards=SHARDS):
    """Tensors split over `shards` with an index (or one model.safetensors without an index)."""
    root.mkdir(parents=True)
    names, weight_map = sorted(tensors), {}
    for i, shard in enumerate(shards):
        part = {n: tensors[n] for n in names[i :: len(shards)]}
        save_file(part, str(root / shard))
        weight_map.update(dict.fromkeys(part, shard))
    if len(shards) > 1:
        (root / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return root


def chassis(root, tensors):
    """A pinned-parent fixture: two shards, an index, every contract file, a real readout and clutter."""
    body(root, tensors)
    for name in models.D3_CONTRACT_FILES:
        (root / name).write_bytes(b"contract " + name.encode())
    save_file({"readout": torch.arange(6.0).reshape(2, 3)}, str(root / "readout.safetensors"))
    models.write_metadata(root, kind="base", model=models.D3)
    (root / "README.md").write_text("upstream card")
    (root / ".cache").mkdir()
    (root / ".cache" / "lock").write_text("")
    return root


def read(root):
    return {k: v.float().tolist() for s in SHARDS for k, v in load_file(str(root / s)).items()}


class SoupTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.chassis = chassis(self.root / "chassis", {"a": t(0), "b": t(0), "c": t(0)})
        pinned = hashlib.sha256((self.chassis / "readout.safetensors").read_bytes()).hexdigest()
        self.enterContext(patch.dict(d3.PINS["files"], {"readout.safetensors": pinned}))

    def build(self, *bodies):
        return soup.build(self.root / "out", self.chassis, list(bodies), model=models.D3)

    def test_uniform_average_in_the_chassis_layout(self):
        one = body(self.root / "one", {"a": t(1), "b": t(2), "c": t(3)})
        two = body(
            self.root / "two", {"a": t(3), "b": t(4), "c": t(6)}, shards=("model.safetensors",)
        )
        out = self.build((one, None), (two, None))
        self.assertEqual(out, self.root / "out")
        self.assertEqual(read(out), {"a": [2.0, 2.0], "b": [3.0, 3.0], "c": [4.5, 4.5]})
        for shard in SHARDS:
            self.assertEqual(
                set(load_file(str(out / shard))), set(load_file(str(self.chassis / shard)))
            )
            self.assertEqual(
                {v.dtype for v in load_file(str(out / shard)).values()}, {torch.bfloat16}
            )
        self.assertEqual(d3.weight_map(out), d3.weight_map(self.chassis))

    def test_strip_model_renames_and_ignores_tensors_the_chassis_lacks(self):
        hf = body(
            self.root / "hf",
            {"model.a": t(4), "model.b": t(4), "model.c": t(4), "lm_head.weight": t(9),
             "mtp.fc.weight": t(7)},
            shards=("model.safetensors",),
        )  # fmt: skip
        plain = body(self.root / "plain", {"a": t(2), "b": t(0), "c": t(2)})
        out = self.build((hf, "strip-model"), (plain, None))
        self.assertEqual(read(out), {"a": [3.0, 3.0], "b": [2.0, 2.0], "c": [3.0, 3.0]})

    def test_rejects_a_body_missing_a_chassis_tensor_before_writing(self):
        good = body(self.root / "good", {"a": t(1), "b": t(1), "c": t(1)})
        short = body(self.root / "short", {"a": t(1), "b": t(1)}, shards=("model.safetensors",))
        unrenamed = body(self.root / "hf", {"model.a": t(1), "model.b": t(1), "model.c": t(1)})
        for bad in ((short, None), (unrenamed, None)):
            with self.subTest(bad=bad[0].name), self.assertRaises(KeyError):
                self.build((good, None), bad)
            self.assertFalse((self.root / "out").exists())

    def test_rejects_a_body_tensor_of_another_shape(self):
        wide = body(self.root / "wide", {"a": t(1, 3), "b": t(1), "c": t(1)})
        with self.assertRaisesRegex(ValueError, "shape"):
            self.build((wide, None))
        self.assertFalse((self.root / "out").exists())

    def test_single_body_build_equals_that_body(self):
        one = body(self.root / "one", {"a": t(1.5), "b": t(-2), "c": t(7)})
        out = self.build((one, None))
        self.assertEqual(soup.soup_hash(out), soup.soup_hash(one))
        self.assertNotEqual(soup.soup_hash(out), soup.soup_hash(self.chassis))

    def test_output_is_exactly_a_candidate_with_the_chassis_files(self):
        one = body(self.root / "one", {"a": t(1), "b": t(1), "c": t(1)})
        out = self.build((one, None))
        files = models.candidate_files(models.D3, "candidate")
        self.assertEqual({p.name for p in out.iterdir()}, set(files))
        meta = models.metadata(out)
        self.assertEqual(meta["kind"], "candidate")
        self.assertEqual(meta["model"], models.D3)
        expected = self.root / "expected"
        expected.mkdir()
        models.write_metadata(expected, kind="candidate", model=models.D3)
        self.assertEqual((out / "model.json").read_bytes(), (expected / "model.json").read_bytes())
        for name in (INDEX, *models.D3_CONTRACT_FILES):
            self.assertEqual((out / name).read_bytes(), (self.chassis / name).read_bytes(), name)
        pins = {
            name: hashlib.sha256((self.chassis / name).read_bytes()).hexdigest()
            for name in models.D3_CONTRACT_FILES
        }
        with (
            patch.dict(d3.PINS["files"], pins),
            patch("zils.d3.pinned_base_path", return_value=self.chassis),
        ):
            d3.validate_candidate(out, models.D3)

    def test_a_chassis_with_a_retrained_readout_is_rejected(self):
        one = body(self.root / "one", {"a": t(1), "b": t(1), "c": t(1)})
        save_file({"readout": torch.ones(2, 3)}, str(self.chassis / "readout.safetensors"))
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.build((one, None))

    def test_a_tampered_readout_is_rejected(self):
        one = body(self.root / "one", {"a": t(1), "b": t(1), "c": t(1)})
        copy = shutil.copyfile

        def tamper(source, target):
            copy(source, target)
            if Path(target).name == "readout.safetensors":
                save_file({"readout": torch.ones(2, 3)}, str(target))
            return target

        with (
            patch("zils.soup.shutil.copyfile", side_effect=tamper),
            self.assertRaisesRegex(ValueError, "readout"),
        ):
            self.build((one, None))

    def test_soup_hash_covers_names_shapes_and_bytes(self):
        base = soup.soup_hash(body(self.root / "x", {"a": t(1), "b": t(2)}))
        resharded = body(self.root / "y", {"a": t(1), "b": t(2)}, shards=("model.safetensors",))
        self.assertEqual(soup.soup_hash(resharded), base)
        for name, tensors in (
            ("value", {"a": t(1), "b": t(3)}),
            ("name", {"a": t(1), "d": t(2)}),
            ("shape", {"a": t(1), "b": torch.full((1, 2), 2.0, dtype=torch.bfloat16)}),
        ):
            with self.subTest(name=name):
                self.assertNotEqual(soup.soup_hash(body(self.root / name, tensors)), base)
        self.assertRegex(base, "^[0-9a-f]{64}$")

    def test_rejects_an_unknown_rename_and_no_bodies(self):
        one = body(self.root / "one", {"a": t(1), "b": t(1), "c": t(1)})
        with self.assertRaises(ValueError):
            self.build((one, "lowercase"))
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()
