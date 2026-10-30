"""Soup body intake: zilsb2 commitments, name mapping, body hash, relative change and validation."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from zils import bodies, models

SHA = "a" * 64
REVISION = "b" * 40
REPO = "scout/open-body"
HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SHAPES = {
    "language_model.embed_tokens.weight": (8, 4),
    "language_model.layers.0.mlp.weight": (4, 4),
    "language_model.layers.1.mlp.weight": (4, 4),
    "language_model.norm.weight": (4,),
    "visual.patch.weight": (4, 2),
}
NAMES = sorted(SHAPES)


def weights(seed=0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    return {n: torch.randn(s, generator=generator).to(dtype) for n, s in SHAPES.items()}


def checkpoint(
    root, tensors, *, prefix="", shards=1, extras=True, license="apache-2.0", owner=HOTKEY
):
    """A tiny body repo: tensors (renamed with prefix) split over `shards` files, plus card and owner."""
    root = Path(root)
    root.mkdir(parents=True)
    named = {prefix + n: t for n, t in tensors.items()}
    if extras:
        named["lm_head.weight"] = torch.zeros(8, 4, dtype=torch.bfloat16)
        named["mtp.layers.0.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    keys = sorted(named)
    if shards == 1:
        save_file(named, str(root / "model.safetensors"))
    else:
        weight_map = {}
        for i in range(shards):
            file = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
            part = {k: named[k] for k in keys[i::shards]}
            save_file(part, str(root / file))
            weight_map.update(dict.fromkeys(part, file))
        index = {"metadata": {}, "weight_map": weight_map}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
    if license is not None:
        (root / "README.md").write_text(f"---\nlicense: {license}\ntags:\n- text\n---\n# Body\n")
    if owner is not None:
        (root / "zils-owner.json").write_text(json.dumps({"hotkey": owner}))
    return root


class CommitmentTest(unittest.TestCase):
    def test_round_trip(self):
        text = bodies.format_commitment(SHA, REVISION, REPO)
        self.assertEqual(text, f"zilsb2:{SHA}:{REVISION}:{REPO}")
        self.assertEqual(
            bodies.parse_commitment(text), {"sha256": SHA, "revision": REVISION, "repo": REPO}
        )

    def test_rejects(self):
        for text in (
            f"zils1:{SHA}:{REVISION}:{REPO}",
            f"zilsb2:{'A' * 64}:{REVISION}:{REPO}",
            f"zilsb2:{SHA}:main:{REPO}",
            f"zilsb2:{SHA}:{REVISION}:not-a-repo",
            f"zilsb2:{SHA}:{REVISION}",
            None,
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bodies.parse_commitment(text)
        with self.assertRaises(ValueError):
            bodies.format_commitment("x", REVISION, REPO)


class MapNamesTest(unittest.TestCase):
    def test_strips_one_prefix_and_drops_extras(self):
        keys = ["model." + n for n in NAMES] + ["lm_head.weight", "mtp.fc.weight", "model.mtp.x"]
        self.assertEqual(bodies.map_names(keys, NAMES), {n: "model." + n for n in NAMES})
        self.assertEqual(bodies.map_names(NAMES, NAMES), {n: n for n in NAMES})

    def test_rejects(self):
        cases = {
            "unknown": NAMES + ["model.extra.weight"],
            "double prefix": NAMES[1:] + ["model.model." + NAMES[0]],
            "duplicate": NAMES + ["model." + NAMES[0]],
            "missing": NAMES[1:],
            "not in chassis": NAMES + ["language_model.layers.9.mlp.weight"],
        }
        for label, keys in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                bodies.map_names(keys, NAMES)


class LayoutTest(unittest.TestCase):
    def test_load_index_single_and_sharded(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = checkpoint(Path(tmp) / "one", weights(), extras=False)
            self.assertEqual(bodies.load_index(one), {n: ("model.safetensors", n) for n in NAMES})
            many = checkpoint(Path(tmp) / "many", weights(), prefix="model.", shards=3)
            index = bodies.load_index(many)
            self.assertEqual(len(index), len(NAMES) + 2)
            for key, (file, name) in index.items():
                self.assertEqual(key, name)
                self.assertTrue((many / file).is_file())

    def test_hash_stable_across_layouts_and_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tensors = weights()
            folders = [
                checkpoint(Path(tmp) / "a", tensors, extras=False),
                checkpoint(Path(tmp) / "b", tensors, prefix="model.", shards=2),
                checkpoint(Path(tmp) / "c", tensors, prefix="model.", shards=3, extras=False),
            ]
            hashes = {bodies.body_hash(f, NAMES) for f in folders}
            self.assertEqual(len(hashes), 1)
            self.assertRegex(hashes.pop(), "^[a-f0-9]{64}$")
            changed = dict(tensors)
            changed[NAMES[0]] = changed[NAMES[0]].clone()
            changed[NAMES[0]][0, 0] += 1
            other = checkpoint(Path(tmp) / "d", changed)
            self.assertNotEqual(bodies.body_hash(other, NAMES), bodies.body_hash(folders[0], NAMES))

    def test_hash_rejects_non_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = checkpoint(Path(tmp) / "a", weights(dtype=torch.float32))
            with self.assertRaises(ValueError):
                bodies.body_hash(folder, NAMES)

    def test_relative_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_tensors = weights(1)
            base = checkpoint(Path(tmp) / "base", base_tensors, prefix="model.", shards=2)
            same = checkpoint(Path(tmp) / "same", base_tensors)
            self.assertEqual(bodies.relative_change(same, base, NAMES), 0.0)
            moved = dict(base_tensors)
            layer = "language_model.layers.0.mlp.weight"
            moved[layer] = (2 * base_tensors[layer].float()).to(torch.bfloat16)
            moved["language_model.embed_tokens.weight"] = torch.zeros(8, 4, dtype=torch.bfloat16)
            body = checkpoint(Path(tmp) / "body", moved)
            layers = [n for n in NAMES if n.startswith("language_model.layers.")]
            norm = math.sqrt(sum(float(base_tensors[n].float().square().sum()) for n in layers))
            expected = float(base_tensors[layer].float().norm()) / norm
            self.assertAlmostEqual(bodies.relative_change(body, base, NAMES), expected, places=5)
            self.assertGreater(bodies.relative_change(body, base, NAMES, prefix=""), expected)


class ValidateBodyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.chassis = checkpoint(self.root / "chassis", weights(), extras=False)
        self.base = checkpoint(self.root / "qwen", weights(), prefix="model.")

    def tearDown(self):
        self.tmp.cleanup()

    def body(self, name="body", tensors=None, **kwargs):
        return checkpoint(
            self.root / name, tensors or weights(), prefix="model.", shards=2, **kwargs
        )

    def test_accepts_and_returns_hash(self):
        body = self.body()
        for chassis in (self.chassis, SHAPES):
            digest = bodies.validate_body(body, chassis, hotkey=HOTKEY, base=self.base)
            self.assertEqual(digest, bodies.body_hash(body, NAMES))
        for license in ("mit", "bsd-3-clause", "bsd-2-clause", '"Apache-2.0"'):
            folder = self.body(f"body-{license.strip(chr(34))}", license=license)
            bodies.validate_body(folder, self.chassis, hotkey=HOTKEY)

    def test_rejects(self):
        wrong_shape = weights()
        wrong_shape["visual.patch.weight"] = torch.zeros(2, 4, dtype=torch.bfloat16)
        far = {n: (t.float() * 3).to(torch.bfloat16) for n, t in weights().items()}
        cases = {
            "shape": ({"tensors": wrong_shape}, "shape"),
            "dtype": ({"tensors": weights(dtype=torch.float16)}, "bfloat16"),
            "no card": ({"license": None}, "license"),
            "bad licence": ({"license": "llama3"}, "license"),
            "no owner": ({"owner": None}, "zils-owner.json"),
            "wrong owner": (
                {"owner": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"},
                "hotkey",
            ),
            "too far": ({"tensors": far}, "change"),
        }
        for label, (kwargs, message) in cases.items():
            with self.subTest(label):
                body = self.body(label.replace(" ", "-"), **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    bodies.validate_body(body, self.chassis, hotkey=HOTKEY, base=self.base)

    def test_rejects_malformed_owner_and_card(self):
        body = self.body()
        (body / "zils-owner.json").write_text(json.dumps({"hotkey": HOTKEY, "extra": 1}))
        with self.assertRaisesRegex(ValueError, "zils-owner.json"):
            bodies.validate_body(body, self.chassis, hotkey=HOTKEY)
        (body / "zils-owner.json").write_text("not json")
        with self.assertRaisesRegex(ValueError, "zils-owner.json"):
            bodies.validate_body(body, self.chassis, hotkey=HOTKEY)
        other = self.body("no-front-matter")
        (other / "README.md").write_text("license: apache-2.0\n")
        with self.assertRaisesRegex(ValueError, "license"):
            bodies.validate_body(other, self.chassis, hotkey=HOTKEY)

    def test_rejects_mapping_errors(self):
        body = self.body()
        save_file(
            {"model.extra.weight": torch.zeros(1, dtype=torch.bfloat16)},
            str(body / "x.safetensors"),
        )
        index = json.loads((body / "model.safetensors.index.json").read_text())
        index["weight_map"]["model.extra.weight"] = "x.safetensors"
        (body / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "extra"):
            bodies.validate_body(body, self.chassis, hotkey=HOTKEY)

    def test_size_limit(self):
        self.assertEqual(bodies.MAX_BYTES, models.MAX_CANDIDATE_BYTES)
        body = self.body()
        with patch.object(bodies, "MAX_BYTES", 64), self.assertRaisesRegex(ValueError, "bytes"):
            bodies.validate_body(body, self.chassis, hotkey=HOTKEY)

    def test_change_bound_is_configurable(self):
        far = {n: (t.float() * 1.2).to(torch.bfloat16) for n, t in weights().items()}
        body = self.body(tensors=far)
        with self.assertRaisesRegex(ValueError, "change"):
            bodies.validate_body(body, self.chassis, hotkey=HOTKEY, base=self.base)
        bodies.validate_body(body, self.chassis, hotkey=HOTKEY, base=self.base, max_change=0.25)
        bodies.validate_body(body, self.chassis, hotkey=HOTKEY)


def raw_body(root, header_bytes, data=b""):
    """A single-file body whose safetensors header is exactly `header_bytes`."""
    root = Path(root)
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(
        len(header_bytes).to_bytes(8, "little") + header_bytes + data
    )
    return root


def raw_header(**changes):
    """A valid header for weights() laid out back to back; `changes` replace whole tensor entries."""
    header, offset = {}, 0
    for name, shape in sorted(SHAPES.items()):
        size = 2 * math.prod(shape)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    header.update(changes)
    return header, offset


class DeadlineTest(unittest.TestCase):
    def test_a_deadline_overrun_mid_download_is_transient(self):
        import itertools

        from zils import commitments

        class Response:
            status = 200
            headers = {"Content-Length": "4"}

            def __init__(self):
                self.parts = [b"ab", b"cd"]

            def read(self, n):
                return self.parts.pop(0) if self.parts else b""

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        clock = itertools.chain([0, 0], itertools.repeat(10**9))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                commitments,
                "opener",
                return_value=type("O", (), {"open": lambda *a, **k: Response()})(),
            ),
            patch.object(commitments.time, "monotonic", side_effect=lambda: next(clock)),
            self.assertRaisesRegex(OSError, "deadline"),
        ):
            claim = {"repo": REPO, "revision": REVISION}
            commitments.download_files(
                claim, Path(tmp) / "out", ["a.bin"], {"a.bin": {"size": 4}}, 100
            )


class HostileHeaderTest(unittest.TestCase):
    def test_malformed_offsets_and_shapes_are_rejected_as_invalid(self):
        first = NAMES[0]
        size = 2 * math.prod(SHAPES[first])
        entry = {"dtype": "BF16", "shape": list(SHAPES[first])}
        cases = {
            "string offsets": {**entry, "data_offsets": ["0", str(size)]},
            "float offsets": {**entry, "data_offsets": [0.0, float(size)]},
            "bool offsets": {**entry, "data_offsets": [False, size]},
            "negative offsets": {**entry, "data_offsets": [-2, size - 2]},
            "three offsets": {**entry, "data_offsets": [0, size, size]},
            "past the end": {**entry, "data_offsets": [0, 10**12]},
            "negative shape": {"dtype": "BF16", "shape": [-8, -4], "data_offsets": [0, size]},
            "not a mapping": [1, 2],
        }
        with tempfile.TemporaryDirectory() as tmp:
            for label, bad in cases.items():
                with self.subTest(label):
                    header, total = raw_header(**{first: bad})
                    folder = raw_body(
                        Path(tmp) / label.replace(" ", "-"),
                        json.dumps(header).encode(),
                        bytes(total),
                    )
                    with self.assertRaises(ValueError):
                        bodies.body_hash(folder, NAMES)

    def test_deeply_nested_header_is_rejected_as_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = raw_body(Path(tmp) / "deep", b"[" * 200000 + b"]" * 200000)
            with self.assertRaises(ValueError):
                bodies.body_hash(folder, NAMES)
            header, total = raw_header()
            header["__metadata__"] = {"x": json.loads("[" * 50 + "]" * 50)}
            nested = raw_body(Path(tmp) / "meta", json.dumps(header).encode(), bytes(total))
            with self.assertRaises(ValueError):
                bodies.body_hash(nested, NAMES)

    def test_a_body_with_a_non_finite_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chassis = checkpoint(root / "chassis", weights(), extras=False)
            base = checkpoint(root / "qwen", weights(), prefix="model.")
            for label, value in (("nan", math.nan), ("inf", math.inf)):
                with self.subTest(label):
                    tensors = weights()
                    tensors["visual.patch.weight"][0, 0] = value
                    body = checkpoint(root / label, tensors, prefix="model.")
                    with self.assertRaisesRegex(ValueError, "finite"):
                        bodies.validate_body(body, chassis, hotkey=HOTKEY)
                    with self.assertRaisesRegex(ValueError, "finite|change"):
                        bodies.validate_body(body, chassis, hotkey=HOTKEY, base=base)

    def test_a_nan_change_from_the_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chassis = checkpoint(root / "chassis", weights(), extras=False)
            body = checkpoint(root / "body", weights(), prefix="model.")
            with (
                patch.object(bodies, "relative_change", return_value=math.nan),
                self.assertRaisesRegex(ValueError, "change"),
            ):
                bodies.validate_body(body, chassis, hotkey=HOTKEY, base=chassis)


if __name__ == "__main__":
    unittest.main()
