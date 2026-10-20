"""Synthetic box contract checks; fixtures are not model benchmark evidence."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zils import box

EDGES = ("top", "bottom", "left", "right")
EXACT = {"top": 0.2, "bottom": 0.8, "left": 0.1, "right": 0.7}


def cases():
    return [
        {
            "id": name,
            "group_id": "image-" + name,
            "family": "synthetic",
            "state": {"instruction": "Inspect the image."},
            "target": "red rectangle",
            "image": digit * 64 + ".png",
            "box": truth,
        }
        for name, digit, truth in (("present", "a", EXACT), ("absent", "b", None))
    ]


def predictions(data, *, rectangle=EXACT, present=1.0, absent=0.0):
    rows = []
    for case in data:
        for edge in EDGES:
            selected = str(round(rectangle[edge] * 100))
            rows.append(
                {
                    "id": case["id"] + ":" + edge,
                    "probabilities": {str(i): float(str(i) == selected) for i in range(101)},
                    "elapsed_ms": 1,
                }
            )
        confidence = present if case["box"] else absent
        rows.append(
            {
                "id": case["id"] + ":presence",
                "probabilities": {"false": 1 - confidence, "true": confidence},
                "elapsed_ms": 1,
            }
        )
    return rows


class BoxTest(unittest.TestCase):
    def test_every_case_gets_five_questions_without_presence_in_state(self):
        data = cases()
        expanded = box.expand_cases(data)
        self.assertEqual(len(expanded), 10)
        for present, absent in zip(expanded[:5], expanded[5:], strict=True):
            self.assertEqual(present["state"], absent["state"])
            self.assertEqual(present["question"], absent["question"])
            self.assertEqual(present["question"]["type"], "choice")
            self.assertNotIn("box", present["state"])
        self.assertEqual(expanded[0]["label"], "20")
        self.assertEqual(expanded[4]["label"], "true")
        self.assertEqual(expanded[9]["label"], "false")
        self.assertEqual(len(expanded[0]["question"]["criteria"]), 101)

    def test_exact_boxes_and_absence_score_perfectly(self):
        data = cases()
        scores = box.score(data, predictions(data))
        self.assertEqual(scores["mean_iou"], 1)
        self.assertEqual(scores["false_positive_rate"], 0)
        self.assertEqual(scores["presence_brier"], 0)
        self.assertEqual(scores["family_iou"], {"synthetic": 1})

    def test_crossed_edges_are_sorted_and_confidence_is_presence_only(self):
        data = cases()
        crossed = {"top": 0.8, "bottom": 0.2, "left": 0.7, "right": 0.1}
        decoded = box.decode(data, predictions(data, rectangle=crossed, present=0.7))
        self.assertEqual(decoded[0]["box"], EXACT)
        self.assertEqual(decoded[0]["presence_confidence"], 0.7)
        self.assertNotIn("confidence", decoded[0])
        self.assertIsNone(decoded[1]["box"])

    def test_missed_present_and_false_positive_targets_are_penalized(self):
        data = cases()
        scores = box.score(data, predictions(data, present=0.49, absent=0.5))
        self.assertEqual(scores["mean_iou"], 0)
        self.assertEqual(scores["false_positive_rate"], 1)
        self.assertAlmostEqual(scores["presence_brier"], (0.51**2 + 0.5**2) / 2)

    def test_invalid_cases_are_rejected(self):
        mutations = [
            lambda data: data.append(copy.deepcopy(data[0])),
            lambda data: data[0].update(image="../" + "a" * 64 + ".png"),
            lambda data: data[0].update(extra=True),
            lambda data: data[0].update(target=""),
            lambda data: data[0].update(state={"bad": float("nan")}),
            lambda data: data[0]["box"].update(top=0.9),
            lambda data: data[0]["box"].update(left=True),
            lambda data: data[0]["box"].update(right=float("inf")),
            lambda data: data[0]["box"].pop("top"),
            lambda data: data.pop(),
            lambda data: data[0].update(box=None),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(cases())
                mutate(data)
                with self.assertRaises(ValueError):
                    box.expand_cases(data)

    def test_every_prediction_is_validated_even_absent_edges(self):
        data = cases()
        for mutate in (
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(id=rows[1]["id"]),
            lambda rows: rows[0].update(id="unknown"),
            lambda rows: rows[-2]["probabilities"].update({"0": float("nan")}),
            lambda rows: rows[0]["probabilities"].update({"0": True}),
            lambda rows: rows[0]["probabilities"].pop("0"),
            lambda rows: rows[0]["probabilities"].update({"0": 0.5}),
        ):
            with self.subTest(mutate=mutate):
                rows = predictions(data)
                mutate(rows)
                with self.assertRaises(ValueError):
                    box.score(data, rows)

    def test_paired_comparison_rejects_ties_and_accepts_clear_improvement(self):
        data = cases()
        good = predictions(data)
        poor = predictions(data, rectangle={"top": 0, "bottom": 0.1, "left": 0, "right": 0.1})
        result = box.compare(data, poor, good, {"bootstrap_resamples": 100})
        self.assertTrue(result["eligible"])
        self.assertEqual(result["gain"], 1)
        self.assertEqual(result["lower_95"], 1)
        self.assertFalse(box.compare(data, good, good, {"bootstrap_resamples": 100})["eligible"])

    def test_overall_gain_cannot_hide_family_regression(self):
        data = cases()
        extra = copy.deepcopy(data[0])
        extra.update(id="other", group_id="other-image", family="other", image="c" * 64 + ".png")
        data.append(extra)
        good, poor = predictions(data), predictions(data, present=0)
        incumbent = poor[:5] + good[5:]
        candidate = good[:10] + poor[10:]
        result = box.compare(data, incumbent, candidate, {"bootstrap_resamples": 100})
        self.assertFalse(result["eligible"])
        self.assertFalse(result["checks"]["family_iou"])

    def test_positive_gain_with_uncertain_pairing_fails_lower_bound(self):
        data = cases()
        for name, digit in (("second", "c"), ("third", "d")):
            extra = copy.deepcopy(data[0])
            extra.update(id=name, group_id=name, image=digit * 64 + ".png")
            data.append(extra)
        good = predictions(data)
        poor = predictions(data, rectangle={"top": 0, "bottom": 0.1, "left": 0, "right": 0.1})
        incumbent = good[:10] + poor[10:]
        candidate = poor[:5] + good[5:]
        result = box.compare(data, incumbent, candidate, {"bootstrap_resamples": 1000})
        self.assertAlmostEqual(result["gain"], 1 / 3)
        self.assertGreater(result["candidate"]["mean_iou"], 0.4)
        self.assertLess(result["lower_95"], 0)
        self.assertFalse(result["eligible"])

    def test_each_absolute_and_calibration_gate_is_enforced(self):
        data = cases()
        poor = predictions(data, rectangle={"top": 0, "bottom": 0.1, "left": 0, "right": 0.1})
        for candidate, key in (
            (predictions(data, absent=1), "false_positive_rate"),
            (predictions(data, present=0.5), "presence_brier"),
            (
                predictions(data, rectangle={"top": 0.2, "bottom": 0.3, "left": 0.1, "right": 0.7}),
                "mean_iou",
            ),
        ):
            result = box.compare(data, poor, candidate, {"bootstrap_resamples": 100})
            self.assertFalse(result["eligible"])
            self.assertFalse(result["checks"][key])

    def test_policy_rejects_unknown_nonfinite_and_invalid_values(self):
        for policy in (
            {"wrong": 1},
            {"min_iou_gain": float("nan")},
            {"bootstrap_resamples": True},
            {"bootstrap_resamples": 0},
            {"min_mean_iou": -1},
            {"max_false_positive_rate": 2},
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                box.validate_policy(policy)


class DatasetTest(unittest.TestCase):
    def test_seeded_draw_has_auditable_files_and_independent_image_groups(self):
        from scripts.box_dataset import generate

        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / name for name in ("first", "repeat", "other")]
            for root, seed in zip(roots, (23, 23, 24), strict=True):
                generate(root, 8, seed)
            first, repeat, other = roots
            self.assertEqual(
                (first / "manifest.json").read_bytes(), (repeat / "manifest.json").read_bytes()
            )
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(set(manifest), {"version", "scope", "files", "assets"})
            self.assertEqual(manifest["version"], "zils-box-draw/v1")
            self.assertEqual(manifest["scope"], "synthetic-only")
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), digest)
            for name, digest in manifest["assets"].items():
                self.assertEqual(name, digest + ".png")
                self.assertEqual(
                    hashlib.sha256((first / "images" / name).read_bytes()).hexdigest(), digest
                )
            data = [json.loads(line) for line in (first / "cases.jsonl").read_text().splitlines()]
            expanded = [
                json.loads(line) for line in (first / "expanded.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(data), 8)
            self.assertEqual(len(expanded), 40)
            self.assertEqual(len({c["group_id"] for c in data}), 8)
            self.assertEqual(len({c["image"] for c in data}), 8)
            self.assertEqual(sum(c["box"] is not None for c in data), 4)
            self.assertEqual(len({json.dumps(c["state"], sort_keys=True) for c in data}), 1)
            other_manifest = json.loads((other / "manifest.json").read_text())
            self.assertFalse(set(manifest["assets"]) & set(other_manifest["assets"]))
            with self.assertRaises((ValueError, FileExistsError)):
                generate(first, 8, 25)

    def test_generator_command_and_invalid_count(self):
        from scripts.box_dataset import generate

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate(Path(tmp) / "bad", 1, 42)
            output = Path(tmp) / "cli"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.box_dataset",
                    "--out",
                    str(output),
                    "--count",
                    "2",
                    "--seed",
                    "42",
                ],
                check=True,
                capture_output=True,
            )
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
