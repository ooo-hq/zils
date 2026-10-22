"""Sealed exams bind provenance and independent questions before inference."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_base_metrics import exam
from zils import base_exam, base_round, benchmark


def inputs(prefix="one"):
    cases = {
        s: sum((exam(k, prefix=f"{prefix}-{s}") for k in ("choice", "noul", "score")), [])
        for s in ("calibration", "test")
    }
    for rows in cases.values():
        for c in rows:
            if c["block"] == "new":
                c["family"] += "-" + prefix
    registry = {
        c["family"]: {
            "type": c["question"]["type"],
            "category": c["category"],
            **({"ordered_equally_spaced": True} if c["question"]["type"] == "score" else {}),
        }
        for c in cases["test"]
    }
    receipt = {"artifact_sha256": "a" * 64, "url": "https://example.org/seed", "published_at": 10.0}
    history = {
        "version": base_exam.HISTORY_VERSION,
        "families": {c["family"]: receipt for c in cases["test"] if c["block"] == "known"},
        "cases": [],
    }
    for family, entry in registry.items():
        if family in history["families"]:
            entry["publication"] = receipt
    return cases, registry, history


def sealed(root, prefix="one"):
    cases, registry, history = inputs(prefix)
    base_exam.seal(
        **cases,
        registry=registry,
        blueprint=base_exam.blueprint(cases),
        pool_id="d3-mini",
        history=history,
        sealed_at=20.0,
        out=root,
    )
    return base_exam.audit(root)


class BaseExamTest(unittest.TestCase):
    def test_seal_audit_identity_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "draw"
            manifest, cases = sealed(root)
            self.assertEqual(manifest["version"], "zils-base-holdout/v2")
            self.assertEqual(base_round.audit(root), (manifest, cases))
            self.assertEqual(
                base_round.draw(root)["manifest_sha256"],
                benchmark.file_hash(root / "manifest.json"),
            )
            with self.assertRaises(FileExistsError):
                sealed(root)
            self.assertNotIn("reasoning-known-choice", json.dumps(base_exam.blueprint(cases)))
            (root / "test.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                base_exam.audit(root)

    def test_invalid_content_never_seals(self):
        def no_publication(c, r, h):
            h["families"].clear()

        def soft(c, r, h):
            c["test"][0]["target_probabilities"] = {"a": 0.5, "b": 0.5}

        def ordinal(c, r, h):
            next(v for v in r.values() if v["type"] == "score")["ordered_equally_spaced"] = False

        def mixed(c, r, h):
            c["test"][0]["category"] = "arts"

        def missing_type(c, r, h):
            c["test"][:] = [x for x in c["test"] if x["question"]["type"] != "score"]

        def collision(c, r, h):
            c["test"][0]["source_id"] = c["calibration"][0]["source_id"]

        def unsafe(c, r, h):
            c["test"][0]["image"] = "../private.png"

        for mutate in (no_publication, soft, ordinal, mixed, missing_type, collision, unsafe):
            with self.subTest(mutate=mutate.__name__), tempfile.TemporaryDirectory() as tmp:
                c, r, h = inputs()
                mutate(c, r, h)
                with self.assertRaises(ValueError):
                    base_exam.seal(
                        **c,
                        registry=r,
                        blueprint=base_exam.blueprint(c),
                        history=h,
                        pool_id="d3-mini",
                        sealed_at=20.0,
                        out=Path(tmp) / "draw",
                    )

    def test_disjoint_rejects_copies_lineage_and_new_family_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = sealed(Path(tmp) / "a", "a")
            second = sealed(Path(tmp) / "b", "b")
            base_exam.disjoint([first, second])
            for field in ("id", "source_id", "state"):
                copied = copy.deepcopy(second)
                copied[1]["test"][0][field] = first[1]["calibration"][0][field]
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlap"):
                    base_exam.disjoint([first, copied])
            copied = copy.deepcopy(second)
            new = next(c for c in first[1]["test"] if c["block"] == "new")
            next(c for c in copied[1]["test"] if c["block"] == "new")["family"] = new["family"]
            with self.assertRaisesRegex(ValueError, "new.*famil"):
                base_exam.disjoint([first, copied])

    def test_fingerprints_normalize_unicode_whitespace_and_choice_order(self):
        c = inputs()[0]["test"][0]
        c["state"] = {"text": "caf\u00e9  here"}
        duplicate = copy.deepcopy(c)
        duplicate.update(id="renamed", label="b")
        duplicate["state"] = {"text": "cafe\u0301\n here"}
        duplicate["question"] = dict(reversed(list(c["question"].items())))
        duplicate["question"]["criteria"] = {"b": "B", "a": "A"}
        self.assertTrue(set(base_exam.fingerprints(c)) & set(base_exam.fingerprints(duplicate)))

    def test_history_retires_future_evaluation_preserves_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "draw"
            sealed(root)
            history = inputs()[2]
            # Record actual public artifact; helper never uploads.
            artifact = Path(tmp) / "published.json"
            artifact.write_text("public export")
            updated = base_exam.publish_history(
                root,
                artifact=artifact,
                url="https://example.org/export",
                published_at=40.0,
                history=history,
            )
            self.assertEqual(
                base_exam.audit(root, history=updated, as_of=30.0)[0]["pool_id"], "d3-mini"
            )
            with self.assertRaisesRegex(ValueError, "disclosed"):
                base_exam.audit(root, history=updated, as_of=41.0)
            other = Path(tmp) / "other"
            c, r, _ = inputs()
            with self.assertRaisesRegex(ValueError, "disclosed"):
                base_exam.seal(
                    **c,
                    registry=r,
                    blueprint=base_exam.blueprint(c),
                    history=updated,
                    pool_id="d3-lite",
                    sealed_at=50.0,
                    out=other,
                )

    def test_seed_starts_empty_history_and_first_publication_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r, _ = inputs()
            for entry in r.values():
                entry.pop("publication", None)
            root = Path(tmp) / "seed"
            base_exam.seed(c["test"], r, out=root)
            empty = {"version": base_exam.HISTORY_VERSION, "families": {}, "cases": []}
            artifact = root / "cases.jsonl"
            h = base_exam.publish_history(
                root,
                artifact=artifact,
                url="https://example.org/seed",
                published_at=10.0,
                history=empty,
            )
            again = base_exam.publish_history(
                root,
                artifact=artifact,
                url="https://example.org/seed2",
                published_at=30.0,
                history=h,
            )
            self.assertEqual(h, again)
            self.assertEqual(len(h["families"]), len(r))

    def test_manifest_bounds_and_blueprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, value in enumerate((float("nan"), -1, "tomorrow")):
                root = Path(tmp) / str(i)
                sealed(root)
                path = root / "manifest.json"
                m = json.loads(path.read_text())
                m["sealed_at"] = value
                path.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError, "time"):
                    base_exam.audit(root)
            root = Path(tmp) / "link"
            sealed(root)
            target = root / "test.jsonl"
            target.rename(root / "real")
            target.symlink_to(root / "real")
            with self.assertRaisesRegex(ValueError, "regular|files"):
                base_exam.audit(root)
            c, r, h = inputs()
            with self.assertRaisesRegex(ValueError, "blueprint"):
                base_exam.seal(
                    **c,
                    registry=r,
                    blueprint={},
                    pool_id="d3-mini",
                    history=h,
                    sealed_at=20.0,
                    out=Path(tmp) / "bad",
                )
