import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import author_exam


def item(kind, answer, options=(), state="s"):
    return {
        "state": state,
        "type": kind,
        "instructions": "Decide.",
        "options": [{"key": k, "description": k.upper()} for k in options],
        "answer": answer,
        "explanation": "because",
    }


FAMILIES = {
    "arts-known-choice-k2": ("arts", "choice", "x", "b"),
    "arts-known-noul-k2": ("arts", "noul", "x", "b"),
    "arts-known-score-k2": ("arts", "score", "x", "b"),
}


class AuthorExamTest(unittest.TestCase):
    def test_candidates_keep_valid_items_and_drop_malformed(self):
        plan = {f"{f}--0": ["r1", "r2"] for f in FAMILIES}
        texts = {
            "arts-known-choice-k2--0": json.dumps(
                {"items": [item("choice", "a", "ab"), item("choice", "z", "ab")]}
            ),
            "arts-known-noul-k2--0": json.dumps(
                {"items": [item("noul", "true"), item("noul", "yes")]}
            ),
            "arts-known-score-k2--0": json.dumps(
                {"items": [item("score", "3", "01234"), item("score", "1", "0123")]}
            ),
        }
        rows = author_exam.candidates(texts, FAMILIES, plan, "d1")
        self.assertEqual(
            [(r["family"], r["label"]) for r in rows],
            [
                ("arts-known-choice-k2", "a"),
                ("arts-known-noul-k2", "true"),
                ("arts-known-score-k2", "3"),
            ],
        )
        score = rows[2]["question"]
        self.assertEqual(score["criteria"], list("01234".upper()))
        self.assertEqual({r["source_id"] for r in rows}, {"r1"})
        self.assertTrue(all(r["block"] == "known" for r in rows))

    def test_finish_keeps_agreements_splits_and_writes_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d1"
            root.mkdir()
            items = []
            for n in range(4):
                items.append({
                    "id": f"c{n}", "family": "arts-known-choice-k2", "category": "arts",
                    "block": "known", "source_id": f"r{n}", "state": f"state {n}",
                    "question": {"type": "choice", "instructions": "Pick.",
                                 "criteria": {"a": "A", "b": "B", "c": "C"}},
                    "label": "a", "explanation": "e",
                })  # fmt: skip
            items.append(dict(items[0], id="dup", source_id="r9"))  # same prompt as c0
            (root / "candidates.json").write_text(json.dumps(items))
            answers = {"c0": "a", "c1": "a", "c2": "b", "c3": "a", "dup": "a"}
            (root / "answers.json").write_text(json.dumps(answers))
            choice = {"arts-known-choice-k2": FAMILIES["arts-known-choice-k2"]}
            (root / "families.json").write_text(json.dumps(choice))
            author_exam.finish(tmp, "d1", calibration=1, test=2)
            calibration = [
                json.loads(x) for x in (root / "calibration.jsonl").read_text().splitlines()
            ]
            test = [json.loads(x) for x in (root / "test.jsonl").read_text().splitlines()]
            self.assertEqual([c["id"] for c in calibration], ["c0"])
            self.assertEqual([c["id"] for c in test], ["c1", "c3"])  # c2 disagreed, dup removed
            self.assertEqual(test[0]["state"], {"text": "state 1"})
            self.assertEqual(set(test[0]["question"]["criteria"]), {"A", "B", "C"})
            registry = json.loads((root / "registry.json").read_text())
            self.assertEqual(
                registry, {"arts-known-choice-k2": {"category": "arts", "type": "choice"}}
            )

    def test_finish_trims_to_exact_counts_and_refuses_short_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d1"
            root.mkdir()
            items = [
                {
                    "id": f"n{n}", "family": "arts-known-noul-k2", "category": "arts",
                    "block": "known", "source_id": f"r{n}", "state": f"state {n}",
                    "question": {"type": "noul", "instructions": "Ironic?", "criteria": None},
                    "label": "true" if n % 2 else "false", "explanation": "e",
                }
                for n in range(6)
            ]  # fmt: skip
            (root / "candidates.json").write_text(json.dumps(items))
            (root / "answers.json").write_text(json.dumps({i["id"]: i["label"] for i in items}))
            noul = {"arts-known-noul-k2": FAMILIES["arts-known-noul-k2"]}
            (root / "families.json").write_text(json.dumps(noul))
            author_exam.finish(tmp, "d1", calibration=2, test=3)  # 6 agreements, 5 kept
            test = (root / "test.jsonl").read_text().splitlines()
            self.assertEqual(len(test), 3)
            with self.assertRaisesRegex(ValueError, "arts-known-noul-k2"):
                author_exam.finish(tmp, "d1", calibration=2, test=5)

    def test_finish_renames_choice_keys_so_names_cannot_leak_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d1"
            root.mkdir()
            criteria = {"opt_wrong": "Paris", "opt_correct": "Lyon", "opt_other": "Nice"}
            items = [
                {
                    "id": f"c{n}", "family": "arts-known-choice-k2", "category": "arts",
                    "block": "known", "source_id": f"r{n}", "state": f"state {n}",
                    "question": {"type": "choice", "instructions": "Pick.", "criteria": criteria},
                    "label": "opt_correct", "explanation": "e",
                }
                for n in range(4)
            ]  # fmt: skip
            (root / "candidates.json").write_text(json.dumps(items))
            (root / "answers.json").write_text(json.dumps({i["id"]: "opt_correct" for i in items}))
            choice = {"arts-known-choice-k2": FAMILIES["arts-known-choice-k2"]}
            (root / "families.json").write_text(json.dumps(choice))
            author_exam.finish(tmp, "d1", calibration=1, test=3)
            for line in (root / "test.jsonl").read_text().splitlines():
                case = json.loads(line)
                keys = list(case["question"]["criteria"])
                self.assertEqual(keys, ["A", "B", "C"])
                self.assertEqual(case["question"]["criteria"][case["label"]], "Lyon")

    def test_new_domain_tasks_rotate_across_draws(self):
        pools = {c: [f"t{c}{i}" for i in range(4)] for c in author_exam.BIGBENCH}
        sources = {f"bb-{t}": (lambda: [("r", {})] * 100) for ts in pools.values() for t in ts}
        sources["glaive"] = lambda: [("r", {})] * 100
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(author_exam.BIGBENCH, pools, clear=True),
            patch.dict(author_exam.NEW_DOMAIN, {}, clear=True),
            patch.dict(author_exam.SOURCES, sources, clear=True),
        ):
            first = author_exam.new_families(tmp, "d1", 72)
            self.assertEqual(len(first), 15)
            (Path(tmp) / "d1").mkdir()
            (Path(tmp) / "d1" / "families.json").write_text(json.dumps(first))
            with self.assertRaises(RuntimeError):  # only one unused task left per category
                author_exam.new_families(tmp, "d2", 72)
            tools = {v[2] for k, v in first.items() if k.startswith("tools-")}
            self.assertIn("glaive", tools)

    def test_new_domain_pools_extend_bigbench_without_reuse(self):
        known = {source for _, _, source, _ in author_exam.KNOWN.values()}
        keys = [k for pool in author_exam.NEW_DOMAIN.values() for k in pool]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(set(author_exam.NEW_DOMAIN) <= set(author_exam.BIGBENCH))
        self.assertFalse(known & set(keys))
        self.assertTrue(all(k in author_exam.SOURCES for k in keys))
        # A used BIG-bench task stays used; a new-domain dataset fills the gap.
        pools = {c: [f"t{c}{i}" for i in range(3)] for c in author_exam.BIGBENCH}
        extra = {c: {f"ds-{c}{i}": ("repo", "path", None) for i in range(3)} for c in pools}
        sources = {f"bb-{t}": (lambda: [("r", {})] * 100) for ts in pools.values() for t in ts}
        sources |= {k: (lambda: [("r", {})] * 100) for p in extra.values() for k in p}
        sources["glaive"] = lambda: [("r", {})] * 100
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(author_exam.BIGBENCH, pools, clear=True),
            patch.dict(author_exam.NEW_DOMAIN, extra, clear=True),
            patch.dict(author_exam.SOURCES, sources, clear=True),
        ):
            first = author_exam.new_families(tmp, "d1", 72)
            (Path(tmp) / "d1").mkdir()
            (Path(tmp) / "d1" / "families.json").write_text(json.dumps(first))
            second = author_exam.new_families(tmp, "d2", 72)
            for category in ("reasoning", "language", "retrieval", "arts"):
                used = {v[2] for k, v in first.items() if k.startswith(category)}
                again = {v[2] for k, v in second.items() if k.startswith(category)}
                self.assertFalse(used & again)
                self.assertEqual(len(used | again), 6)


if __name__ == "__main__":
    unittest.main()
