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
            (root / "families.json").write_text(json.dumps(FAMILIES))
            author_exam.finish(tmp, "d1", calibration=1)
            calibration = [
                json.loads(x) for x in (root / "calibration.jsonl").read_text().splitlines()
            ]
            test = [json.loads(x) for x in (root / "test.jsonl").read_text().splitlines()]
            self.assertEqual([c["id"] for c in calibration], ["c0"])
            self.assertEqual([c["id"] for c in test], ["c1", "c3"])  # c2 disagreed, dup removed
            self.assertEqual(test[0]["state"], {"text": "state 1"})
            self.assertEqual(set(test[0]["question"]["criteria"]), {"a", "b", "c"})
            registry = json.loads((root / "registry.json").read_text())
            self.assertEqual(
                registry, {"arts-known-choice-k2": {"category": "arts", "type": "choice"}}
            )

    def test_new_domain_tasks_rotate_across_draws(self):
        pools = {c: [f"t{c}{i}" for i in range(4)] for c in author_exam.BIGBENCH}
        sources = {f"bb-{t}": (lambda: [("r", {})] * 100) for ts in pools.values() for t in ts}
        sources["glaive"] = lambda: [("r", {})] * 100
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(author_exam.BIGBENCH, pools, clear=True),
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


if __name__ == "__main__":
    unittest.main()
