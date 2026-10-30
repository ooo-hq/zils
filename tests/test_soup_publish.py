import copy
import hashlib
import json
import unittest

from zils import soup_publish

SECRET_IDS = ["zz-secret-item-alpha", "zz-secret-item-beta", "zz-secret-item-gamma"]
SECRET_LABELS = ["zz-label-approve-xyz", "zz-label-deny-xyz"]
SECRET_FAMILIES = ["zz-family-tax-law", "zz-family-rare-birds"]
SECRET_TEXT = ["zz-question-what-is", "zz-state-hidden"]


def private_round():
    return {
        "round_id": "mini-2026-11-03",
        "pool_id": "mini",
        "seed": "0xabc123",
        "subset_ids": list(reversed(SECRET_IDS)),
        "members": [
            {
                "hotkey": "5Hot1",
                "body_sha256": "a" * 64,
                "rate": 1.0,
                "contribution": 1.26,
                "seat_weight": 0.4,
            },
            {
                "hotkey": "5Hot2",
                "body_sha256": "b" * 64,
                "rate": 0.25,
                "contribution": 0.74,
                "seat_weight": 0.1,
            },
        ],
        "crown": {"hotkey": "5Hot1", "body_sha256": "a" * 64, "rate": 1.0, "weight": 0.2},
        "category_scores": {
            "reasoning": 61.234,
            "language": 70.06,
            "retrieval": 44.96,
            "tools": 52.0,
            "arts": 39.449,
        },
        "family_scores": {
            SECRET_FAMILIES[0]: {"skill": 0.4, "label": SECRET_LABELS[0]},
            SECRET_FAMILIES[1]: {"skill": 0.7, "label": SECRET_LABELS[1]},
        },
        "predictions": {
            "soup": [
                {
                    "id": SECRET_IDS[0],
                    "family": SECRET_FAMILIES[0],
                    "label": SECRET_LABELS[0],
                    "question": SECRET_TEXT[0],
                    "state": SECRET_TEXT[1],
                    "p": 0.8,
                },
                {"id": SECRET_IDS[1], "family": SECRET_FAMILIES[1], "label": SECRET_LABELS[1]},
            ],
            "a" * 64: [{"id": SECRET_IDS[2], "p": 0.3}],
        },
        "soup_sha256": "c" * 64,
    }


class SoupPublishTest(unittest.TestCase):
    def test_no_private_strings_leak(self):
        public = soup_publish.public_record(private_round())
        text = json.dumps(public)
        for secret in SECRET_IDS + SECRET_LABELS + SECRET_FAMILIES + SECRET_TEXT:
            self.assertNotIn(secret, text)
        forbidden = {
            "family",
            "family_scores",
            "label",
            "question",
            "state",
            "subset_ids",
            "predictions",
        }
        self.assertFalse(forbidden & set(public))
        for row in public["members"] + [public["crown"]]:
            self.assertFalse(forbidden & set(row))

    def test_public_fields_and_rounding(self):
        public = soup_publish.public_record(private_round())
        self.assertEqual(public["round_id"], "mini-2026-11-03")
        self.assertEqual(public["pool_id"], "mini")
        self.assertEqual(public["seed"], "0xabc123")
        self.assertEqual(public["soup_sha256"], "c" * 64)
        self.assertEqual(public["crown"]["hotkey"], "5Hot1")
        self.assertEqual(
            public["members"],
            [
                {
                    "hotkey": "5Hot1",
                    "body_sha256": "a" * 64,
                    "rate": 1.0,
                    "contribution": 1.5,
                    "seat_weight": 0.4,
                },
                {
                    "hotkey": "5Hot2",
                    "body_sha256": "b" * 64,
                    "rate": 0.25,
                    "contribution": 0.5,
                    "seat_weight": 0.1,
                },
            ],
        )
        self.assertEqual(
            public["category_scores"],
            {"reasoning": 61.2, "language": 70.1, "retrieval": 45.0, "tools": 52.0, "arts": 39.4},
        )
        self.assertEqual(
            soup_publish.public_record({**private_round(), "crown": None})["crown"], None
        )

    def test_half_point_rounds_up(self):
        record = private_round()
        record["members"][0]["contribution"] = 0.25
        record["members"][1]["contribution"] = 0.75
        public = soup_publish.public_record(record)
        self.assertEqual([m["contribution"] for m in public["members"]], [0.5, 1.0])

    def test_unknown_category_rejected(self):
        record = private_round()
        record["category_scores"][SECRET_FAMILIES[0]] = 1.0
        with self.assertRaises(ValueError):
            soup_publish.public_record(record)

    def test_commitments_reproducible(self):
        record = private_round()
        public = soup_publish.public_record(record)
        self.assertEqual(
            public["subset_sha256"],
            hashlib.sha256(
                json.dumps(sorted(SECRET_IDS), separators=(",", ":")).encode()
            ).hexdigest(),
        )
        self.assertEqual(
            public["predictions_sha256"],
            hashlib.sha256(soup_publish.canonical_json(record["predictions"])).hexdigest(),
        )
        shuffled = copy.deepcopy(record)
        shuffled["subset_ids"] = list(SECRET_IDS)
        shuffled["predictions"] = dict(reversed(list(record["predictions"].items())))
        self.assertEqual(soup_publish.public_record(shuffled), public)
        soup_publish.verify_public(public, record)

    def test_canonical_json(self):
        self.assertEqual(
            soup_publish.canonical_json({"b": [1, 2], "a": "x"}), b'{"a":"x","b":[1,2]}'
        )
        with self.assertRaises(ValueError):
            soup_publish.canonical_json({"a": float("nan")})

    def test_tampered_private_record_fails(self):
        public = soup_publish.public_record(private_round())
        tampers = [
            lambda r: r["subset_ids"].append("zz-extra"),
            lambda r: r["predictions"]["soup"][0].update(p=0.81),
            lambda r: r.update(soup_sha256="d" * 64),
            lambda r: r["members"][0].update(seat_weight=0.5),
        ]
        for tamper in tampers:
            record = private_round()
            tamper(record)
            with self.assertRaises(ValueError):
                soup_publish.verify_public(public, record)

    def test_tampered_public_record_fails(self):
        public = soup_publish.public_record(private_round())
        public["predictions_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            soup_publish.verify_public(public, private_round())


if __name__ == "__main__":
    unittest.main()
