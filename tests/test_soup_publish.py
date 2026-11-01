import copy
import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from zils import models, soup_publish

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

    def test_pooled_decision_rounds_are_published_as_round_ids_and_a_count(self):
        private = private_round()
        private["decision"] = {
            "requested": 3,
            "rounds": ["mini-2026-11-01", "mini-2026-11-02", "mini-2026-11-03"],
            "items": 3,
            "dropped_items": 0,
            "skipped": {},
            "fallback": None,
            "estimates": {"1": 10.0},
        }
        private["decision_ids"] = sorted(SECRET_IDS)
        public = soup_publish.public_record(private)
        self.assertEqual(
            public["decision"],
            {"rounds": ["mini-2026-11-01", "mini-2026-11-02", "mini-2026-11-03"], "items": 3},
        )
        for secret in SECRET_IDS:
            self.assertNotIn(secret, json.dumps(public))
        self.assertNotIn("decision", soup_publish.public_record(private_round()))


HOUSE_SHA = "e" * 64
FEED_HOUSE = [
    {"name": "house-pinned", "repo": "open/pinned", "revision": "1" * 40, "body_sha256": HOUSE_SHA},
    {"name": "house-unpinned", "repo": "open/unpinned", "revision": "2" * 40, "body_sha256": None},
]


def soup_entry(**changes):
    return {
        "type": "soup",
        "model": models.D3,
        "enabled": True,
        "share": 0.25,
        "house": FEED_HOUSE,
        "schedule": {"deadline_utc": "00:00"},
        "sketch_seed": "5e" * 32,
        **changes,
    }


def feed_config(pools):
    return {"base_pools": {"share": 1.0, "burn_uid": 0, "pools": pools}}


def public_round(round_id, members, crown=None):
    private = private_round()
    private["round_id"] = round_id
    private["pool_id"] = "d3-mini"
    private["members"] = [
        {
            "hotkey": f"5H{i}",
            "body_sha256": sha,
            "rate": 1.0,
            "contribution": 1.0,
            "seat_weight": 0.1,
        }
        for i, sha in enumerate(members)
    ]
    private["crown"] = None if crown is None else {
        "hotkey": "5Hc", "body_sha256": crown, "rate": 1.0, "weight": 0.05,
    }  # fmt: skip
    return soup_publish.public_record(private)


class SoupFeedTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = self.directory / "base-pools" / "d3-mini" / "soup"

    def write_round(self, round_id, record, mtime, private=None):
        folder = self.state / "rounds" / round_id
        folder.mkdir(parents=True)
        (folder / "public.json").write_text(
            record if isinstance(record, str) else json.dumps(record)
        )
        os.utime(folder / "public.json", (mtime, mtime))
        if private is not None:
            (folder / "private.json").write_text(json.dumps(private))

    def write_pool(self, bodies, house=None):
        self.state.mkdir(parents=True, exist_ok=True)
        state = {"bodies": bodies, "rejected": {}, "members": [], "soup": None}
        if house is not None:
            state["house"] = house
        (self.state / "pool.json").write_text(json.dumps(state))

    def test_every_enabled_soup_pool_appears_even_without_rounds(self):
        config = feed_config({
            "d3-mini": soup_entry(),
            "d3-lite": soup_entry(model=models.D3_POOL_MODELS["d3-lite"]),
            "d3-flash": soup_entry(model=models.D3_POOL_MODELS["d3-flash"], enabled=False),
        })  # fmt: skip
        feed = soup_publish.feed(config, self.directory)
        self.assertEqual(feed["kind"], "soup")
        self.assertEqual(
            feed["pools"],
            {
                "d3-mini": {"bodies": {}, "rounds": []},
                "d3-lite": {"bodies": {}, "rounds": []},
            },
        )

    def test_no_base_pools_is_an_empty_feed(self):
        self.assertEqual(soup_publish.feed({}, self.directory), {"kind": "soup", "pools": {}})

    def test_rounds_are_newest_first_with_finished_at_and_capped_at_sixty(self):
        base = datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp()
        for day in range(65):
            round_id = f"d3-mini-r{day:03d}"
            self.write_round(round_id, public_round(round_id, ["a" * 64]), base + day * 86400)
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        rounds = feed["pools"]["d3-mini"]["rounds"]
        self.assertEqual(len(rounds), 60)
        self.assertEqual(rounds[0]["round_id"], "d3-mini-r064")
        self.assertEqual(rounds[-1]["round_id"], "d3-mini-r005")
        self.assertEqual(rounds[0]["finished_at"], "2026-03-06T00:00:00+00:00")
        expected = public_round("d3-mini-r064", ["a" * 64])
        self.assertEqual({k: v for k, v in rounds[0].items() if k != "finished_at"}, expected)

    def test_bodies_name_house_and_repo_and_never_invent(self):
        resolved = "d" * 64  # the unpinned house entry, resolved by the validator's download
        miner, unknown, crown = "a" * 64, "b" * 64, "c" * 64
        self.write_pool(
            {
                miner: {
                    "hotkey": "5H1",
                    "repo": "miner/body",
                    "revision": "3" * 40,
                    "house": False,
                },
                resolved: {
                    "hotkey": None,
                    "repo": "open/unpinned",
                    "house": True,
                    "name": "house-unpinned",
                },
                crown: {"hotkey": "5Hc", "repo": "crown/body", "house": False},
            },
            house={"open/unpinned@" + "2" * 40: resolved},
        )
        self.write_round(
            "d3-mini-2026-11-03",
            public_round("d3-mini-2026-11-03", [HOUSE_SHA, resolved, miner, unknown], crown),
            1_800_000_000,
        )
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        self.assertEqual(
            feed["pools"]["d3-mini"]["bodies"],
            {
                HOUSE_SHA: {"house": "house-pinned", "repo": "open/pinned"},
                resolved: {"house": "house-unpinned", "repo": "open/unpinned"},
                miner: {"house": None, "repo": "miner/body"},
                unknown: {"house": None, "repo": None},
                crown: {"house": None, "repo": "crown/body"},
            },
        )

    def test_bodies_only_cover_published_rounds(self):
        self.write_pool({"9" * 64: {"repo": "not/seated", "house": False}})
        self.write_round("r1", public_round("r1", ["a" * 64]), 1_800_000_000)
        bodies = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        self.assertEqual(
            bodies["pools"]["d3-mini"]["bodies"], {"a" * 64: {"house": None, "repo": None}}
        )

    def test_malformed_rounds_are_skipped_and_private_records_never_read(self):
        secret = private_round()
        self.write_round(
            "r-good", public_round("r-good", ["a" * 64]), 1_800_000_000, private=secret
        )
        self.write_round("r-bad", "{not json", 1_800_000_100, private=secret)
        self.write_round("r-list", "[1, 2]", 1_800_000_200)
        (self.state / "rounds" / "r-empty").mkdir()
        (self.state / "pool.json").write_text("{broken")
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        rounds = feed["pools"]["d3-mini"]["rounds"]
        self.assertEqual([r["round_id"] for r in rounds], ["r-good"])
        text = json.dumps(feed)
        for value in [*SECRET_IDS, *SECRET_LABELS, *SECRET_FAMILIES, *SECRET_TEXT]:
            self.assertNotIn(value, text)

    def test_an_unreadable_house_path_leaves_house_names_null(self):
        self.write_pool({HOUSE_SHA: {"repo": "open/pinned", "house": True, "name": "house-pinned"}})
        self.write_round("r1", public_round("r1", [HOUSE_SHA]), 1_800_000_000)
        config = feed_config({"d3-mini": soup_entry(house="missing-house.json")})
        feed = soup_publish.feed(config, self.directory)
        self.assertEqual(
            feed["pools"]["d3-mini"]["bodies"],
            {HOUSE_SHA: {"house": "house-pinned", "repo": "open/pinned"}},
        )

    def test_the_feed_is_json_serialisable(self):
        self.write_round("r1", public_round("r1", ["a" * 64], "c" * 64), 1_800_000_000)
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        self.assertEqual(json.loads(json.dumps(feed, allow_nan=False)), feed)


if __name__ == "__main__":
    unittest.main()
