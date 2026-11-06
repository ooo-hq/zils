import copy
import hashlib
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

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
        self.house = self.enterContext(patch("zils.soup_rates.house", return_value=FEED_HOUSE))

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
        empty = {
            "round_id": None,
            "max_new_per_round": 6,
            "graded": [],
            "rejected": [],
            "claims": [],
        }
        self.assertEqual(
            feed["pools"],
            {
                "d3-mini": {"bodies": {}, "rounds": [], "intake": empty},
                "d3-lite": {"bodies": {}, "rounds": [], "intake": empty},
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

    def test_an_unreadable_house_list_falls_back_to_the_pool_state_names(self):
        self.write_pool({HOUSE_SHA: {"repo": "open/pinned", "house": True, "name": "house-pinned"}})
        self.write_round("r1", public_round("r1", [HOUSE_SHA]), 1_800_000_000)
        self.house.side_effect = ValueError("unreadable")
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
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


class SoupLiveTest(unittest.TestCase):
    """GET /soup's `live` block: the round in progress, its stage and each claim's state, no reasons."""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = self.directory / "base-pools" / "d3-mini" / "soup"
        self.enterContext(patch("zils.soup_rates.house", return_value=FEED_HOUSE))
        self.round = self.state / "rounds" / "d3-mini-2026-11-05"
        self.round.mkdir(parents=True)
        self.shas = {uid: f"{uid:x}" * 64 for uid in range(1, 7)}
        self.shas = {uid: sha[:64] for uid, sha in self.shas.items()}
        commitments = {
            str(uid): {"block": 100 + uid, "data": f"zilsb2:{sha}:{'0' * 40}:someone/body-{uid}"}
            for uid, sha in self.shas.items()
        }
        (self.state / f"request-{self.round.name}.json").write_text(json.dumps({
            "round_id": self.round.name, "snapshot": {"members": {}, "commitments": commitments},
        }))  # fmt: skip
        (self.state / "pool.json").write_text(json.dumps({"bodies": {self.shas[5]: {"uid": 5}}}))
        self.started = datetime.fromisoformat("2026-11-05T01:22:10+00:00").timestamp()

    def job(self, pid=None):
        from zils import soup_pool

        pid = os.getpid() if pid is None else pid
        (self.state / "job.json").write_text(json.dumps({
            "round_id": self.round.name, "attempts": 1, "started": self.started, "pid": pid,
            "pid_started": soup_pool._started(pid),
        }))  # fmt: skip

    def progress(self):
        (self.round / "intake-progress.json").write_text(json.dumps({
            "done": ["1", "2"],
            "new": {self.shas[1]: {"uid": 1, "hotkey": "h1"}},
            "refused": {
                "2": "copy: explained by earlier; zz-private-reason",
                "3": "the round's new-body cap is reached; retried next round",
            },
            "rejected": {}, "failed": {}, "explained": {}, "backoff": {}, "bytes": 1, "claims": 2,
        }))  # fmt: skip

    def live(self):
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory)
        return feed["pools"]["d3-mini"].get("live")

    def test_a_running_intake_names_each_claims_state(self):
        self.job()
        self.progress()
        live = self.live()
        self.assertEqual(live["round_id"], "d3-mini-2026-11-05")
        self.assertEqual((live["stage"], live["running"], live["attempt"]), ("intake", True, 1))
        self.assertEqual(live["started_at"], "2026-11-05T01:22:10+00:00")
        self.assertEqual(live["deadline_at"], "2026-11-05T07:22:10+00:00")
        states = {c["uid"]: c["state"] for c in live["claims"]}
        self.assertEqual(states, {1: "admitted", 2: "refused", 3: "deferred", 4: "checking",
                                  5: "in_pool", 6: "waiting"})  # fmt: skip
        self.assertEqual([c["uid"] for c in live["claims"]], [1, 2, 3, 4, 5, 6])

    def test_reasons_never_leak(self):
        self.job()
        self.progress()
        text = json.dumps(soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory))
        self.assertNotIn("zz-private-reason", text)
        self.assertNotIn("explained", text)

    def test_before_intake_writes_progress_it_is_starting(self):
        self.job()
        live = self.live()
        self.assertEqual(live["stage"], "starting")
        self.assertEqual(
            {c["state"] for c in live["claims"] if c["uid"] != 5}, {"waiting", "checking"}
        )

    def test_scoring_once_the_decision_items_exist(self):
        self.job()
        self.progress()
        (self.round / "decision-items.json").write_text("[]")
        live = self.live()
        self.assertEqual(live["stage"], "scoring")
        self.assertNotIn("checking", {c["state"] for c in live["claims"]})

    def test_a_dead_process_is_not_running(self):
        self.job(pid=2**22 + 12345)
        self.assertFalse(self.live()["running"])

    def test_a_published_round_has_no_live_block(self):
        self.job()
        self.progress()
        (self.round / "public.json").write_text(json.dumps(public_round(self.round.name, [])))
        self.assertIsNone(self.live())

    def test_no_job_has_no_live_block(self):
        self.assertIsNone(self.live())


class SoupIntakeTest(unittest.TestCase):
    """GET /soup's `intake` block: the newest finished intake, by uid and body hash, so a page can
    rank waiting bodies against chain. Only reasons that say "retried next round" are published."""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = self.directory / "base-pools" / "d3-mini" / "soup"
        self.enterContext(patch("zils.soup_rates.house", return_value=FEED_HOUSE))
        self.shas = {uid: (f"{uid:x}" * 64)[:64] for uid in range(1, 6)}
        (self.state / "rounds").mkdir(parents=True)
        (self.state / "pool.json").write_text(json.dumps({
            "bodies": {self.shas[1]: {"uid": 1, "house": False}, "f" * 64: {"house": True}},
            "rejected": {self.shas[4]: "zz-private-weights"},
        }))  # fmt: skip

    def finished(self, round_id, refused, uids=range(1, 6)):
        commitments = {
            str(uid): {"block": 100 + uid, "data": f"zilsb2:{self.shas[uid]}:{'0' * 40}:a/b-{uid}"}
            for uid in uids
        }
        (self.state / f"request-{round_id}.json").write_text(json.dumps({
            "round_id": round_id, "snapshot": {"members": {}, "commitments": commitments},
        }))  # fmt: skip
        folder = self.state / "rounds" / round_id
        folder.mkdir()
        (folder / "intake.json").write_text(json.dumps({"refused": refused, "new": {}}))

    def intake(self, **entry):
        feed = soup_publish.feed(feed_config({"d3-mini": soup_entry(**entry)}), self.directory)
        return feed["pools"]["d3-mini"].get("intake")

    def test_deferred_claims_keep_their_reason_and_hash(self):
        self.finished("d3-mini-2026-11-06", {
            "2": "the round's new-body cap is reached; retried next round",
            "3": "copy: explained by earlier; zz-private-reason",
        })  # fmt: skip
        intake = self.intake(max_new_per_round=12)
        self.assertEqual(intake["round_id"], "d3-mini-2026-11-06")
        self.assertEqual(intake["max_new_per_round"], 12)
        self.assertEqual(intake["claims"], [
            {"uid": 2, "sha256": self.shas[2], "state": "deferred",
             "reason": "the round's new-body cap is reached; retried next round"},
            {"uid": 3, "sha256": self.shas[3], "state": "refused", "reason": None},
        ])  # fmt: skip

    def test_graded_and_rejected_hashes_without_house_bodies(self):
        self.finished("d3-mini-2026-11-06", {})
        intake = self.intake()
        self.assertEqual(intake["graded"], [self.shas[1]])
        self.assertEqual(intake["rejected"], [self.shas[4]])
        self.assertEqual(intake["max_new_per_round"], 6)  # the pool default

    def test_private_reasons_never_leak(self):
        self.finished("d3-mini-2026-11-06", {"3": "copy: explained by earlier; zz-private-reason"})
        text = json.dumps(soup_publish.feed(feed_config({"d3-mini": soup_entry()}), self.directory))
        self.assertNotIn("zz-private", text)

    def test_the_newest_finished_intake_wins(self):
        self.finished(
            "d3-mini-2026-11-05", {"2": "the round's claims cap is reached; retried next round"}
        )
        self.finished("d3-mini-2026-11-06", {}, uids=(1, 2))
        (self.state / "rounds" / "d3-mini-2026-11-07").mkdir()  # in progress: no intake.json yet
        intake = self.intake()
        self.assertEqual((intake["round_id"], intake["claims"]), ("d3-mini-2026-11-06", []))

    def test_a_claim_missing_from_the_snapshot_is_skipped(self):
        self.finished(
            "d3-mini-2026-11-06", {"9": "the round's new-body cap is reached; retried next round"}
        )
        self.assertEqual(self.intake()["claims"], [])

    def test_before_any_intake_there_is_still_a_block(self):
        intake = self.intake()
        self.assertEqual((intake["round_id"], intake["claims"]), (None, []))
        self.assertEqual(intake["graded"], [self.shas[1]])
