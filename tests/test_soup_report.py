"""A miner's private soup round report: where its body stopped, why, and how far from each bar."""

import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bittensor_wallet import Keypair

from tests.test_soup_publish import feed_config, soup_entry
from zils import miner, models, soup_report
from zils.runtime import signed, verified

CATEGORIES = ("reasoning", "language", "retrieval", "tools", "arts")
SEATED, THIN, GATED, TRIED, LATE, SCREENED, TWIN, BROKEN, BAD, PAST = (c * 64 for c in "abcdef0123")
SECRETS = ("zz-secret-item", "zz-family-tax-law", "https://secret.example/weights", "/var/lib/zils")


def cats(value):
    return dict.fromkeys(CATEGORIES, value)


def private_round():
    """One round with a body at every stage. The soup is SEATED + THIN + GATED and scores 64."""
    soup = sorted([SEATED, THIN, GATED])
    return {
        "round_id": "d3-mini-2026-11-10",
        "pool_id": "d3-mini",
        "members": [
            {"hotkey": "5Seated", "body_sha256": SEATED, "rate": 1.0, "contribution": 3.26,
             "seat_weight": 0.16},
            {"hotkey": "5Thin", "body_sha256": THIN, "rate": 1.0, "contribution": 0.74,
             "seat_weight": 0.0},
            {"hotkey": "5Gated", "body_sha256": GATED, "rate": 1.0, "contribution": 2.0,
             "seat_weight": 0.0},
        ],
        "pool": [SEATED, THIN, GATED, TRIED, LATE],
        "solo": {SEATED: 62.0, THIN: 60.1, GATED: 61.0, TRIED: 59.3, LATE: 50.0},
        "solo_categories": {
            SEATED: cats(62.0),
            TRIED: {**cats(60.0), "tools": 51.26},
        },
        "soup_score": 64.0,
        "soup_categories": cats(64.24),
        "screen": {TRIED: 0.61, LATE: 0.55, SCREENED: 0.41},
        "rates": {SEATED: 1.0, THIN: 1.0, GATED: 1.0, TRIED: 1.0, LATE: 0.25},
        "gated": {GATED: "does not beat its base soup by 1.2 points"},
        "trace": [
            {"try": sorted([SEATED, GATED]), "score": 63.0, "best_before": 62.0,
             "accepted": True},
            {"try": soup, "score": 64.0, "best_before": 63.0, "accepted": True},
            {"try": sorted([*soup, TRIED]), "score": 64.3, "best_before": 64.0,
             "accepted": False},
        ],
        "soup_bar": 60.0,
        "rejected": {BAD: "body tensor language_model.x shape [2] differs from the chassis"},
        "grade_failures": {BROKEN: f"RuntimeError: cannot read {SECRETS[3]}/bodies"},
        "twins": {
            TWIN: {"key": "5Twin:miner/twin@" + "b" * 40, "partner": SEATED,
                   "reason": f"behavioural twin of pool body {SEATED}; retried when the claim"
                   " changes or that body leaves the pool"},
        },
        "refused": {
            "7": "copy: explained by earlier; retried when the claim changes or one of the "
            "explaining bodies leaves the pool",
            "8": "the round's new-body cap is reached; retried next round",
            "9": f"download failed: HTTPError: {SECRETS[2]}",
            "10": "body committed first by another hotkey",
        },
        "failed": {},
        "family_scores": {SECRETS[1]: 40.0},
        "subset_ids": [SECRETS[0]],
        "predictions": [{"id": SECRETS[0]}],
    }  # fmt: skip


OWNERS = {
    SEATED: "5Seated", THIN: "5Thin", GATED: "5Gated", TRIED: "5Tried", LATE: "5Late",
    SCREENED: "5Screened", TWIN: "5Twin", BROKEN: "5Broken", PAST: "5Past",
}  # fmt: skip
CLAIMS = {
    "1": {"hotkey": "5Seated", "sha256": SEATED},
    "6": {"hotkey": "5Bad", "sha256": BAD},
    "7": {"hotkey": "5Copy", "sha256": "7" * 64},
    "8": {"hotkey": "5Waiting", "sha256": "8" * 64},
    "9": {"hotkey": "5Slow", "sha256": "9" * 64},
    "10": {"hotkey": "5Thief", "sha256": SEATED},
}


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.reports = soup_report.build(private_round(), CLAIMS, OWNERS)

    def one(self, hotkey):
        (entry,) = self.reports[hotkey]
        return entry

    def test_a_paid_seat(self):
        entry = self.one("5Seated")
        self.assertEqual((entry["stage"], entry["reason"]), ("seated", None))
        self.assertEqual(entry["body_sha256"], SEATED)
        self.assertEqual(entry["seat"], {"contribution": 3.5, "floor": 0.8})
        self.assertEqual(entry["solo"], {"score": 62.0, "soup": 64.0})
        self.assertEqual(entry["rate"], 1.0)

    def test_a_seat_below_the_floor_is_unpaid_and_never_rounds_up_to_the_floor(self):
        entry = self.one("5Thin")
        self.assertEqual((entry["stage"], entry["reason"]), ("seated_unpaid", "below_floor"))
        self.assertEqual(entry["seat"], {"contribution": 0.5, "floor": 0.8})

    def test_a_gated_fine_tune_names_the_kind_of_base_and_the_margin(self):
        entry = self.one("5Gated")
        self.assertEqual((entry["stage"], entry["reason"]), ("seated_unpaid", "fine_tune_gate"))
        self.assertEqual(entry["gate"], {"base": "soup", "needed": 1.2})

    def test_a_body_tried_in_the_soup_shows_its_gain_short_of_the_margin(self):
        entry = self.one("5Tried")
        self.assertEqual((entry["stage"], entry["reason"]), ("not_selected", "no_soup_gain"))
        # 0.3 would round to 0.5, the margin itself; a missed bar is always shown as missed.
        self.assertEqual(entry["trial"], {"gain": 0.0, "needed": 0.5})
        self.assertEqual(entry["solo"], {"score": 59.5, "soup": 64.0})
        self.assertEqual(entry["screen"], {"score": 61.0, "needed": 55.0})

    def test_the_tried_body_is_found_even_when_the_first_two_tie_alone(self):
        private = private_round()
        private["members"] = private["members"][1:2]  # THIN alone is the soup
        private["solo"] = {SEATED: 60.0, THIN: 60.0}
        private["trace"] = [
            {"try": sorted([SEATED, THIN]), "score": 60.2, "best_before": 60.0, "accepted": False}
        ]
        entry = soup_report.build(private, CLAIMS, OWNERS)["5Seated"][0]
        self.assertEqual((entry["stage"], entry["reason"]), ("not_selected", "no_soup_gain"))

    def test_a_pool_body_greedy_never_reached_says_so_instead_of_a_gain(self):
        entry = self.one("5Late")
        self.assertEqual((entry["stage"], entry["reason"]), ("not_selected", "soup_full"))
        self.assertNotIn("trial", entry)
        self.assertEqual(entry["rate"], 0.25)

    def test_a_soup_that_did_not_beat_the_current_one_is_named(self):
        private = private_round()
        private["members"] = private["members"][:1]  # the current soup, SEATED alone, was kept
        private["trace"].append(
            {"try": sorted([SEATED, THIN, GATED]), "score": 64.0, "best_before": 63.8,
             "accepted": False, "versus": "current"}
        )  # fmt: skip
        entry = soup_report.build(private, CLAIMS, OWNERS)["5Thin"][0]
        self.assertEqual((entry["stage"], entry["reason"]), ("not_selected", "current_soup_kept"))
        self.assertEqual(entry["trial"], {"gain": 0.0, "needed": 0.5})

    def test_a_screened_out_body_shows_the_lowest_screen_score_that_entered_the_pool(self):
        entry = self.one("5Screened")
        self.assertEqual((entry["stage"], entry["reason"]), ("screened_out", "below_screen"))
        self.assertEqual(entry["screen"], {"score": 41.0, "needed": 55.0})
        self.assertNotIn("solo", entry)

    def test_an_earlier_body_left_out_of_a_full_pool(self):
        entry = self.one("5Past")
        self.assertEqual((entry["stage"], entry["reason"]), ("screened_out", "pool_full"))

    def test_refusals_are_codes_and_deferrals_are_told_apart(self):
        expected = {
            "5Twin": ("refused", "twin"),
            "5Bad": ("refused", "bad_format"),
            "5Copy": ("refused", "near_copy"),
            "5Thief": ("refused", "committed_first_by_another"),
            "5Waiting": ("deferred", "round_limit"),
            "5Slow": ("deferred", "download_failed"),
            "5Broken": ("deferred", "grade_failed"),
        }
        for hotkey, pair in expected.items():
            entry = self.one(hotkey)
            self.assertEqual((entry["stage"], entry["reason"]), pair, hotkey)
            self.assertEqual(set(entry), {"body_sha256", "stage", "reason"}, hotkey)

    def test_a_refused_copy_of_anothers_hash_gets_none_of_that_bodys_numbers(self):
        self.assertEqual(self.one("5Thief")["body_sha256"], SEATED)
        self.assertNotIn("solo", self.one("5Thief"))

    def test_category_scores_sit_next_to_the_soups_at_half_points(self):
        categories = self.one("5Tried")["categories"]
        self.assertEqual(set(categories), set(CATEGORIES))
        self.assertEqual(categories["tools"], {"score": 51.5, "soup": 64.0})
        self.assertEqual(categories["arts"], {"score": 60.0, "soup": 64.0})
        self.assertNotIn("categories", self.one("5Thin"))  # none recorded for it

    def test_every_reason_is_from_the_fixed_list(self):
        for entries in self.reports.values():
            for entry in entries:
                self.assertIn(entry["reason"], (None, *soup_report.REASONS))

    def test_reason_codes_for_body_checks(self):
        cases = {
            "body repo needs a small zils-owner.json naming the committing hotkey": "owner_file",
            "zils-owner.json names hotkey 5A, not the committing hotkey 5B": "owner_file",
            "README.md front matter has no license": "license",
            "body license 'x' is not one of apache-2.0": "license",
            "downloaded body hash differs from the commitment": "hash_mismatch",
            "body change from the base 0.2000 exceeds 0.15": "change_cap",
            "body tensor a.b is not finite": "not_finite",
            "body safetensors total 9 bytes, over the 8 byte limit": "too_large",
            "commitment is not zilsb2:<sha256>:<revision>:<repo>": "bad_commitment",
            "copy: cosine": "near_copy",
            "copy: house re-upload": "house_reupload",
            "copy: no change from the base": "no_change",
            "body rejected in an earlier round": "rejected_earlier",
            "commitment failed in an earlier round": "failed_earlier",
            "the hotkey's last claim failed; it backs off until round 9": "backing_off",
            "body is missing chassis tensor x": "bad_format",
            "KeyError: 'surprise'": "check_failed",
        }
        for text, code in cases.items():
            self.assertEqual(soup_report.reason_code(text), code, text)

    def test_nothing_private_reaches_a_report(self):
        text = json.dumps(self.reports)
        for secret in SECRETS:
            self.assertNotIn(secret, text)
        self.assertNotIn("explained by earlier", text)  # raw reasons never pass through
        twin = json.dumps(self.reports["5Twin"])
        self.assertNotIn(SEATED, twin)  # the body it matched stays private

    def test_house_bodies_have_no_report(self):
        reports = soup_report.build(private_round(), CLAIMS, {**OWNERS, LATE: None})
        self.assertNotIn(None, reports)

    def test_claims_come_from_the_round_snapshot(self):
        snapshot = {
            "members": {"1": "5A", "2": "5B"},
            "commitments": {
                "1": {"block": 5, "data": f"zilsb2:{SEATED}:{'b' * 40}:miner/a"},
                "2": {"block": 6, "data": "zilsb2:not-a-hash"},
                "3": {"block": 7, "data": f"zilsb2:{THIN}:{'b' * 40}:miner/c"},
            },
        }
        self.assertEqual(
            soup_report.claims(snapshot),
            {"1": {"hotkey": "5A", "sha256": SEATED}, "2": {"hotkey": "5B", "sha256": None}},
        )


class AnswerTest(unittest.TestCase):
    """The validator's reply to a report request: only the signer's own report, signed requests only."""

    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.validator = Keypair.create_from_uri("//Validator")
        self.alice = Keypair.create_from_uri("//Alice")
        self.bob = Keypair.create_from_uri("//Bob")
        self.config = {
            **feed_config({
                "d3-mini": soup_entry(),
                "d3-lite": soup_entry(model=models.D3_POOL_MODELS["d3-lite"]),
            }),
            "validator_hotkey": self.validator.ss58_address,
        }  # fmt: skip
        state = self.directory / "base-pools" / "d3-mini" / "soup"
        round_dir = state / "rounds" / "d3-mini-2026-11-10"
        round_dir.mkdir(parents=True)
        (state / "pool.json").write_text(json.dumps({"round_id": "d3-mini-2026-11-10"}))
        self.reports = {
            self.alice.ss58_address: [{"body_sha256": SEATED, "stage": "seated", "reason": None}],
            self.bob.ss58_address: [{"body_sha256": TWIN, "stage": "refused", "reason": "twin"}],
        }
        (round_dir / "reports.json").write_text(json.dumps(self.reports))
        self.seen = {}

    def ask(self, message, now=None):
        return soup_report.answer(self.config, self.directory, message, self.seen, now)

    def request(self, key, **changes):
        message = soup_report.sign(key, self.validator.ss58_address)
        if changes:
            message = signed({**message["payload"], **changes}, key)
        return message

    def test_a_signed_request_returns_only_the_signers_report(self):
        code, payload = self.ask(self.request(self.alice))
        self.assertEqual(code, 200)
        self.assertEqual(payload["hotkey"], self.alice.ss58_address)
        self.assertEqual(
            payload["pools"]["d3-mini"],
            {"round_id": "d3-mini-2026-11-10", "bodies": self.reports[self.alice.ss58_address]},
        )
        self.assertNotIn(TWIN, json.dumps(payload))
        self.assertNotIn(self.bob.ss58_address, json.dumps(payload))

    def test_a_pool_with_no_finished_round_and_a_hotkey_with_no_body_are_empty(self):
        code, payload = self.ask(self.request(self.alice))
        self.assertEqual(payload["pools"]["d3-lite"], {"round_id": None, "bodies": []})
        code, payload = self.ask(self.request(Keypair.create_from_uri("//Carol")))
        self.assertEqual(code, 200)
        self.assertEqual(payload["pools"]["d3-mini"]["bodies"], [])

    def test_a_request_signed_by_another_key_is_refused(self):
        forged = self.request(self.bob)
        forged["payload"]["hotkey"] = self.alice.ss58_address
        self.assertEqual(self.ask(forged)[0], 401)

    def test_unsigned_and_malformed_requests_are_refused(self):
        payload = self.request(self.alice)["payload"]
        for message in ({"payload": payload}, {"payload": payload, "signature": "00" * 64}, {}, []):
            code, reply = self.ask(message)
            self.assertEqual(code, 401, message)
            self.assertNotIn("pools", reply)

    def test_a_stale_or_future_request_is_refused(self):
        message = self.request(self.alice)
        stamp = message["payload"]["timestamp"]
        self.assertEqual(self.ask(message, now=stamp + 121)[0], 401)
        self.assertEqual(self.ask(message, now=stamp - 121)[0], 401)
        self.assertEqual(self.ask(message, now=stamp + 119)[0], 200)

    def test_a_replayed_request_is_refused(self):
        message = self.request(self.alice)
        self.assertEqual(self.ask(message)[0], 200)
        self.assertEqual(self.ask(message)[0], 401)
        self.assertEqual(self.ask(self.request(self.alice))[0], 200)  # a fresh one still works

    def test_old_nonces_are_forgotten(self):
        message = self.request(self.alice)
        stamp = message["payload"]["timestamp"]
        self.ask(message, now=stamp)
        self.ask(self.request(self.bob), now=stamp + 500)
        self.assertNotIn(message["payload"]["nonce"], self.seen)

    def test_a_request_made_for_another_validator_is_refused(self):
        message = self.request(self.alice, validator=self.bob.ss58_address)
        self.assertEqual(self.ask(message)[0], 401)

    def test_another_kind_of_signed_message_is_refused(self):
        self.assertEqual(self.ask(self.request(self.alice, kind="ack"))[0], 401)


class CommandTest(unittest.TestCase):
    """`zils-miner report`: signs with the hotkey, checks the validator's signature, prints the report."""

    def setUp(self):
        self.validator = Keypair.create_from_uri("//Validator")
        self.alice = Keypair.create_from_uri("//Alice")
        self.reply = {
            "kind": "soup-report",
            "hotkey": self.alice.ss58_address,
            "pools": {
                "d3-mini": {
                    "round_id": "d3-mini-2026-11-10",
                    "bodies": soup_report.build(private_round(), CLAIMS, OWNERS)["5Tried"],
                },
                "d3-lite": {"round_id": None, "bodies": []},
            },
        }

    def reply_for(self, hotkey):
        self.reply["pools"]["d3-mini"]["bodies"] = soup_report.build(
            private_round(), CLAIMS, OWNERS
        )[hotkey]
        return miner.report_text(self.reply)

    def test_a_body_the_soup_did_not_take_reads_as_a_sentence_with_its_gaps(self):
        text = miner.report_text(self.reply)
        self.assertEqual(
            text,
            "\n".join([
                "Zils round report",
                "",
                "d3-mini · round d3-mini-2026-11-10",
                "",
                f"✗ body {TRIED[:12]}  not selected",
                "  Adding your body did not raise the soup's score by the margin needed.",
                "",
                "  soup gain      +0.0   needed +0.5",
                "  alone          59.5   soup 64.0",
                "  screening      61.0   needed 55.0",
                "",
                "  category      yours   soup",
                "  reasoning      60.0   64.0",
                "  language       60.0   64.0",
                "  retrieval      60.0   64.0",
                "  tools          51.5   64.0   ← furthest behind",
                "  arts           60.0   64.0",
                "",
                "  Next: train where the soup is weak, not where it is already strong.",
                '  Codes: not_selected / no_soup_gain (docs/mining.md, "Why a body has no seat")',
            ]),
        )  # fmt: skip

    def test_a_refusal_says_what_is_wrong_and_what_to_do(self):
        text = self.reply_for("5Copy")
        self.assertIn(f"✗ body {'7' * 12}  refused", text)
        self.assertIn("near-copy of an earlier body", text)
        self.assertIn("Next: commit a body with your own training.", text)
        self.assertIn("Codes: refused / near_copy", text)
        self.assertNotIn("category", text)

    def test_a_deferral_is_a_warning_and_a_paid_seat_a_tick(self):
        self.assertIn(f"⚠ body {'8' * 12}  waiting", self.reply_for("5Waiting"))
        seated = self.reply_for("5Seated")
        self.assertIn(f"✓ body {SEATED[:12]}  seated", seated)
        self.assertIn("  contribution    3.5   floor 0.8", seated)
        self.assertNotIn("Next:", seated)
        self.assertNotIn("furthest behind", seated.replace("← furthest behind", "", 1))

    def test_an_unpaid_seat_names_its_base_and_a_scout_rate_is_shown(self):
        gated = self.reply_for("5Gated")
        self.assertIn(f"⚠ body {GATED[:12]}  seated, unpaid", gated)
        self.assertIn("  base           soup   must beat it by 1.2", gated)
        self.assertNotIn("rate", gated)  # the trainer rate, 1.0, is not worth a line
        self.assertIn("  rate           0.25", self.reply_for("5Late"))

    def test_pools_without_a_body_of_this_hotkey_are_left_out(self):
        self.assertNotIn("d3-lite", miner.report_text(self.reply))
        self.reply["pools"]["d3-mini"]["bodies"] = []
        text = miner.report_text(self.reply)
        self.assertNotIn("d3-mini", text)
        self.assertIn("No body of this hotkey was in the newest round of any pool.", text)

    def test_every_reason_has_a_sentence(self):
        self.assertEqual(set(soup_report.EXPLAIN), {None, *soup_report.REASONS})
        for sentence, step in soup_report.EXPLAIN.values():
            self.assertTrue(sentence.endswith("."))
            self.assertTrue(step is None or step.endswith("."))

    def test_a_refused_request_says_what_to_check(self):
        def refuse(config, path, message):
            raise urllib.error.HTTPError(config["validator"] + path, 401, "Unauthorized", {}, None)

        argv = ["report", "--validator", "https://validator.example", "--validator-hotkey",
                self.validator.ss58_address, "--wallet-name", "w", "--wallet-hotkey", "h"]  # fmt: skip
        err = io.StringIO()
        with (
            patch.object(miner, "wallet", return_value=SimpleNamespace(hotkey=self.alice)),
            patch.object(miner, "load_config", return_value={}),
            patch("zils.runtime.request", refuse),
            redirect_stderr(err),
            self.assertRaises(SystemExit),
        ):
            miner.main(argv)
        self.assertIn("the validator refused the request (HTTP 401)", err.getvalue())
        self.assertIn("--validator-hotkey", err.getvalue())
        self.assertIn("clock", err.getvalue())

    def test_the_command_sends_a_request_signed_by_the_hotkey(self):
        sent = {}

        def request(config, path, message):
            sent.update(config=config, path=path, message=message)
            return self.reply

        argv = ["report", "--validator", "https://validator.example/", "--validator-hotkey",
                self.validator.ss58_address, "--wallet-name", "w", "--wallet-hotkey", "h"]  # fmt: skip
        out = io.StringIO()
        with (
            patch.object(miner, "wallet", return_value=SimpleNamespace(hotkey=self.alice)),
            patch.object(miner, "load_config", return_value={}),
            patch("zils.runtime.request", request),
            redirect_stdout(out),
        ):
            miner.main(argv)
        self.assertEqual(sent["path"], "/soup/report")
        self.assertEqual(
            sent["config"],
            {"validator": "https://validator.example",
             "validator_hotkey": self.validator.ss58_address},
        )  # fmt: skip
        payload = verified(sent["message"], self.alice.ss58_address)
        self.assertEqual(payload["validator"], self.validator.ss58_address)
        self.assertIn("Codes: not_selected / no_soup_gain", out.getvalue())

    def run_report(self, argv, request, validators=None):
        out, err = io.StringIO(), io.StringIO()
        listed = [self.validator.ss58_address] if validators is None else validators
        with (
            patch.object(miner, "wallet", return_value=SimpleNamespace(hotkey=self.alice)),
            patch.object(miner, "load_config", return_value={}),
            patch("zils.runtime.request", request),
            patch.object(soup_report, "relay_validators", return_value=listed) as asked,
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            try:
                miner.main(["report", *argv, "--wallet-name", "w", "--wallet-hotkey", "h"])
            except SystemExit:
                pass
        return out.getvalue(), err.getvalue(), asked

    def test_with_no_validator_named_the_hosted_api_relays_the_request(self):
        sent = {}

        def request(config, path, message):
            sent.update(config=config, path=path, message=message)
            return self.reply

        out, _, asked = self.run_report([], request)
        asked.assert_called_once_with("https://training.zils.ai")
        self.assertEqual(
            (sent["config"], sent["path"]),
            ({"validator": "https://training.zils.ai/v1/network",
              "validator_hotkey": self.validator.ss58_address}, "/soup/report"),
        )  # fmt: skip
        payload = verified(sent["message"], self.alice.ss58_address)
        self.assertEqual(payload["validator"], self.validator.ss58_address)
        self.assertIn("Codes: not_selected / no_soup_gain", out)

    def test_a_named_hotkey_and_relay_are_used_as_given(self):
        sent = {}

        def request(config, path, message):
            sent.update(config=config)
            return self.reply

        other = Keypair.create_from_uri("//Other").ss58_address
        _, _, asked = self.run_report(
            ["--relay", "https://api.example/", "--validator-hotkey", other], request
        )
        asked.assert_not_called()
        self.assertEqual(
            sent["config"],
            {"validator": "https://api.example/v1/network", "validator_hotkey": other},
        )

    def test_a_validator_url_needs_its_hotkey(self):
        _, err, asked = self.run_report(["--validator", "https://validator.example"], None)
        self.assertIn("--validator needs --validator-hotkey", err)
        asked.assert_not_called()

    def test_a_relay_that_names_no_validator_is_an_error(self):
        _, err, _ = self.run_report([], None, validators=[])
        self.assertIn("names no validator", err)

    def test_busy_and_unavailable_answers_say_to_try_again(self):
        def refuse(code):
            def request(config, path, message):
                raise urllib.error.HTTPError(config["validator"] + path, code, "x", {}, None)

            return request

        self.assertIn("try again in a minute", self.run_report([], refuse(429))[1])
        self.assertIn("try again later", self.run_report([], refuse(503))[1])
        self.assertIn("clock", self.run_report([], refuse(401))[1])

    def run_dashboard(self, argv, answer, validators=None):
        """(stdout, stderr, the request posted) for `zils-miner dashboard`; ``answer`` is the
        hosted API's reply or the HTTP status it refuses with."""
        out, err, sent = io.StringIO(), io.StringIO(), {}

        def open_(request, timeout):
            sent.update(url=request.full_url, message=json.loads(request.data))
            if isinstance(answer, int):
                raise urllib.error.HTTPError(request.full_url, answer, "x", {}, None)
            return io.BytesIO(json.dumps(answer).encode())

        listed = [self.validator.ss58_address] if validators is None else validators
        with (
            patch.object(miner, "wallet", return_value=SimpleNamespace(hotkey=self.alice)),
            patch.object(miner, "load_config", return_value={}),
            patch("zils.protocol.opener", return_value=SimpleNamespace(open=open_)),
            patch.object(soup_report, "relay_validators", return_value=listed),
            redirect_stdout(out),
            redirect_stderr(err),
        ):
            try:
                miner.main(["dashboard", *argv, "--wallet-name", "w", "--wallet-hotkey", "h"])
            except SystemExit:
                pass
        return out.getvalue(), err.getvalue(), sent

    def test_dashboard_posts_a_signed_report_request_and_prints_the_private_link(self):
        link = "https://zils.ai/miners/dashboard#token"
        out, _, sent = self.run_dashboard([], {"url": link, "expires_at": "2026-11-13T15:00:00Z"})
        self.assertEqual(sent["url"], "https://training.zils.ai/v1/network/miner/dashboard")
        payload = verified(sent["message"], self.alice.ss58_address)
        self.assertEqual(
            (payload["kind"], payload["validator"]), ("soup-report", self.validator.ss58_address)
        )
        self.assertIn(link, out)
        self.assertIn("2026-11-13 15:00 UTC", out)
        self.assertIn("zils-miner dashboard", out)

    def test_dashboard_uses_a_named_relay_and_refuses_a_link_that_is_not_https(self):
        link = "https://site.example/miners/dashboard#t"
        answer = {"url": link, "expires_at": "2026-11-13T15:00:00Z"}
        out, _, sent = self.run_dashboard(["--relay", "https://api.example/"], answer)
        self.assertEqual(sent["url"], "https://api.example/v1/network/miner/dashboard")
        self.assertIn(link, out)
        for bad in ({"url": "javascript:alert(1)"}, {"url": 5}, {}, []):
            out, err, _ = self.run_dashboard([], bad)
            self.assertIn("did not return a dashboard link", err)
            self.assertEqual(out, "")

    def test_dashboard_says_what_to_do_when_refused_or_unavailable(self):
        self.assertIn("try again in a minute", self.run_dashboard([], 429)[1])
        self.assertIn("try again later", self.run_dashboard([], 503)[1])
        self.assertIn("clock", self.run_dashboard([], 401)[1])
        self.assertIn("names no validator", self.run_dashboard([], {}, validators=[])[1])

    def test_relay_validators_reads_the_hosted_api_list(self):
        body = io.BytesIO(json.dumps({"validators": ["5A", "5B"]}).encode())
        opener = SimpleNamespace(open=lambda url, timeout: body)
        with patch("zils.protocol.opener", return_value=opener):
            self.assertEqual(soup_report.relay_validators("https://api.example"), ["5A", "5B"])
        for bad in ({"validators": "5A"}, {"validators": [5]}, [], {}):
            opener = SimpleNamespace(
                open=lambda url, timeout, b=bad: io.BytesIO(json.dumps(b).encode())
            )
            with patch("zils.protocol.opener", return_value=opener), self.assertRaises(ValueError):
                soup_report.relay_validators("https://api.example")


if __name__ == "__main__":
    unittest.main()
