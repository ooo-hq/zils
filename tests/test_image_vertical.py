"""End-to-end image contracts. Fixture scores are not a model benchmark."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from zils import models
from zils.coordinator import Processor


class ImageVerticalTest(unittest.TestCase):
    def test_rehearsal_recovers_lost_claim_reply_without_waiting_for_lease_expiry(self):
        from tests.image_vertical_fixture import Storage, run
        from zils.cloud import APIError

        original = Storage.rpc
        interrupted = False

        def lose_reply(store, name, values):
            nonlocal interrupted
            result = original(store, name, values)
            if name.startswith("zils_image_claim_finalize") and not interrupted:
                interrupted = True
                raise APIError(503, "Supabase is unavailable; please retry.")
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(Storage, "rpc", lose_reply):
            result = run(Path(tmp) / "run", mode="negative")
        self.assertTrue(interrupted)
        self.assertEqual(result["status"], "completed")

    def test_rehearsal_recovers_lost_finalization_reply_without_rewriting_the_image(self):
        from tests.image_vertical_fixture import Storage, run
        from zils.cloud import APIError

        original = Storage.rpc
        interrupted = False
        finishes = {}

        def lose_reply(store, name, values):
            nonlocal interrupted
            result = original(store, name, values)
            if name == "zils_image_finish":
                asset = values["p_asset"]
                finishes[asset] = finishes.get(asset, 0) + 1
                if not interrupted:
                    interrupted = True
                    raise APIError(503, "Supabase is unavailable; please retry.")
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.object(Storage, "rpc", lose_reply):
            result = run(Path(tmp) / "run", mode="negative")
        self.assertTrue(interrupted)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(max(finishes.values()), 1)
        retries = [step for step in result["steps"] if step["name"] == "transient_retry"]
        self.assertEqual(len(retries), 1)

    def test_rehearsal_does_not_replay_a_billable_prediction_after_a_lost_reply(self):
        from tests.image_vertical_fixture import run
        from zils.api import Gateway
        from zils.decisions import DecisionError

        original = Gateway.evaluate
        predictions = 0

        def lose_reply(gateway, *args, **kwargs):
            nonlocal predictions
            original(gateway, *args, **kwargs)
            predictions += 1
            raise DecisionError(503, "service_unavailable", "Response interrupted.")

        with tempfile.TemporaryDirectory() as tmp, patch.object(Gateway, "evaluate", lose_reply):
            with self.assertRaises(RuntimeError):
                run(Path(tmp) / "run", mode="accepted")
        self.assertEqual(predictions, 1)

    def test_capacity_deferral_does_not_claim_an_image_ahead_of_ready_text(self):
        engine = object.__new__(Processor)
        engine.args = SimpleNamespace(device="cuda")
        engine.references = {models.JEVK5: Path("text"), models.IMAJEV: Path("image")}
        engine.store = Mock()
        engine.store.rpc.return_value = None
        with patch(
            "zils.coordinator.gpu_ready",
            side_effect=lambda device, **kw: kw.get("model") != models.IMAJEV,
        ):
            self.assertFalse(engine.tick())
        calls = engine.store.rpc.call_args_list
        self.assertEqual(
            calls[0].args,
            (
                "zils_claim_profile_processing",
                {"p_stage": "validating", "p_profiles": [models.JEVK5, models.IMAJEV]},
            ),
        )
        self.assertEqual(
            calls[1].args,
            (
                "zils_claim_profile_processing",
                {"p_stage": "evaluating", "p_profiles": [models.JEVK5]},
            ),
        )

    def test_rehearsal_refuses_overwrite_and_preserves_redacted_failure(self):
        self.assertIsNotNone(importlib.util.find_spec("scripts.rehearse_image_vertical"))
        from scripts.rehearse_image_vertical import create_run_directory, rehearse

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = create_run_directory(root / "existing")
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                create_run_directory(out)
            with self.assertRaises(ValueError):
                rehearse({}, root / "missing", root / "failed")
            report = json.loads((root / "failed/report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "ValueError")
            self.assertNotIn("missing", str(report))

    def test_full_http_pipeline_isolation_and_saved_negative_outcome(self):
        from tests.image_vertical_fixture import run

        for mode in ("accepted", "negative", "activation_failed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                logs = io.StringIO()
                with redirect_stdout(logs), redirect_stderr(logs):
                    result = run(Path(tmp) / "run", mode=mode)
                for forbidden in (
                    "owner-token",
                    "other-token",
                    "private-fixture-runtime",
                    "http://",
                    "signing-secret",
                ):
                    self.assertNotIn(forbidden, logs.getvalue())
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["text_unchanged"])
                self.assertTrue(result["harness"]["worker_holdouts_denied"])
                self.assertTrue(result["harness"]["text_worker_denied"])
                self.assertEqual(result["serving"]["verified"], mode == "accepted")
                if mode == "negative":
                    self.assertEqual(
                        result["evaluation"]["delivery"]["status"], "no_qualifying_model"
                    )
                if mode == "activation_failed":
                    self.assertEqual(result["workflow"]["state"], "activation_failed")

    def test_image_children_receive_runtime_paths_but_no_account_or_signing_credentials(self):
        from zils.runtime import run_child

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                "os.environ",
                {
                    "ZILS_COMPUTE_LOCK": str(Path(tmp) / "gpu.lock"),
                    "ZILS_IMAGE_REFERENCE": "/pinned/reference",
                    "CUSTOMER_SESSION_TOKEN": "session-secret",
                    "REHEARSAL_LIVE_TEXT_TOKEN": "live-secret",
                    "BTT_WALLET_SEED": "signing-secret",
                    "SUPABASE_SERVICE_ROLE_KEY": "server-secret",
                },
            ),
            patch("zils.runtime.gpu_ready", return_value=True),
            patch("zils.runtime._run_child") as child,
        ):
            run_child(["python"], Path(tmp) / "log", "cuda", model=models.IMAJEV)
            environment = child.call_args.args[2]
            self.assertEqual(environment["ZILS_IMAGE_REFERENCE"], "/pinned/reference")
            for name in (
                "CUSTOMER_SESSION_TOKEN",
                "REHEARSAL_LIVE_TEXT_TOKEN",
                "BTT_WALLET_SEED",
                "SUPABASE_SERVICE_ROLE_KEY",
            ):
                self.assertFalse(name in environment, name)

    def test_rehearsal_accepts_text_list_and_image_dictionary_health_formats(self):
        from scripts.rehearse_image_vertical import verify_health_identity

        expected = "a" * 64
        entry = {"release_id": "model", "fingerprint": expected}
        for reply in ({"models": [entry]}, {"models": {"model": entry}}, entry):
            self.assertEqual(verify_health_identity(reply, expected)["fingerprint"], expected)
        with self.assertRaises(ValueError):
            verify_health_identity({"models": []}, expected)
