import unittest

from zils import models
from zils.coordinator import public_job


class ImagePublicTest(unittest.TestCase):
    def test_public_aggregates_are_versioned_and_strip_private_inference(self):
        policy = {"min_accuracy": 0.8, "min_brier_improvement": 0.01}
        metrics = {
            "cases": 100,
            "accuracy": 0.84,
            "brier": 0.2,
            "skill": 0.6,
            "nll": 0.4,
            "unknown_rate": 0.02,
            "predictions": ["secret-photo"],
        }
        job = {
            "id": "fixture",
            "status": "completed",
            "model_profile": models.spec(models.IMAJEV),
            "acceptance": policy,
            "result": {
                "baseline": metrics,
                "miners": [dict(metrics, uid=1, status="evaluated")],
                "delivery": {"status": "no_qualifying_model", "acceptance": policy},
                "weights": {},
                "raw_logits": ["secret"],
            },
        }
        result = public_job(job)
        self.assertEqual(result["acceptance"], policy)
        self.assertEqual(result["result"]["image_metrics_version"], "zils-image-metrics/v1")
        self.assertEqual(result["result"]["baseline"]["count"], 100)
        self.assertNotIn("secret", str(result))
