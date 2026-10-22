"""Image job data for isolated fleet tests."""

import hashlib
import uuid

from zils import models

OWNER = "10000000-0000-4000-8000-000000000002"


JOB = "10000000-0000-4000-8000-000000000003"


# Image targets of the retired Imajev profile; image jobs now take the text policy only.
POLICY = {
    "min_accuracy": 0.8,
    "min_brier_improvement": 0.01,
    "positive_class": "damaged",
    "min_positive_recall": 0.9,
    "max_false_positive_rate": 0.1,
}


TEXT_POLICY = {"min_accuracy": 0.8, "min_brier_improvement": 0.01}


def fixture(model=models.D3):
    """A d3 image job freezes its photos under the text acceptance policy."""
    job = {
        "id": JOB,
        "owner_id": OWNER,
        "status": "uploading",
        "model_profile": models.spec(model),
        "acceptance": TEXT_POLICY,
    }
    splits, assets = {}, {}
    for split in ("train", "calibration", "test"):
        splits[split] = []
        for i, label in enumerate(("normal", "damaged")):
            aid = str(uuid.uuid4())
            sha = hashlib.sha256((split + str(i)).encode()).hexdigest()
            assets[aid] = {
                "id": aid,
                "owner_id": OWNER,
                "job_id": JOB,
                "purpose": "training",
                "state": "ready",
                "canonical_sha256": sha,
                "pixel_sha256": sha,
                "canonical_bytes": 100,
                "width": 8,
                "height": 8,
                "preprocessor": "zils-image-rgb-png/v1",
                "expires_at": "2099-01-01T00:00:00Z",
                "filename": "secret-" + label + ".png",
                "canonical_path": "private/path",
            }
            splits[split].append(
                {
                    "id": split + str(i),
                    "group_id": split + str(i),
                    "family": "inspection",
                    "state": {},
                    "question": {
                        "type": "choice",
                        "instructions": "Inspect",
                        "criteria": {"normal": None, "damaged": None},
                    },
                    "label": label,
                    "image": {"asset_id": aid},
                }
            )
    return job, splits, assets
