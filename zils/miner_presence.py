"""Optional signed readiness. Capacity and quality certificates stay server-owned."""

import re
import threading
from contextlib import contextmanager

from . import models
from .cloud import APIError
from .miner_grading import trainer_identity
from .runtime import gpu_ready


def validate_presence(body):
    if not isinstance(body, dict) or set(body) != {"profiles"}:
        raise APIError(400, "Presence contains only installed profiles and readiness.")
    profiles = body["profiles"]
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= len(models.SPECS):
        raise APIError(400, "Advertise 1–3 installed profiles.")
    seen = set()
    for profile in profiles:
        if (
            not isinstance(profile, dict)
            or set(profile)
            != {"model", "profile_sha256", "runtime_sha256", "trainer_sha256", "ready"}
            or not isinstance(profile["model"], str)
            or profile["model"] not in models.SPECS
            or profile["model"] in seen
            or type(profile["ready"]) is not bool
            or any(
                not isinstance(profile[k], str) or not re.fullmatch("[a-f0-9]{64}", profile[k])
                for k in ("profile_sha256", "runtime_sha256", "trainer_sha256")
            )
            or any(profile[k] != v for k, v in models.profile_identity(profile["model"]).items())
        ):
            raise APIError(400, "Invalid installed profile or readiness.")
        seen.add(profile["model"])
    return profiles


def presence_payload(references, device, *, minimum_mib=0, busy=False):
    profiles = []
    for model in sorted(references):
        try:
            ready = (
                not busy
                and minimum_mib > 0
                and gpu_ready(device, model=model, minimum_mib=minimum_mib)
            )
        except (ValueError, OSError, RuntimeError):
            ready = False
        profiles.append(
            {
                "model": model,
                **models.profile_identity(model),
                "trainer_sha256": trainer_identity(),
                "ready": ready,
            }
        )
    body = {"profiles": profiles}
    validate_presence(body)
    return body


@contextmanager
def presence_loop(client, references, device, interval=15, *, enabled=False, minimum_mib=0):
    busy, stop = threading.Event(), threading.Event()
    if not enabled:
        yield busy
        return
    if type(minimum_mib) is not int or minimum_mib < 1:
        raise ValueError("Graded scheduling requires a measured positive graded_min_free_mib guard")

    def advertise():
        while not stop.is_set():
            try:
                client.call(
                    "heartbeat",
                    presence_payload(
                        references, device, minimum_mib=minimum_mib, busy=busy.is_set()
                    ),
                )
            except (APIError, ValueError, OSError):
                pass  # Missing receipts expire server-side; never report an invented success.
            stop.wait(interval)

    thread = threading.Thread(target=advertise, name="miner-presence", daemon=True)
    thread.start()
    try:
        yield busy
    finally:
        stop.set()
        thread.join(timeout=41)
