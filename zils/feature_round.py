"""Audited synthetic box exams, private prediction caches and reviewable promotion evidence."""

import http.client
import json
import math
import os
import shutil
import stat
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError

from . import (
    base_round,
    benchmark,
    box,
    feature_commitments as bundles,
    feature_tracks,
    models,
    protocol,
)
from .runtime import digest

VERSION = "zils-box-draw/v1"
DELAY = 12 * 3600
INCUMBENT_UID = 65535
BINDINGS = ("track_id", "model", "incumbent_sha256", "draws", "policy", "decoder", "scope")


def _regular(path):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("draw assets must be regular files without symlinks")
    return path


def audit(root):
    """Verify labels, exact expansion and every SHA-addressed image before inference."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("draw root must be a directory without symlinks")
    manifest = json.loads(_regular(root / "manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "scope", "files", "assets"}
        or manifest["version"] != VERSION
        or manifest["scope"] != feature_tracks.SCOPE
        or not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != {"cases.jsonl", "expanded.jsonl"}
        or not isinstance(manifest["assets"], dict)
    ):
        raise ValueError("unsupported synthetic box draw manifest")
    for name, expected in manifest["files"].items():
        if digest(_regular(root / name)) != expected:
            raise ValueError("box draw file differs from its manifest hash")
    cases = benchmark.read_jsonl(root / "cases.jsonl")
    box.validate_cases(cases)
    expanded = benchmark.read_jsonl(root / "expanded.jsonl")
    if expanded != box.expand_cases(cases):
        raise ValueError("box question expansion differs from its source cases")
    images = root / "images"
    if images.is_symlink() or not images.is_dir():
        raise ValueError("images must be a directory without symlinks")
    expected = {case["image"] for case in cases}
    if set(manifest["assets"]) != expected or {path.name for path in images.iterdir()} != expected:
        raise ValueError("draw image assets differ from the cases")
    for name, expected_hash in manifest["assets"].items():
        if (
            not isinstance(expected_hash, str)
            or name != expected_hash + ".png"
            or digest(_regular(images / name)) != expected_hash
        ):
            raise ValueError("box image bytes differ from their SHA256 filename")
    return manifest, cases


def draw(root):
    _, cases = audit(root)
    return {
        "manifest_sha256": digest(Path(root) / "manifest.json"),
        "dataset_sha256": base_round.dataset_hash(box.expand_cases(cases)),
    }


def rounds(config, directory):
    if "feature_tracks" not in config:
        return {}
    normalized = feature_tracks.validate(config["feature_tracks"])
    enabled = {name: entry for name, entry in normalized["tracks"].items() if entry["enabled"]}
    private = config.get("feature_track_rounds", {})
    if not isinstance(private, dict) or set(private) != set(enabled):
        raise ValueError("feature_track_rounds must name each enabled track exactly")
    result = {}
    for name, entry in enabled.items():
        paths = private[name]
        if (
            not isinstance(paths, dict)
            or set(paths) != {"reference", "draw_dirs"}
            or not isinstance(paths["reference"], str)
            or not isinstance(paths["draw_dirs"], list)
            or len(paths["draw_dirs"]) != 2
            or any(not isinstance(p, str) for p in paths["draw_dirs"])
        ):
            raise ValueError("each feature track requires reference and two draw_dirs")
        reference = Path(directory) / paths["reference"]
        manifest = bundles.validate_bundle(reference, allow_base=True)
        if (
            manifest["track_id"] != name
            or manifest["model"] != entry["model"]
            or bundles.bundle_hash(reference) != entry["incumbent_sha256"]
        ):
            raise ValueError("feature reference differs from its track incumbent or parent")
        seen = {key: set() for key in ("id", "group_id", "image")}
        draws = []
        for binding, location in zip(entry["draws"], paths["draw_dirs"], strict=True):
            path = Path(directory) / location
            _, cases = audit(path)
            if draw(path) != binding:
                raise ValueError("feature draw differs from the configured binding")
            for key, previous in seen.items():
                values = {case[key] for case in cases}
                if values & previous:
                    raise ValueError("feature draws must have disjoint case ids, groups and images")
                previous.update(values)
            draws.append((binding, path))
        result[name] = {
            "entry": entry,
            "reference": reference,
            "root": Path(directory) / "feature-tracks" / name,
            "draws": draws,
        }
    return result


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("feature evaluation time must be finite and nonnegative")
    return value


def _binding(name, entry):
    return {
        "track_id": name,
        "model": entry["model"],
        "incumbent_sha256": entry["incumbent_sha256"],
        "draws": entry["draws"],
        "policy": entry["policy"],
        "decoder": box.DECODER,
        "scope": feature_tracks.SCOPE,
    }


def confirm(first, second):
    """Confirm the same eligible bundle on separately audited draws, at least 12h apart."""
    if any(first.get(k) != second.get(k) for k in BINDINGS):
        raise ValueError("feature confirmation differs from the first competition contract")
    entry = {k: first[k] for k in ("model", "incumbent_sha256", "draws", "policy")}
    for receipt in (first, second):
        feature_tracks.validate_decision(first["track_id"], entry, receipt)
        if receipt.get("evaluated_draw") not in first["draws"]:
            raise ValueError("feature confirmation names an unconfigured draw")
    if any(
        first["evaluated_draw"][k] == second["evaluated_draw"][k]
        for k in ("manifest_sha256", "dataset_sha256")
    ):
        raise ValueError("feature confirmation requires disjoint draws")
    pending = first.get("pending_winner")
    if pending is None:
        return None
    selected = []
    for receipt in (first, second):
        matches = [
            row
            for row in receipt["candidates"]
            if all(row.get(k) == pending[k] for k in ("uid", "hotkey", "sha256"))
        ]
        if len(matches) != 1 or not _eligible(matches[0], first["policy"]):
            return None
        selected.append(matches[0])
    if _time(selected[1]["evaluated_at"]) - _time(selected[0]["evaluated_at"]) < DELAY:
        raise ValueError("feature confirmation must be evaluated at least 12 hours later")
    return pending


def _eligible(row, policy):
    policy = box.validate_policy(policy)
    inc, candidate = row["incumbent"], row["candidate"]
    for scores in (inc, candidate):
        for value in [
            scores["mean_iou"],
            scores["false_positive_rate"],
            scores["presence_brier"],
            *scores["family_iou"].values(),
        ]:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid feature comparison metrics")
    for key in ("gain", "lower_95"):
        if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
            raise ValueError("invalid feature paired comparison")
    return (
        math.isclose(row["gain"], candidate["mean_iou"] - inc["mean_iou"], abs_tol=1e-12)
        and row["gain"] >= policy["min_iou_gain"]
        and row["lower_95"] > 0
        and candidate["mean_iou"] >= policy["min_mean_iou"]
        and set(candidate["family_iou"]) == set(inc["family_iou"])
        and all(
            candidate["family_iou"][k] >= v - policy["max_family_iou_drop"]
            for k, v in inc["family_iou"].items()
        )
        and candidate["false_positive_rate"] <= policy["max_false_positive_rate"]
        and candidate["presence_brier"]
        <= inc["presence_brier"] + policy["max_presence_brier_increase"]
    )


def _scores(
    args,
    attempt,
    store,
    uid,
    bundle,
    path,
    item,
    draw_path,
    model,
    wait_for_capacity,
    *,
    hotkey=None,
):
    from . import validator

    manifest = bundles.validate_bundle(path, allow_base=uid == INCUMBENT_UID)
    if bundles.bundle_hash(path) != bundle:
        raise ValueError("feature bundle changed before evaluation")
    checkpoint = manifest["checkpoint_sha256"]
    expected = {
        "bundle_sha256": bundle,
        "checkpoint_sha256": checkpoint,
        "draw": item,
        "model": models.spec(model),
        "decoder": box.DECODER,
        "hotkey": hotkey,
    }
    role = "incumbent-" if uid == INCUMBENT_UID else ""
    cache = store / bundle / (role + item["manifest_sha256"] + "-scores.json")
    _, cases = audit(draw_path)
    expanded = box.expand_cases(cases)
    cached = cache.exists()
    if cached:
        saved = json.loads(_regular(cache).read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("private feature cache binding changed")
    else:
        report = validator.run_evaluate(
            args,
            attempt,
            uuid.uuid4().hex,
            [{"uid": uid, "sha256": checkpoint, "checkpoint": str(path.resolve())}],
            draw_path / "expanded.jsonl",
            base_revision=models.spec(model)["base_revision"],
            images=draw_path / "images",
            wait_for_capacity=wait_for_capacity,
        )
        if len(report["miners"]) != 1 or report["miners"][0]["uid"] != uid:
            raise RuntimeError("feature evaluator returned an incomplete batch")
        saved = {**expected, **report, "evaluated_at": time.time()}
    _time(saved.get("evaluated_at"))
    if len(saved["miners"]) != 1 or saved["miners"][0].get("sha256") != checkpoint:
        raise ValueError("feature prediction checkpoint binding changed")
    rows = base_round.rows_for(saved, expanded)
    if not rows:
        if uid == INCUMBENT_UID:
            raise RuntimeError("feature incumbent evaluation failed")
        return None, saved["evaluated_at"]
    row = next(iter(rows.values()))
    if row.get("runtime", {}).get("model") != models.spec(model):
        raise RuntimeError("feature evaluator runtime differs from the pinned model")
    if not cached:
        cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        protocol.write_json(cache, saved)
    return row["predictions"], saved["evaluated_at"]


def _track_epoch(name, prepared, registry, args, work, wait_for_capacity):
    entry, root = prepared["entry"], prepared["root"]
    sidecar, store = root / "decision.json", root / "store"
    sidecar.unlink(missing_ok=True)
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempt = work / ("feature-" + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700, parents=True)
    model = models.validate_spec(entry["model"])
    rejected, candidates, verified = {}, {}, {}
    incumbent_checkpoint = bundles.validate_bundle(prepared["reference"], allow_base=True)[
        "checkpoint_sha256"
    ]
    for uid, claim in sorted(registry.items()):
        try:
            if uid == INCUMBENT_UID or claim["sha256"] == entry["incumbent_sha256"]:
                raise ValueError("incumbent copy")
            target = store / claim["sha256"] / "checkpoint"
            if not target.exists():
                temporary = target.with_name("fetch-" + uuid.uuid4().hex)
                temporary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                try:
                    bundles.fetch(claim, temporary, track_id=name, model=model)
                    temporary.rename(target)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
            manifest = bundles.validate_bundle(target)
            if (
                manifest["track_id"] != name
                or manifest["model"] != entry["model"]
                or manifest["checkpoint_sha256"] == incumbent_checkpoint
                or bundles.bundle_hash(target) != claim["sha256"]
            ):
                raise ValueError("candidate binding mismatch")
            verified.setdefault(manifest["checkpoint_sha256"], {})[uid] = (claim["sha256"], target)
        except (ValueError, HTTPError) as error:
            if isinstance(error, HTTPError) and (
                not 400 <= error.code < 500 or error.code in (408, 429)
            ):
                raise
            rejected[str(uid)] = "invalid_feature_bundle"
    # Establish ownership only after all downloads succeed, grouping manifest aliases by weights.
    for checkpoint, holders in verified.items():
        ownership = store / "claims" / (checkpoint + ".json")
        if ownership.exists():
            owner = json.loads(_regular(ownership).read_text())["hotkey"]
            winner = next((uid for uid in holders if registry[uid]["hotkey"] == owner), None)
        else:
            ordered = sorted(holders, key=lambda uid: (registry[uid]["block"], uid))
            winner = (
                ordered[0]
                if len(ordered) == 1
                or registry[ordered[0]]["block"] < registry[ordered[1]]["block"]
                else None
            )
            if winner is not None:
                ownership.parent.mkdir(mode=0o700, exist_ok=True)
                protocol.write_json(
                    ownership, {k: registry[winner][k] for k in ("hotkey", "block")}
                )
        for uid in holders:
            if uid == winner:
                candidates[uid] = holders[uid]
            else:
                rejected[str(uid)] = "duplicate_feature_checkpoint"
    decisions, qualification = [], []
    for index, (item, path) in enumerate(prepared["draws"]):
        _, cases = audit(path)
        incumbent, baseline_time = _scores(
            args,
            attempt,
            store,
            INCUMBENT_UID,
            entry["incumbent_sha256"],
            prepared["reference"],
            item,
            path,
            model,
            wait_for_capacity,
        )
        qualification.append(
            {"draw": item, "evaluated_at": baseline_time, "scores": box.score(cases, incumbent)}
        )
        rows = []
        for uid, (bundle, checkpoint) in sorted(candidates.items()):
            if index:
                previous = next((r for r in decisions[0]["candidates"] if r["uid"] == uid), None)
                if (
                    previous is None
                    or not previous["eligible"]
                    or time.time() - previous["evaluated_at"] < DELAY
                ):
                    continue
            predictions, evaluated_at = _scores(
                args,
                attempt,
                store,
                uid,
                bundle,
                checkpoint,
                item,
                path,
                model,
                wait_for_capacity,
                hotkey=registry[uid]["hotkey"],
            )
            if predictions is None:
                rejected[str(uid)] = "candidate_evaluation_failed"
                continue
            rows.append(
                {
                    "uid": uid,
                    "hotkey": registry[uid]["hotkey"],
                    "sha256": bundle,
                    "evaluated_at": evaluated_at,
                    **box.compare(cases, incumbent, predictions, entry["policy"]),
                }
            )
        eligible = sorted(
            (row for row in rows if row["eligible"]), key=lambda r: (-r["gain"], r["uid"])
        )
        pending = {k: eligible[0][k] for k in ("uid", "hotkey", "sha256")} if eligible else None
        decisions.append(
            {
                **_binding(name, entry),
                "evaluated_draw": item,
                "evaluated_at": max([baseline_time, *[row["evaluated_at"] for row in rows]]),
                "candidates": rows,
                "pending_winner": pending,
            }
        )
    confirmed = confirm(*decisions) if decisions[1]["candidates"] else None
    result = {
        **decisions[0],
        "qualification": qualification,
        "rejected": rejected,
        "confirmation": decisions[1] if decisions[1]["candidates"] else None,
        "confirmed_winner": confirmed,
        "evaluated_at": max(d["evaluated_at"] for d in decisions),
    }
    protocol.write_json(sidecar, result)
    return result


def epoch(config, directory, work, snapshot, args, prepared=None, *, wait_for_capacity=True):
    # Audit again on every epoch; a prepared path mapping is never evidence that bytes stayed fixed.
    if "feature_tracks" not in config:
        return {}
    normalized = feature_tracks.validate(config["feature_tracks"])
    for name, entry in normalized["tracks"].items():
        if entry["enabled"]:
            (Path(directory) / "feature-tracks" / name / "decision.json").unlink(missing_ok=True)
    prepared = rounds(config, directory)
    routed, rejected = {name: {} for name in prepared}, {}
    for uid, claim in bundles.candidates(snapshot).items():
        try:
            if "error" in claim:
                raise ValueError("invalid commitment")
            hotkey = snapshot.get("members", {}).get(str(uid))
            if not isinstance(hotkey, str) or not hotkey:
                raise ValueError("feature claimant is not registered in the frozen snapshot")
            manifest = bundles.parent(claim)
            name = manifest["track_id"]
            if name not in prepared or manifest["model"] != prepared[name]["entry"]["model"]:
                raise ValueError("unregistered feature parent")
            routed[name][uid] = {**claim, "hotkey": hotkey}
        except (ValueError, HTTPError) as error:
            if isinstance(error, HTTPError) and (
                not 400 <= error.code < 500 or error.code in (408, 429)
            ):
                raise
            rejected[str(uid)] = "invalid_feature_commitment"
    decisions, errors = {}, {}
    for name, settings in prepared.items():
        try:
            decisions[name] = _track_epoch(
                name, settings, routed[name], args, Path(work), wait_for_capacity
            )
        except (ValueError, OSError, RuntimeError, http.client.HTTPException) as error:
            (settings["root"] / "decision.json").unlink(missing_ok=True)
            errors[name] = str(error)
    intake = Path(work) / ("feature-intake-" + uuid.uuid4().hex + ".json")
    protocol.write_json(intake, {"rejected": rejected, "errors": errors})
    os.replace(intake, Path(work) / "feature-intake.json")
    if errors:
        raise RuntimeError("feature evaluation failed; inspect private feature-intake.json")
    return decisions


def attach(config, directory, work, report):
    if "feature_tracks" not in config:
        return report
    normalized = feature_tracks.validate(config["feature_tracks"])
    frozen = Path(work) / "feature-tracks.json"
    if frozen.exists():
        saved = json.loads(_regular(frozen).read_text())
        if saved.get("config") != normalized:
            raise ValueError("pending feature report differs from the installed configuration")
    else:
        decisions = {}
        for name, entry in normalized["tracks"].items():
            if entry["enabled"]:
                path = Path(directory) / "feature-tracks" / name / "decision.json"
                if not path.exists():
                    raise ValueError("each enabled feature track requires a current decision")
                decisions[name] = json.loads(_regular(path).read_text())
        saved = {"config": normalized, "decisions": decisions}
    enabled = {name for name, entry in normalized["tracks"].items() if entry["enabled"]}
    if set(saved["decisions"]) != enabled:
        raise ValueError("frozen feature decisions must cover every enabled track")
    for name in enabled:
        feature_tracks.validate_decision(name, normalized["tracks"][name], saved["decisions"][name])
    if not frozen.exists():
        protocol.write_json(frozen, saved)
    return {**report, "feature_tracks": saved}
