"""Reviewable base releases: private evidence, complete frozen rosters, no deployment."""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from . import (
    base_exam,
    base_round,
    benchmark,
    commitments,
    migration_regression,
    model_pools,
    models,
    regression_suite,
)

SOUP_OWNER = "validator-soup"
MAX_SOUP_MEMBERS = 5
PATHS = {"base_round_dir", "base_reference", "family_history", "suite_manifest", "pilot_report"}


def read(path):
    return json.loads(base_exam._regular(Path(path)).read_text())


def bound(path):
    return {
        "path": str(Path(path).resolve()),
        "sha256": benchmark.file_hash(base_exam._regular(Path(path))),
    }


def evidence_read(evidence):
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"path", "sha256"}
        or bound(evidence["path"]) != evidence
    ):
        raise ValueError("evidence file hash/path changed")
    return read(evidence["path"])


def addressed(value):
    return {**value, "sha256": base_round.dataset_hash(value)}


def verify_address(value):
    if not isinstance(value, dict) or value.get("sha256") != base_round.dataset_hash(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("receipt content hash changed")


def persist(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError("conflicting reuse of immutable receipt/window ID")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    return value


def active_context(config, pool_id, *, directory):
    from . import base_pilot, validator

    pools = model_pools.validate(config["base_pools"])
    entry = pools["pools"].get(pool_id)
    if not entry or not entry["enabled"]:
        raise ValueError("pool is not enabled in active configuration")
    paths = config.get("base_pool_rounds", {}).get(pool_id)
    if (
        not isinstance(paths, dict)
        or set(paths) != PATHS
        or any(not isinstance(v, str) or not v for v in paths.values())
    ):
        raise ValueError("active pool requires reference, draws, history, suite and pilot paths")
    resolved = {k: str((Path(directory) / v).resolve()) for k, v in paths.items()}
    track = model_pools.track(entry)
    settings = {
        **config,
        **resolved,
        "base_tier": track,
        "pool_id": pool_id,
        "model": entry["model"],
    }
    draws = validator.base_draws(settings, Path(directory))
    if len(draws) < 2:
        raise ValueError("active pool requires two independent election draws")
    structures = [read(path / "blueprint.json") for _, path in draws]
    if any(value != structures[0] for value in structures):
        raise ValueError("all pool draws must use the frozen blueprint")
    suite = read(resolved["suite_manifest"])
    regression_suite.validate_suite(suite)
    base_pilot.admit(settings, directory, draws)
    root = Path(directory) / "base-pools" / pool_id
    return {
        **track,
        "pool_id": pool_id,
        "model": entry["model"],
        "draw_roots": {i["manifest_sha256"]: str(p) for i, p in draws},
        "history": read(resolved["family_history"]),
        "blueprint": structures[0],
        "blueprint_sha256": base_round.dataset_hash(structures[0]),
        "suite_manifest": suite,
        "suite_sha256": base_round.dataset_hash(suite),
        "pilot_report": read(resolved["pilot_report"]),
        "pilot_sha256": benchmark.file_hash(Path(resolved["pilot_report"])),
        "ready_dir": str(root / "ready"),
        "windows_dir": str(root / "windows"),
        "settings": settings,
    }


def binding(active):
    return {
        k: active[k]
        for k in (
            "pool_id",
            "model",
            "incumbent_sha256",
            "policy",
            "blueprint_sha256",
            "suite_sha256",
            "pilot_sha256",
        )
    }


def ownership(candidate, snapshot, incumbent):
    claims = commitments.candidates(snapshot, incumbent)
    uid = candidate["uid"]
    claim = claims.get(uid, {})
    if (
        claim.get("sha256") != candidate["sha256"]
        or claim.get("block") != candidate["commitment_block"]
        or snapshot.get("members", {}).get(str(uid)) != candidate["hotkey"]
    ):
        raise ValueError("candidate ownership differs from frozen chain snapshot")
    return claim


def soup_ownership(soup):
    """A validator-built soup has no chain owner: its identity is its own hash and its provenance
    is the list of member bodies, which are paid through the soup pool, never as kings."""
    if (
        not isinstance(soup, dict)
        or set(soup) != {"soup_sha256", "members", "round_id"}
        or not isinstance(soup["round_id"], str)
        or not soup["round_id"]
        or not isinstance(soup["members"], list)
        or not 1 <= len(soup["members"]) <= MAX_SOUP_MEMBERS
    ):
        raise ValueError("soup requires soup_sha256, round_id and 1-5 members")
    base_exam._hash(soup["soup_sha256"])
    for member in soup["members"]:
        if (
            not isinstance(member, dict)
            or set(member) != {"body_sha256", "hotkey"}
            or not isinstance(member["hotkey"], str)
            or not member["hotkey"]
        ):
            raise ValueError("soup members require body_sha256 and hotkey")
        base_exam._hash(member["body_sha256"])
    if len({m["body_sha256"] for m in soup["members"]}) != len(soup["members"]):
        raise ValueError("soup members must be distinct bodies")
    return {"owner": SOUP_OWNER, **soup}


def confirmed(first, second, suite_report, *, candidate_sha256, active, expected_suite, evidence):
    for key, report in (("first", first), ("second", second), ("suite", suite_report)):
        if evidence_read(evidence[key]) != report:
            raise ValueError("readiness input differs from hash-bound evidence")
    if expected_suite != active["suite_manifest"]:
        raise ValueError("suite differs from active suite")
    ready = base_round.confirm_candidates(first, second, active=active)
    candidate = next((c for c in ready if c["sha256"] == candidate_sha256), None)
    if candidate is None:
        raise ValueError("candidate has not qualified on both draws")
    regression_suite.verify_election(
        suite_report,
        candidate_sha256,
        active["incumbent_sha256"],
        active["model"]["id"],
        expected_suite=expected_suite,
    )
    return candidate


def receipt(active, candidate, evidence, **extra):
    return addressed(
        {
            "version": "zils-base-ready/v1",
            "active": binding(active),
            "candidate": {k: candidate[k] for k in ("uid", "sha256", "hotkey", "commitment_block")},
            **extra,
            "completed_at": max(v["completed_at"] for v in candidate["evaluation"].values()),
            "evidence": evidence,
        }
    )


def qualify(first, second, suite_report, *, candidate_sha256, active, expected_suite, evidence):
    if set(evidence) != {"first", "second", "suite", "snapshot_first", "snapshot_second"}:
        raise ValueError("readiness requires both decisions, snapshots and suite evidence")
    candidate = confirmed(
        first,
        second,
        suite_report,
        candidate_sha256=candidate_sha256,
        active=active,
        expected_suite=expected_suite,
        evidence=evidence,
    )
    prior = next(c for c in first["candidates"] if c["sha256"] == candidate_sha256)
    ownership(prior, evidence_read(evidence["snapshot_first"]), active["incumbent_sha256"])
    ownership(candidate, evidence_read(evidence["snapshot_second"]), active["incumbent_sha256"])
    return receipt(active, candidate, evidence)


def qualify_soup(first, second, suite_report, *, soup, active, expected_suite, evidence):
    """The same gate as `qualify` for a validator-built soup: both private decisions and the
    adapter suite, with the soup's decision rows owned by `SOUP_OWNER` instead of a chain claim."""
    owner = soup_ownership(soup)
    if set(evidence) != {"first", "second", "suite"}:
        raise ValueError("soup readiness requires both decisions and suite evidence")
    candidate = confirmed(
        first,
        second,
        suite_report,
        candidate_sha256=soup["soup_sha256"],
        active=active,
        expected_suite=expected_suite,
        evidence=evidence,
    )
    if candidate["hotkey"] != SOUP_OWNER:
        raise ValueError("soup decision rows must be validator-built, not a chain commitment")
    return receipt(active, candidate, evidence, ownership=owner)


def verify_ready(receipt, active):
    verify_address(receipt)
    if receipt.get("version") != "zils-base-ready/v1" or receipt["active"] != binding(active):
        raise ValueError("ready receipt differs from active competition")
    e = receipt["evidence"]
    reports = [evidence_read(e[k]) for k in ("first", "second", "suite")]
    if "ownership" in receipt:
        soup = {k: v for k, v in receipt["ownership"].items() if k != "owner"}
        actual = qualify_soup(
            *reports, soup=soup, active=active, expected_suite=active["suite_manifest"], evidence=e
        )
    else:
        actual = qualify(
            *reports,
            candidate_sha256=receipt["candidate"]["sha256"],
            active=active,
            expected_suite=active["suite_manifest"],
            evidence=e,
        )
    if actual != receipt:
        raise ValueError("ready receipt differs from recomputed evidence")
    return receipt


def inventory(active):
    root = Path(active["ready_dir"])
    if not root.exists():
        return []
    result = {}
    for path in sorted(root.iterdir()):
        receipt = read(path)
        verify_address(receipt)
        if path.name != receipt["sha256"] + ".json":
            raise ValueError("ready inventory filename differs from receipt")
        if receipt["active"] == binding(active):
            verify_ready(receipt, active)
            candidate = receipt["candidate"]
            sha = candidate["sha256"]
            previous = result.get(sha)
            if previous and (
                any(
                    previous["candidate"][k] != candidate[k] for k in ("hotkey", "commitment_block")
                )
                or previous.get("ownership") != receipt.get("ownership")
            ):
                raise ValueError("ready checkpoint has conflicting ownership")
            if previous is None or (receipt["completed_at"], receipt["sha256"]) < (
                previous["completed_at"],
                previous["sha256"],
            ):
                result[sha] = receipt
    return [result[sha] for sha in sorted(result)]


def window_path(active, window_id, kind):
    return Path(active["windows_dir"]) / (window_id.replace(":", "-") + f"-{kind}.json")


def validate_channels(channels):
    if (
        not isinstance(channels, dict)
        or set(channels) != {"stable", "latest"}
        or any(not isinstance(v, str) or not v for v in channels.values())
    ):
        raise ValueError("window requires an opening stable/latest snapshot")
    return dict(channels)


def window(schedule, pool_id, at):
    base_exam._time(at)
    if (
        set(schedule) != {"version", "anchor_utc", "interval_days"}
        or schedule["version"] != "zils-base-windows/v1"
        or type(schedule["interval_days"]) is not int
        or not 14 <= schedule["interval_days"] <= 28
    ):
        raise ValueError("release schedule requires a 14–28 day interval")
    anchor = datetime.fromisoformat(schedule["anchor_utc"].replace("Z", "+00:00"))
    if anchor.tzinfo is None or anchor.utcoffset().total_seconds() != 0:
        raise ValueError("release anchor must use UTC")
    interval = schedule["interval_days"] * 86400
    index = int((at - anchor.timestamp()) // interval)
    if index < 1:
        raise ValueError("release window has not opened")
    return (
        f"{pool_id}:{index}",
        anchor.timestamp() + index * interval,
        anchor.timestamp() + (index + 1) * interval,
    )


def freeze(ready, *, active, schedule, window_id, frozen_at, opening_channels):
    derived, opening, deadline = window(schedule, active["pool_id"], frozen_at)
    if window_id != derived:
        raise ValueError("window ID differs from configured schedule")
    current = inventory(active)
    if sorted(r["sha256"] for r in ready) != sorted(r["sha256"] for r in current):
        raise ValueError("freeze must include the complete active receipt inventory")
    opening_channels = validate_channels(opening_channels)
    target = window_path(active, window_id, "frozen")
    if target.exists():
        saved = read(target)
        verify_address(saved)
        if (
            saved.get("version") != "zils-base-frozen/v1"
            or saved["active"] != binding(active)
            or saved["schedule"] != schedule
            or saved["opening_channels"] != opening_channels
        ):
            raise ValueError("conflicting reuse of frozen window inputs")
        for receipt in saved["ready"]:
            verify_ready(receipt, active)
        return saved
    for receipt in ready:
        verify_ready(receipt, active)
        if receipt["completed_at"] > frozen_at:
            raise ValueError("ready candidate completed after roster freeze")
    if len({r["candidate"]["sha256"] for r in ready}) != len(ready):
        raise ValueError("duplicate candidate in ready inventory")
    payload = addressed(
        {
            "version": "zils-base-frozen/v1",
            "window_id": window_id,
            "active": binding(active),
            "opening_channels": opening_channels,
            "schedule": schedule,
            "frozen_at": frozen_at,
            "opens_at": opening,
            "deadline": deadline,
            "inventory_sha256": base_round.dataset_hash(sorted(r["sha256"] for r in ready)),
            "ready": sorted(ready, key=lambda r: r["candidate"]["sha256"]),
        }
    )
    return persist(target, payload)


def close(frozen, closing, *, active, members, now):
    verify_address(frozen)
    if frozen["version"] != "zils-base-frozen/v1" or frozen["active"] != binding(active):
        raise ValueError("frozen roster differs from active competition")
    path = window_path(active, frozen["window_id"], "frozen")
    if read(path) != frozen:
        raise ValueError("frozen roster differs from persisted window")
    validate_channels(frozen["opening_channels"])
    target = window_path(active, frozen["window_id"], "closed")
    if target.exists():
        saved = read(target)
        verify_address(saved)
        if any(
            saved[k] != v
            for k, v in {"frozen": frozen, "closing": closing, "members": members}.items()
        ):
            raise ValueError("conflicting reuse of closed window inputs")
        now = saved["closed_at"]
    if not frozen["opens_at"] <= now < frozen["deadline"]:
        raise ValueError("closing is outside the configured window")
    snapshot = evidence_read(closing["snapshot"])
    if snapshot.get("members") != members:
        raise ValueError("closing members differ from frozen snapshot")
    decision = closing["decision"]
    identity = base_round.draw(closing["draw_root"])
    extended = {
        **active,
        "draws": [identity],
        "draw_roots": {identity["manifest_sha256"]: closing["draw_root"]},
    }
    audited, _ = base_round.verify_decision(decision, active=extended)
    if audited[0]["pool_id"] != active["pool_id"]:
        raise ValueError("closing draw belongs to another pool")
    qualified_draws = {k: base_exam.audit(p) for k, p in active["draw_roots"].items()}
    roster = {r["candidate"]["sha256"]: r for r in frozen["ready"]}
    for receipt in roster.values():
        verify_ready(receipt, active)
        for split in ("first", "second"):
            report = evidence_read(receipt["evidence"][split])
            manifest = report["manifest_sha256"]
            qualified_draws[manifest] = base_exam.audit(active["draw_roots"][manifest])
    base_exam.disjoint([*qualified_draws.values(), audited])
    if len(decision["candidates"]) != len(roster) or {
        c["sha256"] for c in decision["candidates"]
    } != set(roster):
        raise ValueError("closing must account for every frozen candidate exactly once")
    survivors = []
    for candidate in decision["candidates"]:
        expected = roster[candidate["sha256"]]["candidate"]
        if any(candidate[k] != expected[k] for k in ("hotkey", "commitment_block")):
            raise ValueError("closing candidate differs from frozen ownership")
        if "ownership" not in roster[candidate["sha256"]]:
            ownership(candidate, snapshot, active["incumbent_sha256"])
        if candidate["eligible"]:
            survivors.append(candidate)
        elif not candidate.get("error") and "evaluation" not in candidate:
            raise ValueError("failed closing candidate requires a recorded reason")
    for row in [decision["incumbent"], *decision["candidates"]]:
        for evaluation in row.get("evaluation", {}).values():
            if (
                not frozen["frozen_at"]
                <= evaluation["started_at"]
                <= evaluation["completed_at"]
                <= now
            ):
                raise ValueError("closing inference must follow roster freeze")
    winner = (
        min(survivors, key=lambda c: (-c["score"], c["commitment_block"], c["sha256"]))
        if survivors
        else None
    )
    soup = None
    if winner:
        soup = roster[winner["sha256"]].get("ownership")
        winner = {k: winner[k] for k in ("uid", "sha256", "hotkey", "commitment_block", "score")}
        if soup:
            winner["ownership"] = soup
    kings = active.get("kings", [])
    # A soup is released, not crowned: its members are paid through the soup pool, so the
    # king history is unchanged and the window's winner records the soup hash and members.
    proposed = {
        "incumbent_sha256": winner["sha256"] if winner else active["incumbent_sha256"],
        "kings": [winner["hotkey"], *kings] if winner and not soup else kings,
    }
    result = addressed(
        {
            "version": "zils-base-window/v1",
            "status": "qualified" if winner else "skipped",
            "winner": winner,
            "uploaded": False,
            "proposed_base_tier": proposed,
            "active": binding(active),
            "frozen": frozen,
            "closing": closing,
            "members": members,
            "closed_at": now,
        }
    )
    return persist(target, result)


def verify_window(report, active):
    verify_address(report)
    if (
        close(
            report["frozen"],
            report["closing"],
            active=active,
            members=report["members"],
            now=report["closed_at"],
        )
        != report
    ):
        raise ValueError("window differs from recomputed evidence")
    return report


def channel_proposal(window_report, *, new_profile_id, migration, release_review, active=None):
    if active is None:
        raise ValueError("channel proposal requires a verified closed window")
    verify_window(window_report, active)
    opening_channels = window_report["frozen"]["opening_channels"]
    if new_profile_id is not None and (
        not isinstance(new_profile_id, str)
        or not new_profile_id
        or window_report["status"] != "qualified"
    ):
        raise ValueError("a new latest profile requires a qualified window winner")
    result = {**opening_channels, "latest": new_profile_id or opening_channels["latest"]}
    if migration is not None and release_review is not None:
        prior = opening_channels["latest"]
        if (
            migration.get("profile_id") != prior
            or release_review.get("profile_id") != prior
            or release_review.get("migration_sha256") != base_round.dataset_hash(migration)
        ):
            raise ValueError("release review must bind migration of the opening latest")
        stable = (
            migration_regression.summary(migration["outcomes"])["stable_eligible"]
            and regression_suite.verdict(migration["legacy_regression"])["passed"]
        )
        if (
            stable
            and release_review.get("approved") is True
            and release_review.get("unresolved_problems") == []
        ):
            result["stable"] = prior
    proposal = addressed(
        {
            "version": "zils-base-channels/v1",
            "window_sha256": window_report["sha256"],
            "opening_channels": opening_channels,
            "new_profile_id": new_profile_id,
            "migration": migration,
            "release_review": release_review,
            "channels": result,
        }
    )
    return persist(window_path(active, window_report["frozen"]["window_id"], "channels"), proposal)


def seed_soup(store, soup_sha256, directories, model):
    """Place a frozen soup in the evaluator's checkpoint cache, so the closing draw evaluates it
    like a fetched commitment. The directory must hash to the receipt's soup_sha256."""
    import zils

    target = Path(store) / soup_sha256 / "checkpoint"
    if not target.exists():
        found = [Path(d) for d in directories if zils.checkpoint_hash(Path(d)) == soup_sha256]
        if not found:
            raise ValueError("closing a soup requires --soup-checkpoint matching soup_sha256")
        models.text_runtime(model).validate_candidate(found[0], model)
        staged = target.with_name("seed-" + target.parent.name)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(found[0], staged)
        staged.rename(target)
    if zils.checkpoint_hash(target) != soup_sha256:
        raise ValueError("cached soup checkpoint changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "freeze", "close"))
    for name in ("config", "pool", "out"):
        parser.add_argument("--" + name, required=True)
    for name in (
        "first",
        "second",
        "suite",
        "snapshot-first",
        "snapshot-second",
        "candidate-sha256",
        "soup",
        "ready-dir",
        "schedule",
        "frozen",
        "closing-round",
        "chain-snapshot",
        "channels",
    ):
        parser.add_argument("--" + name)
    parser.add_argument(
        "--soup-checkpoint",
        action="append",
        default=[],
        help="close: a frozen soup's checkpoint directory (repeat per soup)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-python")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    active = active_context(read(config), args.pool, directory=config.parent)
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.command == "qualify" and args.soup:
        paths = {k: getattr(args, k) for k in ("first", "second", "suite")}
        if not all(paths.values()) or any(
            (args.snapshot_first, args.snapshot_second, args.candidate_sha256)
        ):
            parser.error("soup qualify requires first, second and suite, without chain inputs")
        receipt = qualify_soup(
            read(args.first),
            read(args.second),
            read(args.suite),
            soup=read(args.soup),
            active=active,
            expected_suite=active["suite_manifest"],
            evidence={k: bound(p) for k, p in paths.items()},
        )
        persist(Path(active["ready_dir"]) / (receipt["sha256"] + ".json"), receipt)
    elif args.command == "qualify":
        paths = {
            k: getattr(args, k)
            for k in ("first", "second", "suite", "snapshot_first", "snapshot_second")
        }
        if not all(paths.values()) or not args.candidate_sha256:
            parser.error(
                "qualify requires first, second, suite, both snapshots and candidate-sha256"
            )
        receipt = qualify(
            read(args.first),
            read(args.second),
            read(args.suite),
            candidate_sha256=args.candidate_sha256,
            active=active,
            expected_suite=active["suite_manifest"],
            evidence={k: bound(p) for k, p in paths.items()},
        )
        persist(Path(active["ready_dir"]) / (receipt["sha256"] + ".json"), receipt)
    elif args.command == "freeze":
        if (
            not args.ready_dir
            or not args.schedule
            or not args.channels
            or Path(args.ready_dir).resolve() != Path(active["ready_dir"]).resolve()
        ):
            parser.error("freeze requires the installed ready-dir, schedule and opening channels")
        now = time.time()
        schedule = read(args.schedule)
        receipt = freeze(
            inventory(active),
            active=active,
            schedule=schedule,
            window_id=window(schedule, args.pool, now)[0],
            frozen_at=now,
            opening_channels=read(args.channels),
        )
    else:
        if not (args.frozen and args.closing_round and args.chain_snapshot):
            parser.error("close requires frozen, closing-round and chain-snapshot")
        from . import validator

        frozen, snapshot = read(args.frozen), read(args.chain_snapshot)
        verify_address(frozen)
        frozen_file = Path(active["windows_dir"]) / (
            frozen["window_id"].replace(":", "-") + "-frozen.json"
        )
        if frozen["active"] != binding(active) or read(frozen_file) != frozen:
            raise ValueError("closing requires the persisted active frozen roster")
        target = window_path(active, frozen["window_id"], "closed")
        if target.exists():
            receipt = verify_window(read(target), active)
            if (
                receipt["frozen"] != frozen
                or receipt["closing"]["draw_root"] != str(Path(args.closing_round).resolve())
                or receipt["closing"]["snapshot"] != bound(args.chain_snapshot)
            ):
                raise ValueError("conflicting reuse of closed window inputs")
        else:
            base_exam.disjoint(
                [base_exam.audit(p) for p in active["draw_roots"].values()]
                + [base_exam.audit(args.closing_round, history=active["history"])]
            )
            soups = {
                r["candidate"]["uid"]: r["candidate"] for r in frozen["ready"] if "ownership" in r
            }
            candidates = {r["candidate"]["sha256"] for r in frozen["ready"] if "ownership" not in r}
            registry = commitments.candidates(snapshot, active["incumbent_sha256"])
            registry = {
                u: {**c, "hotkey": snapshot["members"][str(u)]}
                for u, c in registry.items()
                if c.get("sha256") in candidates
            }
            if {c["sha256"] for c in registry.values()} != candidates:
                raise ValueError("frozen candidates are missing from the closing snapshot")
            if set(registry) & set(soups):
                raise ValueError("a frozen soup's uid collides with a chain candidate")
            store = Path(active["ready_dir"]).parent / "base-tier" / "store"
            for uid, row in soups.items():
                seed_soup(store, row["sha256"], args.soup_checkpoint, active["model"]["id"])
                registry[uid] = {
                    "sha256": row["sha256"],
                    "hotkey": SOUP_OWNER,
                    "block": row["commitment_block"],
                }
            identity = base_round.draw(args.closing_round)
            settings = {
                **active["settings"],
                "base_tier": {**active["settings"]["base_tier"], "draws": [identity]},
            }
            work = Path(args.out).resolve().parent / ("closing-" + frozen["sha256"])
            work.mkdir(mode=0o700, exist_ok=True)
            result = validator.base_tier_epoch(
                settings,
                Path(active["ready_dir"]).parent,
                work,
                registry,
                SimpleNamespace(device=args.device, runtime_python=args.runtime_python, poll=1),
                [(identity, Path(args.closing_round))],
                closing_admission=active["settings"],
            )
            receipt = close(
                frozen,
                {
                    "decision": result,
                    "draw_root": str(Path(args.closing_round).resolve()),
                    "snapshot": bound(args.chain_snapshot),
                },
                active=active,
                members=snapshot["members"],
                now=time.time(),
            )
    persist(args.out, receipt)
    print(json.dumps({k: receipt[k] for k in ("version", "sha256")}))


if __name__ == "__main__":
    main()
