"""Private sealed election draws and explicit, cross-pool publication history."""

import argparse
import copy
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import zils

from . import base_metrics, benchmark

VERSION = "zils-base-holdout/v2"
HISTORY_VERSION = "zils-base-family-history/v1"
SEED_VERSION = "zils-base-seed/v1"
SPLITS = ("calibration", "test")
FILES = {"calibration.jsonl", "test.jsonl", "registry.json", "blueprint.json", "history.json"}
POOLS = {"d3-lite", "d3-nano", "d3-mini", "d3-flash", "d3"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid publication/evaluation time")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("invalid SHA256 hash")
    return value


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("exam files must be regular files without symlinks")
    return path


def _normal(value):
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, list):
        return [_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    return value


def fingerprints(case):
    question = _normal(case["question"])
    prompt = {"state": _normal(case["state"]), "question": question}
    unordered = digest(prompt)
    if question["type"] == "choice":
        question["criteria"] = list(question["criteria"].items())
    return sorted({unordered, digest(prompt)})


def _receipt(receipt):
    if not isinstance(receipt, dict) or set(receipt) != {"artifact_sha256", "url", "published_at"}:
        raise ValueError("invalid publication receipt")
    _hash(receipt["artifact_sha256"])
    _time(receipt["published_at"])
    url = urlparse(receipt["url"])
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password:
        raise ValueError("publication requires a public HTTP URL")


def validate_history(history):
    if (
        not isinstance(history, dict)
        or set(history) != {"version", "families", "cases"}
        or history["version"] != HISTORY_VERSION
        or not isinstance(history["families"], dict)
        or not isinstance(history["cases"], list)
    ):
        raise ValueError("invalid publication history")
    for family, receipt in history["families"].items():
        if not isinstance(family, str) or not family:
            raise ValueError("publication family must be named")
        _receipt(receipt)
    seen = set()
    for case in history["cases"]:
        if (
            not isinstance(case, dict)
            or set(case) != {"source_id", "fingerprints", "publication"}
            or not isinstance(case["source_id"], str)
            or not case["source_id"]
            or not isinstance(case["fingerprints"], list)
            or not case["fingerprints"]
        ):
            raise ValueError("invalid disclosed case")
        if case["source_id"] in seen:
            raise ValueError("duplicate disclosed source")
        seen.add(case["source_id"])
        for fingerprint in case["fingerprints"]:
            _hash(fingerprint)
        _receipt(case["publication"])


def _metadata(cases, registry):
    zils.validate_cases(cases)
    if not isinstance(registry, dict) or set(registry) != {c["family"] for c in cases}:
        raise ValueError("registry must cover exactly the families")
    for case in cases:
        if "target_probabilities" in case:
            raise ValueError("base election requires hard labels")
        if (
            "image" in case
            or not isinstance(case["state"], dict)
            or set(case["state"]) != {"text"}
            or not isinstance(case["state"]["text"], str)
        ):
            raise ValueError("base exams require inline text, no asset files")
        if not isinstance(case.get("source_id"), str) or not case["source_id"].strip():
            raise ValueError("case requires a source_id")
        entry = registry[case["family"]]
        if (
            not isinstance(entry, dict)
            or entry.get("type") != case["question"]["type"]
            or entry.get("category") != case.get("category")
            or entry.get("category") not in base_metrics.CATEGORY_WEIGHTS
        ):
            raise ValueError("family type/category differs from registry")
        if entry["type"] == "score" and entry.get("ordered_equally_spaced") is not True:
            raise ValueError("Score family must declare ordered equally spaced outcomes")
        if "publication" in entry:
            _receipt(entry["publication"])


def blueprint(cases):
    families = defaultdict(Counter)
    tags = {}
    for split, rows in cases.items():
        for c in rows:
            families[c["family"]][split] += 1
            tags[c["family"]] = [c.get("category"), c.get("block"), c["question"]["type"]]
    return {
        "version": "zils-base-blueprint/v1",
        "families": sorted(
            [
                {
                    "category": tags[f][0],
                    "block": tags[f][1],
                    "type": tags[f][2],
                    **{s: counts[s] for s in SPLITS},
                }
                for f, counts in families.items()
            ],
            key=lambda r: json.dumps(r, sort_keys=True),
        ),
    }


def disjoint(draws):
    ids, sources, prompts, new_families = set(), set(), set(), set()
    for _, cases in draws:
        new = {c["family"] for rows in cases.values() for c in rows if c.get("block") == "new"}
        if new & new_families:
            raise ValueError("new-domain families overlap draws")
        new_families.update(new)
        for rows in cases.values():
            for c in rows:
                fp = set(fingerprints(c))
                if c["id"] in ids or c["source_id"] in sources or fp & prompts:
                    raise ValueError("case/source/prompt overlap across splits or draws")
                ids.add(c["id"])
                sources.add(c["source_id"])
                prompts.update(fp)


def _exposure(cases, history, as_of):
    exposed = [c for c in history["cases"] if c["publication"]["published_at"] <= as_of]
    sources = {c["source_id"] for c in exposed}
    prompts = {p for c in exposed for p in c["fingerprints"]}
    for rows in cases.values():
        if any(c["source_id"] in sources or set(fingerprints(c)) & prompts for c in rows):
            raise ValueError("case was publicly disclosed before evaluation completed")


def _validate(cases, registry, expected_blueprint, history, sealed_at):
    validate_history(history)
    _time(sealed_at)
    for rows in cases.values():
        _metadata(rows, registry)
        base_metrics.baselines(rows)
        for category in base_metrics.CATEGORY_WEIGHTS:
            if {c["question"]["type"] for c in rows if c["category"] == category} != {
                "choice",
                "noul",
                "score",
            }:
                raise ValueError("category requires Choice, yes/no and Score coverage")
    disjoint([({}, cases)])
    _exposure(cases, history, sealed_at)
    for rows in cases.values():
        for c in rows:
            publication = history["families"].get(c["family"])
            known = publication is not None and publication["published_at"] <= sealed_at
            if c["block"] != ("known" if known else "new") or (
                known and registry[c["family"]].get("publication") != publication
            ):
                raise ValueError("known/new family tag differs from publication history")
    if blueprint(cases) != expected_blueprint:
        raise ValueError("exam differs from frozen blueprint")


def _write(root, payloads, manifest):
    root.mkdir(mode=0o700, parents=False)
    for name, value in payloads.items():
        with (root / name).open("x") as stream:
            if name.endswith(".jsonl"):
                stream.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in value))
            else:
                json.dump(value, stream, allow_nan=False, indent=2)
    manifest["files"] = {n: benchmark.file_hash(root / n) for n in sorted(payloads)}
    with (root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2)


def seal(calibration, test, registry, blueprint, *, pool_id, history, sealed_at, out):
    cases = {"calibration": calibration, "test": test}
    if pool_id not in POOLS:
        raise ValueError("unsupported text pool")
    _validate(cases, registry, blueprint, history, sealed_at)
    _write(
        Path(out),
        {
            **{f"{s}.jsonl": cases[s] for s in SPLITS},
            "registry.json": registry,
            "blueprint.json": blueprint,
            "history.json": history,
        },
        {"version": VERSION, "pool_id": pool_id, "sealed_at": sealed_at},
    )
    return {
        "manifest_sha256": benchmark.file_hash(Path(out) / "manifest.json"),
        "dataset_sha256": digest(test),
    }


def _load(root, version, files):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("exam root must be a directory without symlinks")
    manifest = json.loads(_regular(root / "manifest.json").read_text())
    if (
        manifest.get("version") != version
        or set(manifest.get("files", {})) != files
        or {p.name for p in root.iterdir()} != files | {"manifest.json"}
    ):
        raise ValueError("unsupported exam manifest/files")
    for name, value in manifest["files"].items():
        if benchmark.file_hash(_regular(root / name)) != _hash(value):
            raise ValueError("exam file differs from its manifest hash")
    return manifest


def audit(root, *, history=None, as_of=None):
    root = Path(root)
    manifest = _load(root, VERSION, FILES)
    if (
        set(manifest) != {"version", "pool_id", "sealed_at", "files"}
        or manifest["pool_id"] not in POOLS
    ):
        raise ValueError("invalid exam manifest")
    cases = {s: benchmark.read_jsonl(root / f"{s}.jsonl") for s in SPLITS}
    sealed_history = json.loads((root / "history.json").read_text())
    _validate(
        cases,
        json.loads((root / "registry.json").read_text()),
        json.loads((root / "blueprint.json").read_text()),
        sealed_history,
        manifest["sealed_at"],
    )
    if history is not None:
        validate_history(history)
        at = _time(time.time() if as_of is None else as_of)
        if at < manifest["sealed_at"]:
            raise ValueError("evaluation time predates sealing")
        # A current history may add disclosures, never erase an earlier receipt.
        for family, receipt in sealed_history["families"].items():
            if history["families"].get(family) != receipt:
                raise ValueError("publication history lost its sealed binding")
        for disclosed in sealed_history["cases"]:
            if disclosed not in history["cases"]:
                raise ValueError("publication history lost a disclosure")
        _exposure(cases, history, at)
    return manifest, cases


def seed(cases, registry, *, out):
    _metadata(cases, registry)
    disjoint([({}, {"seed": cases})])
    _write(Path(out), {"cases.jsonl": cases, "registry.json": registry}, {"version": SEED_VERSION})


def publish_history(source, *, artifact, url, published_at, history):
    """Record operator-verified public availability; this function publishes nothing."""
    validate_history(history)
    receipt = {
        "artifact_sha256": benchmark.file_hash(_regular(Path(artifact))),
        "url": url,
        "published_at": _time(published_at),
    }
    _receipt(receipt)
    root = Path(source)
    version = json.loads(_regular(root / "manifest.json").read_text())["version"]
    if version == SEED_VERSION:
        _load(root, SEED_VERSION, {"cases.jsonl", "registry.json"})
        cases = benchmark.read_jsonl(root / "cases.jsonl")
        _metadata(cases, json.loads((root / "registry.json").read_text()))
    else:
        manifest, splits = audit(root)
        if published_at < manifest["sealed_at"]:
            raise ValueError("publication time predates sealing")
        cases = sum(splits.values(), [])
    result = copy.deepcopy(history)
    existing = {c["source_id"]: c for c in result["cases"]}
    for case in cases:
        family = case["family"]
        if (
            family not in result["families"]
            or published_at < result["families"][family]["published_at"]
        ):
            result["families"][family] = receipt
        old = existing.get(case["source_id"])
        if old is None:
            old = {
                "source_id": case["source_id"],
                "fingerprints": fingerprints(case),
                "publication": receipt,
            }
            result["cases"].append(old)
            existing[case["source_id"]] = old
        else:
            old["fingerprints"] = sorted(set(old["fingerprints"]) | set(fingerprints(case)))
            if published_at < old["publication"]["published_at"]:
                old["publication"] = receipt
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("seal")
    for name in ("calibration", "test", "registry", "blueprint", "history", "out"):
        s.add_argument("--" + name, required=True, type=Path)
    s.add_argument("--pool", required=True, choices=sorted(POOLS))
    a = commands.add_parser("audit")
    a.add_argument("--round", required=True, type=Path)
    a.add_argument("--history", required=True, type=Path)
    s = commands.add_parser("seed")
    for name in ("cases", "registry", "out"):
        s.add_argument("--" + name, required=True, type=Path)
    p = commands.add_parser("publish-history")
    for name in ("source", "artifact", "history", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--url", required=True)
    p.add_argument("--published-at", required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text())

    if args.command == "seal":
        print(
            json.dumps(
                seal(
                    benchmark.read_jsonl(args.calibration),
                    benchmark.read_jsonl(args.test),
                    read(args.registry),
                    read(args.blueprint),
                    pool_id=args.pool,
                    history=read(args.history),
                    sealed_at=time.time(),
                    out=args.out,
                )
            )
        )
    elif args.command == "audit":
        print(json.dumps(audit(args.round, history=read(args.history))[0]))
    elif args.command == "seed":
        seed(benchmark.read_jsonl(args.cases), read(args.registry), out=args.out)
    else:
        timestamp = datetime.fromisoformat(args.published_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError("published-at requires UTC")
        result = publish_history(
            args.source,
            artifact=args.artifact,
            url=args.url,
            published_at=timestamp.timestamp(),
            history=read(args.history),
        )
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
