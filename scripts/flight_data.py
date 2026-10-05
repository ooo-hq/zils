"""Freeze a chronological, schedule-only BTS flight-delay experiment."""

import argparse
import csv
import hashlib
import heapq
import io
import json
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SEED = 553
COUNTS = {"train": 3072, "calibration": 512, "test": 1024}
MONTHS = {"train": 1, "calibration": 2, "test": 3}
SOURCE_HASHES = {
    1: "868387dcedaef1b8d8392e608e642219fde50d594c3cf2094aec858167629ccf",
    2: "12dc8dbdb3c8b3c20d4c00a8b188196ae202f781a8760871ae502a9c733d0ecd",
    3: "9c80fbc2112cdbf3f0613ec2001ba019b3ad75511cd69657080736e7eda9cef4",
}
INPUT_FIELDS = (
    "flight_date",
    "day_of_week",
    "airline",
    "flight_number",
    "origin",
    "destination",
    "scheduled_departure_local",
    "scheduled_arrival_local",
    "scheduled_duration_minutes",
    "distance_miles",
)
QUESTION = {
    "type": "choice",
    "instructions": (
        "Estimate whether this flight will arrive at least 15 minutes late. Only scheduled "
        "information is available before departure. This benchmark covers completed, "
        "non-diverted flights with recorded arrival times."
    ),
    "criteria": {
        "not_late": "Arrives less than 15 minutes late, including early or on time.",
        "late": "Arrives at least 15 minutes late.",
    },
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    )


def archive_name(month):
    return f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip"


def download(raw):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for month, checksum in SOURCE_HASHES.items():
        path = raw / archive_name(month)
        if not path.exists():
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(source_url(month), timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if digest(temporary) != checksum:
                raise ValueError("BTS archive changed; preserve it and version a new protocol")
            temporary.replace(path)
        if digest(path) != checksum:
            raise ValueError(f"Source archive hash mismatch: {path.name}")


def source_url(month):
    return "https://transtats.bts.gov/PREZIP/" + archive_name(month)


def clock_time(value):
    number = int(value)
    if number == 2400:
        return "24:00"
    hour, minute = divmod(number, 100)
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError("Invalid scheduled time")
    return f"{hour:02}:{minute:02}"


def case(row):
    """An explicit feature allowlist prevents future outcomes from entering the prompt."""
    if row["Origin"] not in {"JFK", "LGA", "EWR"}:
        return None
    if float(row["Cancelled"]) or float(row["Diverted"]) or not row["ArrDelay"]:
        return None
    flight_date = date.fromisoformat(row["FlightDate"][:10])
    label = "late" if float(row["ArrDelay"]) >= 15 else "not_late"
    if row["ArrDel15"] not in ("0", "0.00", "1", "1.00") or bool(float(row["ArrDel15"])) != (
        label == "late"
    ):
        raise ValueError("BTS delay labels disagree")
    state = dict(
        zip(
            INPUT_FIELDS,
            (
                flight_date.isoformat(),
                flight_date.strftime("%A"),
                row["Reporting_Airline"],
                row["Flight_Number_Reporting_Airline"],
                row["Origin"],
                row["Dest"],
                clock_time(row["CRSDepTime"]),
                clock_time(row["CRSArrTime"]),
                int(float(row["CRSElapsedTime"])),
                int(float(row["Distance"])),
            ),
        )
    )
    identity = [
        state[k]
        for k in (
            "flight_date",
            "airline",
            "flight_number",
            "origin",
            "destination",
            "scheduled_departure_local",
        )
    ]
    record_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return {
        "id": record_id,
        "group_id": flight_date.isoformat(),
        "family": "flight-arrival-delay",
        "state": state,
        "question": QUESTION,
        "label": label,
    }


def sample_archive(path, month, count):
    selected, seen, counts = [], set(), Counter()
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected one BTS CSV")
        with archive.open(names[0]) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")):
                counts["source_rows"] += 1
                when = date.fromisoformat(row["FlightDate"][:10])
                if (when.year, when.month) != (2025, month):
                    raise ValueError("Archive contains a flight outside its declared month")
                if row["Origin"] not in {"JFK", "LGA", "EWR"}:
                    continue
                counts["nyc_departures"] += 1
                item = case(row)
                if item is None:
                    counts["excluded_cancelled_diverted_or_missing_arrival"] += 1
                    continue
                if item["id"] in seen:
                    raise ValueError("Duplicate flight identity")
                seen.add(item["id"])
                counts["eligible"] += 1
                # Identity-only deterministic sampling: independent of labels and source ordering.
                rank = int(hashlib.sha256(f"{SEED}:{item['id']}".encode()).hexdigest(), 16)
                entry = (-rank, item["id"], item)
                if len(selected) < count:
                    heapq.heappush(selected, entry)
                elif entry > selected[0]:
                    heapq.heapreplace(selected, entry)
    if len(selected) != count:
        raise ValueError("Insufficient eligible flights")
    return [entry[2] for entry in sorted(selected, reverse=True)], dict(counts)


def history_key(row):
    state = row["state"]
    hour = int(state["scheduled_departure_local"].split(":")[0]) % 24
    return "|".join([state["airline"], state["origin"], str(hour // 4)])


def fit_history(rows):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        counts = groups[history_key(row)]
        counts[0] += row["label"] == "late"
        counts[1] += 1
    return {
        "global_rate": (sum(r["label"] == "late" for r in rows) + 1) / (len(rows) + 2),
        "prior_strength": 20,
        "groups": dict(groups),
        "training_count": len(rows),
    }


def history_probability(history, row):
    positives, count = history["groups"].get(history_key(row), [0, 0])
    prior = history["prior_strength"]
    return (positives + prior * history["global_rate"]) / (count + prior)


def prepare(raw, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve frozen data; use a new output directory")
    splits, sources = {}, {}
    for split, month in MONTHS.items():
        path = Path(raw) / archive_name(month)
        if digest(path) != SOURCE_HASHES[month]:
            raise ValueError("Source archive differs from the pinned public data")
        splits[split], statistics = sample_archive(path, month, COUNTS[split])
        sources[split] = {"url": source_url(month), "sha256": digest(path), **statistics}
    output.mkdir(parents=True, mode=0o700)
    protocol = {
        "version": "zils-flight-delay-001",
        "seed": SEED,
        "sample_counts": COUNTS,
        "months_2025": MONTHS,
        "origins": ["EWR", "JFK", "LGA"],
        "input_fields": INPUT_FIELDS,
        "question": QUESTION,
        "sampling": "Smallest SHA256(seed:flight-identity), no label balancing",
        "population": "Completed, non-diverted NYC departures with observed arrival delay",
        "training": {
            "epochs": 1,
            "learning_rate": 2e-5,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "gradient_accumulation": 4,
            "seed": SEED,
        },
        "calibration": "Separate NLL temperature fit for base and adapter: 81 log-spaced values in [0.25,4]",
        "historical_baseline": "Same training sample; airline/origin/4-hour departure bin; 20-observation global prior; global Beta(1,1) smoothing",
        "primary_metric": "Mean binary Brier: mean((P(late)-observed_late)^2); lower is better",
        "uncertainty": "2000 paired bootstrap resamples of whole test dates, seed 553",
        "success": "Adapter beats equally calibrated base and historical baseline on test Brier; both paired date-bootstrap 95% intervals exclude zero",
        "promotion": "Research only; no automatic model registration or serving changes",
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "historical.json", fit_history(splits["train"]))
    for split, rows in splits.items():
        write_rows(output / f"{split}.jsonl", rows)
        write_rows(
            output / f"{split}-inputs.jsonl",
            [{k: r[k] for k in ("id", "state", "question")} for r in rows],
        )
    write_rows(
        output / "train-export.jsonl",
        [
            {"state": r["state"], "questions": {"decision": {**r["question"], "label": r["label"]}}}
            for r in splits["train"]
        ],
    )
    write_json(
        output / "manifest.json",
        {
            "sources": sources,
            "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
            "splits": {
                k: {
                    "count": len(v),
                    "late": sum(r["label"] == "late" for r in v),
                    "first_date": min(r["group_id"] for r in v),
                    "last_date": max(r["group_id"] for r in v),
                }
                for k, v in splits.items()
            },
        },
    )
    print(json.dumps(json.loads((output / "manifest.json").read_text())["splits"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.raw)
    prepare(args.raw, args.out)


if __name__ == "__main__":
    main()
