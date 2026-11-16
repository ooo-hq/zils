"""The weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204).

The nightly rounds keep building a soup (``soup_round``). Once a week the judge takes the last
nightly soup, tries to add the week's best left-out bodies, drops a member the soup is better
without, and compares the result with the current weekly soup. Every score here comes from
``evaluate(members)`` on a sealed set no nightly round draws from, so a pick that was lucky on one
night's items does not carry over.
"""

import datetime
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from zils.soup_round import (
    _EPS,
    HOUSE_RATE,
    MARGIN,
    MAX_MEMBERS,
    base_contributions,
    seat_weights,
)

JOIN = MARGIN  # a left-out body joins if it lifts the soup by this much, as at night
DROP = MARGIN  # a member goes if the soup is this much better without it
TRIES = 2  # left-out bodies tried each week
SEAT_BAR = 0.5  # minimum contribution (points) on the sealed set for a weekly seat
CHUNK = 5000  # sealed items graded per hold of the GPU lock, so customer grading takes turns
CAPACITY_TRIES = 120  # GPU-busy retries (one poll apart) before the judge gives up on a chunk
JUDGE_WEEKDAY = 0  # the judge takes the nightly round whose day is a Monday
MAX_STARTS = 5  # starts of one week's judge before the week is skipped (it may give way and resume)
MAX_WEEK_DAYS = 14  # a week's seats are paid this long after its round; then they burn


def add_left_out(members, left_out, evaluate, *, margin=JOIN, tries=TRIES, max_members=MAX_MEMBERS):
    """Try to add up to ``tries`` of ``left_out`` (best first) to the soup. Returns (members, trace).

    Each body is tried on the soup as it stands after the tries before it and joins only if the
    soup's score rises by at least ``margin``. A full soup takes no more. Trace rows are
    ``soup_round.greedy``'s: {"try", "score", "best_before", "accepted"}.
    """
    soup, best, trace = tuple(sorted(members)), None, []
    for body in [b for b in left_out if b not in soup][:tries]:
        if len(soup) >= max_members:
            break
        best = evaluate(soup) if best is None else best
        trial = tuple(sorted((*soup, body)))
        score = evaluate(trial)
        accepted = score - best >= margin - _EPS
        trace.append(
            {"try": list(trial), "score": score, "best_before": best, "accepted": accepted}
        )
        if accepted:
            soup, best = trial, score
    return soup, trace


def drop_members(members, evaluate, *, margin=DROP):
    """Drop members the soup is better without. Returns (members, trace).

    Each pass scores the soup without each member and removes the one whose removal helps most,
    if the soup rises by at least ``margin``; it stops at the first pass that removes nothing.
    The last member is never dropped. Trace rows are {"drop", "score", "best_before", "accepted"}.
    """
    soup, trace = tuple(sorted(members)), []
    while len(soup) > 1:
        best = evaluate(soup)
        score, body = max((evaluate(tuple(m for m in soup if m != b)), b) for b in soup)
        accepted = score - best >= margin - _EPS
        trace.append({"drop": body, "score": score, "best_before": best, "accepted": accepted})
        if not accepted:
            break
        soup = tuple(m for m in soup if m != body)
    return soup, trace


def judge(candidate, left_out, evaluate, *, current, bar, stock=None, **settings):
    """The week's soup. ``candidate`` is the last nightly soup, ``left_out`` the week's bodies outside
    it (best first), ``current`` the weekly soup now serving (None in the first week, when ``stock``,
    the release's own score, stands in for it).

    The candidate, after ``add_left_out`` and ``drop_members``, replaces the current soup only if it
    scores at least ``bar`` above it. Returns {"members", "candidate", "score", "current_score",
    "passed", "trace"}; ``members`` is the soup for the coming week (None: the release stays).
    """
    soup, trace = add_left_out(candidate, left_out, evaluate, **settings)
    soup, dropped = drop_members(soup, evaluate)
    score = evaluate(soup)
    current = tuple(sorted(current)) if current else None
    if current is None and stock is None:
        raise ValueError("the first weekly soup needs the stock score")
    current_score = stock if current is None else evaluate(current)
    passed = soup != current and score - current_score >= bar - _EPS
    return {
        "members": soup if passed else current,
        "candidate": soup,
        "score": score,
        "current_score": current_score,
        "passed": passed,
        "trace": trace + dropped,
    }


def left_out(private):
    """The bodies the judge may try: in the last nightly round's pool, outside its soup, and paid
    (a house body is never added by the judge). Best solo score first, ties by hash."""
    members = {m["body_sha256"] for m in private["members"]}
    solo, rates = private["solo"], private["rates"]
    bodies = [b for b in private["pool"] if b not in members and b in solo and rates.get(b, 0) > 0]
    return sorted(bodies, key=lambda b: (-solo[b], b))


class Missing(Exception):
    """A soup the judge needs has sealed items with no cached answer yet."""

    def __init__(self, key, members):
        super().__init__(key)
        self.key, self.members = key, members


def soup_key(members, scope):
    """The prediction-cache key of a soup of ``members``, as ``soup_round.run`` names it."""
    members = tuple(sorted(members))
    if len(members) == 1 and not members[0].startswith("admission-"):
        return f"{members[0][:32]}-{scope}"
    return f"soup-{hashlib.sha256('+'.join(members).encode()).hexdigest()[:32]}-{scope}"


def cached_scores(cases, cache, *, scope, score):
    """``evaluate(members)`` for the judge, from cached answers only: ``score(cases, rows)`` of the
    soup's rows on the sealed ``cases``, or ``Missing`` when any case is not graded yet. The caller
    grades what is missing and asks again, so the GPU is held one model at a time."""
    ids, scores = [c["id"] for c in cases], {}

    def evaluate(members):
        members = tuple(sorted(members))
        if members not in scores:
            key = soup_key(members, scope)
            rows = cache.get(key, ids)
            if len(rows) < len(ids):
                raise Missing(key, members)
            scores[members] = score(cases, [rows[i] for i in ids])
        return scores[members]

    return evaluate


def decide(
    week_id, private, known, weekly, evaluate, *, bar, stock=None, floor=SEAT_BAR, **settings
):
    """The week's record. ``private`` is the last nightly round's private record, ``known`` the
    pool's {body: {"hotkey"}}, ``weekly`` last week's record (None before the first weekly soup).

    A candidate that passes (``judge``) is the new weekly soup: each member's contribution is the
    soup's sealed score minus the score without it, or for a fine-tune (the round's
    ``fine_tunes``) minus the score with its base in its place (``soup_round.base_contributions``),
    and a contribution of at least ``floor`` holds a seat. A fine-tune whose base the round could
    not build is "gated" and holds none. Otherwise last week's soup and seats carry over unchanged
    ("kept_from"); with no weekly soup yet, nobody is seated.
    """
    candidate = tuple(sorted(m["body_sha256"] for m in private["members"]))
    current = tuple(m["body_sha256"] for m in weekly["members"]) if weekly else None
    result = judge(
        candidate, left_out(private), evaluate, current=current or None, bar=bar, stock=stock,
        **settings,
    )  # fmt: skip
    record = {
        "week_id": week_id,
        "round_id": private["round_id"],
        "passed": result["passed"],
        "candidate": list(result["candidate"]),
        "score": result["score"],
        "current_score": result["current_score"],
        "trace": result["trace"],
    }
    if not result["passed"]:
        if not weekly:
            return {**record, "members": []}
        kept = {k: weekly[k] for k in ("soup_sha256", "temperature") if k in weekly}
        kept["kept_from"] = weekly.get("kept_from", weekly["week_id"])
        return {**record, "members": weekly["members"], **kept}
    members, rates = result["members"], private["rates"]
    contributions, unscored = base_contributions(
        members, evaluate, private.get("fine_tunes", {}), base_score=evaluate, baseline=stock
    )
    paid = {**{m: rates[m] for m in members}, **dict.fromkeys(unscored, HOUSE_RATE)}
    seats = seat_weights(members, contributions, paid, floor=floor)
    record["gated"] = dict.fromkeys(sorted(unscored), "its base could not be scored")
    record["members"] = [
        {
            "hotkey": known[m]["hotkey"],
            "body_sha256": m,
            "rate": rates[m],
            "contribution": contributions[m],
            "seat_weight": seats.get(m, 0.0),
        }
        for m in members
    ]
    return record


def soup_folder(root, record):
    """Where the files of ``record``'s soup are kept under the pool's ``weeks`` folder: the week
    that made it saved it (``_child`` "keep"). None when the week has no soup."""
    if not record["members"]:
        return None
    return Path(root) / record.get("kept_from", record["week_id"]) / "soup"


RELEASE = "release"  # the pool's served model as a judge "member": its score is the stock score


def run_week(week_id, private, known, weekly, cases, cache, *, scope, score, bar, grade, stop, **settings):  # fmt: skip
    """Decide the week, grading what the judge still needs one model at a time. Returns the record
    (``decide``), or None when ``stop()`` says to give way (a nightly round, the end of the window);
    a later call resumes from the cache.

    ``grade(key, members)`` grades a chunk of the sealed items ``members``' soup has no answer for
    and stores them in ``cache`` under ``key``. It is called until the soup is complete, with
    ``stop`` checked before each call, so the GPU is never held for more than a chunk."""
    ids = [c["id"] for c in cases]
    evaluate = cached_scores(cases, cache, scope=scope, score=score)
    while not stop():
        try:
            stock = evaluate((RELEASE,))
            return decide(week_id, private, known, weekly, evaluate, bar=bar, stock=stock, **settings)  # fmt: skip
        except Missing as missing:
            before = len(cache.get(missing.key, ids))
            grade(missing.key, missing.members)
            if len(cache.get(missing.key, ids)) <= before:
                raise RuntimeError(f"grading {missing.key} made no progress") from None
    return None


class Reread:
    """The prediction cache under ``root``, read from disk on every ``get``: the grading child
    writes it in another process, and a ``PredictionCache`` keeps what it first read."""

    def __init__(self, root):
        self.root = root

    def get(self, key, ids):
        from . import soup_grade

        return soup_grade.PredictionCache(self.root).get(key, ids)


def sources(members, state_dir, chassis, *, rename):
    """``soup.build`` sources for a judge soup, as the round's: each body's folder under the pool's
    state with its tensor rename, the chassis itself for the release, and the kept copy of an
    admission soup a fine-tune is credited against."""
    out = []
    for member in members:
        if member == RELEASE:
            out.append((Path(chassis), None))
        elif member.startswith("admission-"):
            out.append((Path(state_dir) / "admission" / member.removeprefix("admission-"), None))
        else:
            folder = Path(state_dir) / "bodies" / member
            out.append((folder, rename(folder)))
    return out


def past(stop, now=None):
    """Whether the UTC clock time ``stop`` ("HH:MM", None for no limit) has been reached today."""
    if not stop:
        return False
    hour, minute = map(int, stop.split(":"))
    return (now or time.gmtime()[3:5]) >= (hour, minute)


def _write(path, value, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n"
    )
    if mode is not None:
        os.chmod(partial, mode)
    os.replace(partial, path)


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _child(request, root, step, *names):
    """One step of the judge in the GPU runtime (no launcher imports torch): ``cases`` fits the
    sealed set to the pool's token cap and names the cache scope, ``build`` writes a soup's
    checkpoint, ``grade`` answers its next ``CHUNK`` ungraded items and removes the checkpoint
    once none are left, ``keep`` saves the week's soup at its fitted temperature."""
    from . import d3, models, soup, soup_grade, soup_pool, soup_round

    model, state_dir = request["config"]["model"]["id"], Path(request["state_dir"])
    if step == "cases":
        sealed = soup_pool._bank(request["sealed"])
        cases = soup_pool.fit_bank(sealed, model)
        key = soup_round.bank_sha256(sealed)
        scope = f"{key[:16]}-{model}-{d3.pins(model)['files'][soup.READOUT][:12]}"
        _write(
            root / "cases.jsonl",
            "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases),
            0o600,
        )
        _write(root / "scope.json", {"scope": scope, "sealed_sha256": key, "items": len(cases)})
        return
    chassis = d3.pinned_base_path(model=model)
    if step == "keep":  # WEEK_ID TEMPERATURE MEMBER...: the soup as served, like a round's
        final, partial = root / names[0] / "soup", root / "tmp" / f"keep-{names[0]}"
        if not final.exists():
            shutil.rmtree(partial, ignore_errors=True)
            partial.parent.mkdir(parents=True, exist_ok=True)
            bodies = sources(names[2:], state_dir, chassis, rename=soup_round._rename)
            soup.build(partial, chassis, bodies, model=model, temperature=float(names[1]))
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, final)
        kept = {"soup_sha256": soup.soup_hash(final), "temperature": models.temperature(final)}
        _write(root / names[0] / "soup.json", kept)
        return
    key, checkpoint = names[0], root / "tmp" / names[0]
    if step == "build":
        partial = checkpoint.with_name(checkpoint.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        soup.build(partial, chassis, sources(names[1:], state_dir, chassis, rename=soup_round._rename), model=model)  # fmt: skip
        os.replace(partial, checkpoint)
        return
    cases, cache = _lines(root / "cases.jsonl"), soup_grade.PredictionCache(root / "cache")
    have = cache.get(key, [c["id"] for c in cases])
    todo = [c for c in cases if c["id"] not in have]
    soup_grade.grade(key, todo[:CHUNK], cache, soup_grade.production_runner(checkpoint, request["runner"]["device"]))  # fmt: skip
    if len(todo) <= CHUNK:
        shutil.rmtree(checkpoint, ignore_errors=True)


def _round_running(directory):
    """A nightly soup round of any pool under the validator ``directory`` is running now."""
    from . import soup_pool

    for path in Path(directory, "base-pools").glob("*/soup/job.json"):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        pool = path.parent.parent.name
        if soup_pool._alive(job.get("pid"), job.get("pid_started")) and not soup_pool.recorded(
            directory, pool, job.get("round_id")
        ):
            return True
    return False


def run(path, stop=None):
    """The weekly judge for the request at ``path`` (validator python). Grades on the sealed set
    under the shared compute lock, a chunk at a time, and writes ``weeks/<week_id>/record.json``
    and ``weeks/week.json`` under the pool's state, with a new weekly soup's files beside the
    record (``soup_folder``). It gives way, to be started again later, when a
    nightly round runs or the UTC time ``stop`` is reached. Nothing here sets weights: the record
    is read by whoever pays on it."""
    from . import runtime, soup_grade, soup_round

    request = json.loads(Path(path).read_text())
    config, state_dir, runner = request["config"], Path(request["state_dir"]), request["runner"]
    root, week_id = state_dir / "weeks", request["week_id"]
    done = root / week_id / "record.json"
    if done.exists():
        return json.loads(done.read_text())
    log = root / "week.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def child(*step):
        return [runner["runtime_python"], "-m", "zils.soup_week", str(path), *step]

    with log.open("ab") as output:
        subprocess.run(child("cases"), stdout=output, stderr=subprocess.STDOUT, check=True)
    scope = json.loads((root / "scope.json").read_text())
    cases = _lines(root / "cases.jsonl")
    secret = hmac.new(
        soup_round.sketch_seed(config, state_dir),
        b"weekly-split:" + scope["sealed_sha256"].encode(),
        hashlib.sha256,
    ).digest()  # one split per sealed set, so every week is scored on the same items
    scoring = config.get("scoring", soup_grade.DEFAULT_SCORING)
    directory = state_dir.parent.parent.parent  # <validator>/base-pools/<pool>/soup

    def grade(key, members):
        if not (root / "tmp" / key).exists():
            with log.open("ab") as output:
                subprocess.run(child("build", key, *members), stdout=output, stderr=subprocess.STDOUT, check=True)  # fmt: skip
        for _ in range(CAPACITY_TRIES):
            try:
                started = time.time()
                runtime.run_child(child("grade", key), log, runner["device"], timeout=3600, model=config["model"]["id"])  # fmt: skip
                print(f"soup week: graded a chunk of {key} in {time.time() - started:.0f} s", flush=True)  # fmt: skip
                return
            except runtime.CapacityUnavailable:
                time.sleep(max(1, runner["poll"]))
        raise RuntimeError("the weekly judge found no GPU capacity")

    weekly, cache = root / "week.json", Reread(root / "cache")
    record = run_week(
        week_id,
        json.loads((state_dir / "rounds" / request["round_id"] / "private.json").read_text()),
        json.loads((state_dir / "pool.json").read_text())["bodies"],
        json.loads(weekly.read_text()) if weekly.exists() else None,
        cases,
        cache,
        scope=scope["scope"],
        score=lambda cases, rows: soup_grade.score(cases, rows, scoring=scoring, secret=secret),
        bar=request["bar"],
        grade=grade,
        stop=lambda: past(stop) or _round_running(directory),
    )
    if record is None:
        print("soup week: stopped to give way; start it again to resume", flush=True)
        return None
    if record["passed"]:  # save the new soup, served at the temperature its sealed answers fit
        ids = [c["id"] for c in cases]
        rows = cache.get(soup_key(record["candidate"], scope["scope"]), ids)
        fitted = soup_grade.evaluate(cases, [rows[i] for i in ids], scoring=scoring, secret=secret)
        with log.open("ab") as output:
            subprocess.run(child("keep", week_id, repr(fitted["temperature"]), *record["candidate"]), stdout=output, stderr=subprocess.STDOUT, check=True)  # fmt: skip
        record.update(json.loads((root / week_id / "soup.json").read_text()))
    record = {**record, "pool_id": config["pool_id"], **scope}
    _write(done, record)
    _write(weekly, record)
    seated = sum(1 for m in record["members"] if m["seat_weight"] > 0)
    print(
        f"soup week {week_id}: {'passed' if record['passed'] else 'kept the current soup'}, "
        f"{record['score']:.2f} against {record['current_score']:.2f}, "
        f"{len(record['members'])} members, {seated} seated",
        flush=True,
    )
    return record


def week_of(day):
    """(week id, judge day) for a nightly round's ``day`` (an ISO date): a week is named after the
    judge weekday at or before that day."""
    date = datetime.date.fromisoformat(day)
    judged = date - datetime.timedelta(days=(date.weekday() - JUDGE_WEEKDAY) % 7)
    year, week, _ = judged.isocalendar()
    return f"{year}-w{week:02d}", judged.isoformat()


def weeks(config, directory):
    """The validator's local ``soup_weeks`` setting, {pool: {"sealed": the sealed set's path,
    resolved against the validator ``directory``, "bar": points a new soup must gain on the current
    weekly soup}}. Without it no pool has a weekly judge."""
    value = config.get("soup_weeks", {})
    if not isinstance(value, dict):
        raise ValueError("soup_weeks must map pools to their sealed set and bar")
    out = {}
    for pool, entry in value.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sealed", "bar"}
            or not isinstance(entry["sealed"], str)
            or not entry["sealed"]
            or type(entry["bar"]) not in (int, float)
            or entry["bar"] < 0
        ):
            raise ValueError("a soup_weeks entry needs sealed (a path) and bar (0 or more)")
        sealed = str((Path(directory) / entry["sealed"]).resolve())
        out[pool] = {"sealed": sealed, "bar": float(entry["bar"])}
    return out


def public_week(record):
    """The week's public part, which pay reads: its soup and seats, contributions to the nearest
    half point as in a round's public record. Sealed-set scores and the trace stay private."""
    from .soup_publish import _half_point

    return {
        "week_id": record["week_id"],
        "round_id": record["round_id"],
        "soup_sha256": record.get("soup_sha256"),
        "kept_from": record.get("kept_from"),
        "members": [
            {
                **{k: m[k] for k in ("hotkey", "body_sha256", "rate", "seat_weight")},
                "contribution": _half_point(m["contribution"]),
            }
            for m in record["members"]
        ],
    }


def closed_seats(week, bodies, listed, before):
    """The week's paid seats whose body is no longer public (ZIL-205): {"week_id", "bodies"}.

    ``bodies`` is the pool's {body: {"repo", "revision"}}, ``listed(repo, revision)`` lists the
    committed revision (``commitments.revision_files``), and ``before`` is the last night's result.
    A revision that is private (401, 403), gone (404) or no longer that repo's commit closes the seat
    for the rest of the week; a network failure leaves it open. A new week starts with none closed."""
    import urllib.error

    week_id = week["week_id"]
    closed = set(before["bodies"]) if before and before.get("week_id") == week_id else set()
    for member in week["members"]:
        body, info = member["body_sha256"], bodies.get(member["body_sha256"])
        if member["seat_weight"] <= 0 or body in closed or info is None:
            continue
        try:
            listed(info["repo"], info["revision"])
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                closed.add(body)
        except ValueError:
            closed.add(body)
        except OSError:
            pass
    return {"week_id": week_id, "bodies": sorted(closed)}


def attach(config, directory, pool, decision, *, now):
    """``decision`` (the pool's nightly public record) with the current week's public part added as
    "week", for ``soup_round.soup_weights``. Unchanged for a pool without a weekly judge or before
    its first week. A week whose round is more than ``MAX_WEEK_DAYS`` old seats nobody."""
    if pool not in weeks(config, directory):
        return decision
    path = Path(directory) / "base-pools" / pool / "soup" / "weeks" / "week.json"
    if not path.exists():
        return decision
    week = public_week(json.loads(path.read_text()))
    day = datetime.datetime.fromisoformat(week["round_id"][-10:]).replace(tzinfo=datetime.UTC)
    if now - day.timestamp() > MAX_WEEK_DAYS * 86400:
        week["members"] = []
    return {**decision, "week": week}


def new_request(nightly, sealed, week_id, bar):
    """The judge's request for the nightly round whose request is ``nightly``: that round's pool
    config, state and runner, pinned to its ``round_id``."""
    return {
        "config": nightly["config"],
        "state_dir": nightly["state_dir"],
        "runner": nightly["runner"],
        "round_id": nightly["round_id"],
        "sealed": str(Path(sealed).resolve()),
        "week_id": week_id,
        "bar": float(bar),
    }


def _judge_running(directory):
    """The pool whose weekly judge is running now under the validator ``directory``, or None."""
    from . import soup_pool

    for path in sorted(Path(directory, "base-pools").glob("*/soup/weeks/job.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if soup_pool._alive(job.get("pid"), job.get("pid_started")):
            return path.parent.parent.parent.name
    return None


def epoch(config, directory, *, start):
    """Start each ``soup_weeks`` pool's weekly judge when it is due; never waits for one. Returns
    {pool: summary}.

    The judge is due once the pool's latest finished nightly round is a Monday's and that week has
    no record. ``start(pool, request path)`` starts it and returns its process id (or None). It runs
    one pool at a time and never beside a nightly round; one that gave way or died is started again
    on a later pass, up to ``MAX_STARTS`` times. If a later round finishes first the week is missed
    and the current weekly soup stays, as it does after any failure reported here."""
    from . import soup_pool

    try:
        pools = weeks(config, directory)
    except ValueError as error:
        return {"error": f"{type(error).__name__}: {error}"[:400]}
    summary = {}
    for pool, entry in pools.items():
        try:
            latest = soup_pool.decision(directory, pool)
            if latest is None:
                continue
            round_id, state = latest["round_id"], soup_pool.state_dir(directory, pool)
            week_id, day = week_of(round_id.removeprefix(f"{pool}-"))
            summary[pool], root = {"week_id": week_id}, state / "weeks"
            if (root / week_id / "record.json").exists():
                summary[pool]["recorded"] = True
                continue
            job = (
                json.loads((root / "job.json").read_text()) if (root / "job.json").exists() else {}
            )
            starts = job.get("starts", 0) if job.get("week_id") == week_id else 0
            if starts and soup_pool._alive(job.get("pid"), job.get("pid_started")):
                summary[pool]["running"] = True
            elif round_id != f"{pool}-{day}":
                summary[pool]["missed"] = "a later round finished before the judge did"
            elif not soup_pool.recorded(directory, pool, round_id):
                summary[pool]["waiting"] = "the round is not recorded yet"
            elif starts >= MAX_STARTS:
                raise RuntimeError(f"week skipped after {starts} starts")
            elif _round_running(directory):
                summary[pool]["waiting"] = "a nightly round is running"
            elif other := _judge_running(directory):
                summary[pool]["waiting"] = f"the {other} judge is running"
            else:
                nightly = json.loads((state / f"request-{round_id}.json").read_text())
                path = root / f"request-{week_id}.json"
                _write(path, new_request(nightly, entry["sealed"], week_id, entry["bar"]))
                job = {"week_id": week_id, "starts": starts + 1}
                _write(root / "job.json", {**job, "pid": None})  # a start counts even if it fails
                pid = start(pool, path)
                started = soup_pool._started(pid) if pid else None
                _write(root / "job.json", {**job, "pid": pid, "pid_started": started})
                summary[pool]["started"] = True
        except Exception as error:  # one pool's judge must never stop the validator's weights
            summary.setdefault(pool, {})["error"] = f"{type(error).__name__}: {error}"[:400]
    return summary


def background():
    """A ``start`` for the validator loop: a detached ``python -m zils.soup_week --run <request>``
    (``run``), its output in the pool's ``weeks/week.log``. Returns its pid."""
    from . import ROOT

    def start(pool, path):
        with (Path(path).parent / "week.log").open("ab") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "zils.soup_week", "--run", str(path)],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT,
            )
        return process.pid

    return start


def main(argv=None):
    """``--new NIGHTLY_REQUEST SEALED WEEK_ID BAR`` writes a judge request beside the pool's weeks;
    ``--run REQUEST [HH:MM]`` runs the judge until done or the UTC stop time; ``REQUEST STEP ...``
    is a GPU-runtime step (``_child``)."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[0] == "--new":
        nightly = json.loads(Path(argv[1]).read_text())
        path = Path(nightly["state_dir"]) / "weeks" / f"request-{argv[3]}.json"
        _write(path, new_request(nightly, *argv[2:5]))
        print(path)
    elif argv[0] == "--run":
        run(argv[1], argv[2] if len(argv) > 2 else None)
    else:
        request = json.loads(Path(argv[0]).read_text())
        _child(request, Path(request["state_dir"]) / "weeks", *argv[1:])


if __name__ == "__main__":
    main()
