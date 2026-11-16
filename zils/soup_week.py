"""The weekly soup judge: which soup becomes the week's soup (pure functions, ZIL-204).

The nightly rounds keep building a soup (``soup_round``). Once a week the judge takes the last
nightly soup, tries to add the week's best left-out bodies, drops a member the soup is better
without, and compares the result with the current weekly soup. Every score here comes from
``evaluate(members)`` on a sealed set no nightly round draws from, so a pick that was lucky on one
night's items does not carry over.
"""

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from zils.soup_round import _EPS, MARGIN, MAX_MEMBERS, leave_one_out, seat_weights

JOIN = MARGIN  # a left-out body joins if it lifts the soup by this much, as at night
DROP = MARGIN  # a member goes if the soup is this much better without it
TRIES = 2  # left-out bodies tried each week
SEAT_BAR = 0.5  # minimum contribution (points) on the sealed set for a weekly seat
CHUNK = 5000  # sealed items graded per hold of the GPU lock, so customer grading takes turns
CAPACITY_TRIES = 120  # GPU-busy retries (one poll apart) before the judge gives up on a chunk


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
    if len(members) == 1:
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
    soup's sealed score minus the score without it (``soup_round.leave_one_out``), and a
    contribution of at least ``floor`` holds a seat. Otherwise last week's soup and seats carry
    over unchanged ("kept_from"); with no weekly soup yet, nobody is seated.
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
        kept = {"kept_from": weekly.get("kept_from", weekly["week_id"])} if weekly else {}
        return {**record, "members": weekly["members"] if weekly else [], **kept}
    members, rates = result["members"], private["rates"]
    contributions = leave_one_out(members, evaluate, baseline=stock)
    seats = seat_weights(members, contributions, {m: rates[m] for m in members}, floor=floor)
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


def sources(members, state_dir, chassis, *, rename):
    """``soup.build`` sources for a judge soup: each body's folder under the pool's state with its
    tensor rename, and the chassis itself for the release."""
    out = []
    for member in members:
        if member == RELEASE:
            out.append((Path(chassis), None))
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
    once none are left."""
    from . import d3, soup, soup_grade, soup_pool, soup_round

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
    key, checkpoint = names[0], root / "tmp" / names[0]
    if step == "build":
        partial = checkpoint.with_name(checkpoint.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        chassis = d3.pinned_base_path(model=model)
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
    and ``weeks/week.json`` under the pool's state. It gives way, to be started again later, when a
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

    decision = json.loads((state_dir / "decision.json").read_text())
    weekly = root / "week.json"
    record = run_week(
        week_id,
        json.loads((state_dir / "rounds" / decision["round_id"] / "private.json").read_text()),
        json.loads((state_dir / "pool.json").read_text())["bodies"],
        json.loads(weekly.read_text()) if weekly.exists() else None,
        cases,
        soup_grade.PredictionCache(root / "cache"),
        scope=scope["scope"],
        score=lambda cases, rows: soup_grade.score(cases, rows, scoring=scoring, secret=secret),
        bar=request["bar"],
        grade=grade,
        stop=lambda: past(stop) or _round_running(directory),
    )
    if record is None:
        print("soup week: stopped to give way; start it again to resume", flush=True)
        return None
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


def main(argv=None):
    """``--new NIGHTLY_REQUEST SEALED WEEK_ID BAR`` writes a judge request beside the pool's weeks;
    ``--run REQUEST [HH:MM]`` runs the judge until done or the UTC stop time; ``REQUEST STEP ...``
    is a GPU-runtime step (``_child``)."""
    argv = sys.argv[1:] if argv is None else argv
    if argv[0] == "--new":
        nightly = json.loads(Path(argv[1]).read_text())
        request = {
            "config": nightly["config"],
            "state_dir": nightly["state_dir"],
            "runner": nightly["runner"],
            "sealed": str(Path(argv[2]).resolve()),
            "week_id": argv[3],
            "bar": float(argv[4]),
        }
        path = Path(nightly["state_dir"]) / "weeks" / f"request-{argv[3]}.json"
        _write(path, request)
        print(path)
    elif argv[0] == "--run":
        run(argv[1], argv[2] if len(argv) > 2 else None)
    else:
        request = json.loads(Path(argv[0]).read_text())
        _child(request, Path(request["state_dir"]) / "weeks", *argv[1:])


if __name__ == "__main__":
    main()
