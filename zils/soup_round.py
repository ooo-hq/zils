"""Soup rounds: greedy soup selection, leave-one-out contributions, seat and crown pay.

Each round, validators build a fresh uniform-weight average ("soup") of up to five model bodies from the
round's pool, chosen greedily (Wortsman et al. 2022, with a minimum gain per added member). The pool's
emission share splits 80% among soup seats and 20% to the round's best single body (the crown); any share
nobody earns burns. The selection and pay functions are pure: grading is passed in as ``evaluate``.
``run`` orchestrates one round on a validator; its model, download and chain dependencies are injected.

A body is any hashable id (a body sha256 in production). ``evaluate(members)`` grades the uniform soup of a
sorted tuple of bodies and returns its exam score in points.
"""

import hashlib
import http.client
import json
import math
import os
import re
import shutil
from pathlib import Path

MARGIN = 0.3  # minimum soup score gain (points) for greedy to add a body; above GPU-to-GPU drift
MAX_MEMBERS = 5
SEAT_SHARE = 0.8
CROWN_SHARE = 0.2
FLOOR = 0.3  # minimum leave-one-out contribution (points) for a paid seat
HOUSE_RATE = 0.0  # open models known at launch: seated but unpaid
SCOUT_RATE = 0.25  # open models released after launch, submitted by a scout
TRAINER_RATE = 1.0
RATES = (HOUSE_RATE, SCOUT_RATE, TRAINER_RATE)
_EPS = 1e-9  # score differences are floats; 3.3 - 3.0 must clear a 0.3 margin


def greedy(solo, evaluate, *, margin=MARGIN, max_members=MAX_MEMBERS, order_key):
    """Greedy soup over ``solo`` ({body: solo score}). Returns (members, trace).

    Bodies are tried in solo-score order, ties broken by ``order_key(body)`` (earlier commitment first).
    The soup starts as the best body alone, which is not re-evaluated; each next body joins only if the
    soup's score rises by at least ``margin``. Selection stops once the soup has ``max_members``.
    ``members`` is a sorted tuple; each trace row is {"try", "score", "best_before", "accepted"}.
    """
    if not solo:
        raise ValueError("greedy soup needs at least one body")
    if type(max_members) is not int or max_members < 1:
        raise ValueError("max_members must be a positive integer")
    order = sorted(solo, key=lambda body: (-solo[body], order_key(body)))
    soup, best, trace = (order[0],), solo[order[0]], []
    for body in order[1:]:
        if len(soup) >= max_members:
            break
        trial = tuple(sorted((*soup, body)))
        score = evaluate(trial)
        accepted = score - best >= margin - _EPS
        trace.append(
            {"try": list(trial), "score": score, "best_before": best, "accepted": accepted}
        )
        if accepted:
            soup, best = trial, score
    return soup, trace


def leave_one_out(members, evaluate, *, baseline=None):
    """Each member's contribution: the soup's score minus the score of the soup without that member.

    A one-member soup has no smaller soup, so its contribution is its score minus ``baseline``, the
    chassis (stock) score, which the caller must pass in that case.
    """
    members = tuple(sorted(members))
    if not members:
        raise ValueError("leave-one-out needs at least one member")
    full = evaluate(members)
    if len(members) == 1:
        if baseline is None:
            raise ValueError("a one-member soup needs the chassis baseline score")
        return {members[0]: full - baseline}
    return {m: full - evaluate(tuple(b for b in members if b != m)) for m in members}


def _check_rate(rate):
    if rate not in RATES:
        raise ValueError("rate must be one of " + ", ".join(map(str, RATES)))
    return rate


def seat_weights(members, contributions, rates, *, share=SEAT_SHARE, floor=FLOOR):
    """Seat pay as fractions of the pool's share.

    Members whose contribution is at least ``floor`` split ``share`` equally; each slice is multiplied by
    the member's rate, so a house ingredient keeps its seat at 0.0 and a scout gets a quarter. Members
    below the floor are omitted. Whatever is not paid burns; if nobody clears the floor the result is {}.
    """
    paid = [m for m in members if contributions[m] >= floor - _EPS]
    return {m: share / len(paid) * _check_rate(rates[m]) for m in paid}


def crown_weight(solo, rates, house, *, share=CROWN_SHARE, order_key):
    """The crown: the best solo body that is not a house ingredient gets ``share`` times its rate.

    Ties go to the earlier commitment (``order_key``). With no eligible body the result is {} and the
    crown share burns.
    """
    eligible = [body for body in solo if body not in house]
    if not eligible:
        return {}
    best = min(eligible, key=lambda body: (-solo[body], order_key(body)))
    return {best: share * _check_rate(rates[best])}


def _weight(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("soup weights must be finite and non-negative")
    return value


def soup_weights(decision, members):
    """{uid: weight} for one soup pool, from a round record and the metagraph roster.

    ``decision`` is {"pool_id", "members": [{"hotkey", "body_sha256", "rate", "contribution",
    "seat_weight"}], "crown": {"hotkey", "body_sha256", "rate", "weight"} | None}; ``members`` is
    {uid: {"hotkey": ...}}. Seat and crown weights are summed per hotkey; hotkeys no longer registered
    are dropped, so their share burns. Weights are fractions of the pool's share and total at most one;
    the caller burns the remainder.
    """
    if not isinstance(decision, dict) or not isinstance(decision.get("members"), list):
        raise ValueError("soup decision needs a members list")
    rows = [(row["hotkey"], _weight(row["seat_weight"])) for row in decision["members"]]
    crown = decision.get("crown")
    if crown is not None:
        rows.append((crown["hotkey"], _weight(crown["weight"])))
    if sum(w for _, w in rows) > 1 + _EPS:
        raise ValueError("soup decision pays more than the pool's share")
    uid_by_hotkey = {item["hotkey"]: int(uid) for uid, item in members.items()}
    result = {}
    for hotkey, weight in rows:
        if hotkey in uid_by_hotkey and weight > 0:
            uid = uid_by_hotkey[hotkey]
            result[uid] = result.get(uid, 0.0) + weight
    return result


# ---- Round orchestration -------------------------------------------------------------------------

SUBSET_SIZE = (
    3000  # round items graded per body; drawn fresh from the private round bank each round
)
SCREEN_SIZE = 300  # items a new body is screened on before it can join the pool
MAX_POOL = 20  # bodies graded alone and offered to the greedy soup each round
MAX_NEW_PER_ROUND = (
    6  # new bodies admitted per round (disk and download time); the rest wait a round
)
PRUNE_ROUNDS = 3  # a body out of the pool this many rounds loses its folder (hash and sketch kept)
SOUP_CACHE_ROUNDS = 3  # a soup's cached predictions are dropped after this many rounds unused
_ROUND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def round_seed(block_hash, pool_id):
    """The round's seed: sha256 of the seed block's hash and the pool id."""
    return hashlib.sha256(f"{block_hash}:{pool_id}".encode()).digest()


def bank_sha256(bank):
    """Commitment to the round bank's content, so cached rows are never shared between banks."""
    from .soup_publish import canonical_json

    return hashlib.sha256(canonical_json(sorted(bank, key=lambda c: c["id"]))).hexdigest()


def default_rate(sha256, info, house):
    """House ingredients are seated unpaid; every other body is paid as a trainer's.

    The scout/trainer rate (``soup_rates.rate``) replaces this through ``run(rates=...)``.
    """
    return HOUSE_RATE if sha256 in house else TRAINER_RATE


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def _read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _reason(error):
    """A rejection reason: a ValueError's message, or any other exception's type and message."""
    text = str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"
    return text[:400]


def _rename(folder):
    from . import bodies

    return "strip-model" if any(k.startswith("model.") for k in bodies.load_index(folder)) else None


def _house(config, state, *, names, download, folder, scratch):
    """Resolve the house registry to body hashes, downloading each house body once.

    An entry pinned with ``body_sha256`` must hash to it; an unpinned one (null, until the registry is
    filled) is identified by the hash of the body downloaded at its pinned revision.
    """
    from . import bodies

    resolved, known = state.setdefault("house", {}), state["bodies"]
    for item in config["house"]:
        ref = f"{item['repo']}@{item['revision']}"
        sha = resolved.get(ref)
        if sha is None or sha not in known:
            claim = {k: item[k] for k in ("repo", "revision")}
            claim.update(sha256=item["body_sha256"], hotkey=None)
            partial = download(claim, scratch / "house.partial")
            sha = bodies.body_hash(partial, names)
            if item["body_sha256"] not in (None, sha):
                shutil.rmtree(partial, ignore_errors=True)
                raise ValueError(f"house body {item['name']} differs from its pinned hash")
            if folder(sha).exists():
                shutil.rmtree(partial)
            else:
                os.replace(partial, folder(sha))
            resolved[ref] = sha
            known[sha] = {
                "hotkey": None, "uid": None, "repo": item["repo"], "revision": item["revision"],
                "block": -1, "house": True, "solo": None, "name": item["name"],
            }  # fmt: skip
    return {resolved[f"{i['repo']}@{i['revision']}"] for i in config["house"]}


def claim_key(claim):
    """A commitment's identity for failures that are not the weights' fault: (hotkey, repo, revision)."""
    return f"{claim['hotkey']}:{claim['repo']}@{claim['revision']}"


def _intake(state, snapshot, *, chassis, base, download, copies, folder, max_new=MAX_NEW_PER_ROUND):
    """Admit new zilsb2 commitments in commitment order. Returns (new, rejected, refused, failed).

    Only failures of weights that hash to the committed hash (``bodies.WeightError``, copies) reject
    the hash itself (``rejected``, by hash, for good). Any other failure (owner file, licence, hash
    mismatch, unreadable files, a bad repo) fails only that commitment (``failed``, by
    ``claim_key``), so another hotkey committing the same hash is still evaluated. Transport errors
    are ``refused`` for this round only and retried next round. At most ``max_new`` bodies are
    admitted, in commitment-block order, then hotkey; later commitments wait for the next round.
    """
    from . import bodies

    known, prior_rejected = state["bodies"], state["rejected"]
    prior_failed = state.setdefault("failed", {})
    claims, refused = [], {}
    for uid, item in snapshot.get("commitments", {}).items():
        if not str(item.get("data", "")).startswith(bodies.PREFIX + ":"):
            continue
        hotkey = snapshot.get("members", {}).get(str(uid))
        try:
            claim = bodies.parse_commitment(item["data"])
            if not isinstance(hotkey, str):
                raise ValueError("commitment uid is not in the round's roster")
        except ValueError as error:
            refused[str(uid)] = str(error)[:400]
            continue
        claims.append({**claim, "uid": int(uid), "hotkey": hotkey, "block": int(item["block"])})
    new, rejected, failed = {}, {}, {}
    for claim in sorted(claims, key=lambda c: (c["block"], c["hotkey"], c["uid"])):
        sha, uid, key = claim["sha256"], str(claim["uid"]), claim_key(claim)
        if sha in known or sha in new:
            if (known.get(sha) or new[sha])["hotkey"] != claim["hotkey"]:
                refused[uid] = "body committed first by another hotkey"
            continue
        if sha in prior_rejected or sha in rejected:
            refused[uid] = "body rejected in an earlier round"
            continue
        if key in prior_failed or key in failed:
            refused[uid] = "commitment failed in an earlier round"
            continue
        if len(new) >= max_new:
            refused[uid] = "the round's new-body cap is reached; retried next round"
            continue
        earlier = {  # house bodies count for everyone: a house re-upload is a copy
            s: folder(s)
            for s, info in [*known.items(), *new.items()]
            if info["house"] or info["hotkey"] != claim["hotkey"]
        }
        # Checked in a partial folder, so a failed claim never leaves files under the hash's name.
        partial = folder(sha).with_name(f".{sha}.partial")
        try:
            shutil.rmtree(folder(sha), ignore_errors=True)  # never trust files of an unknown hash
            download(claim, partial)
            bodies.validate_body(partial, chassis, hotkey=claim["hotkey"], base=base, sha256=sha)
            soup = tuple(s for s in state["members"] if s in known)
            if copies is not None and (reason := copies(sha, partial, earlier, soup=soup)):
                raise bodies.WeightError(f"copy: {reason}")
            os.replace(partial, folder(sha))
        except (OSError, http.client.HTTPException) as error:  # retried next round
            refused[uid] = f"download failed: {_reason(error)}"[:400]
            continue
        except bodies.WeightError as error:
            rejected[sha] = _reason(error)
            continue
        except Exception as error:  # one hostile body must never abort the round
            print(f"soup intake: commitment {key} failed ({type(error).__name__})", flush=True)
            failed[key] = refused[uid] = _reason(error)
            continue
        finally:
            shutil.rmtree(partial, ignore_errors=True)
        new[sha] = {
            "hotkey": claim["hotkey"], "uid": claim["uid"], "repo": claim["repo"],
            "revision": claim["revision"], "block": claim["block"], "house": False, "solo": None,
        }  # fmt: skip
    return new, rejected, refused, failed


def sketch_seed(config, state_dir):
    """This validator's private sketch seed: the configured ``sketch_seed``, else one generated once
    with ``os.urandom`` and kept in ``<state_dir>/sketch-seed``. Sketches are only compared locally,
    so validators need not share it, and it is never published (a known seed lets a copier steer a
    body's sketch away from the one it copies)."""
    if config.get("sketch_seed"):
        return bytes.fromhex(config["sketch_seed"])
    path = Path(state_dir) / "sketch-seed"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        partial.unlink(missing_ok=True)
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(os.urandom(32).hex() + "\n")
        os.replace(partial, path)
    return bytes.fromhex(path.read_text().strip())


class _Sketches:
    """Body sketches (``soup_copies.sketch``) under one fixed per-pool seed, cached on disk by body hash."""

    def __init__(self, root, base, names, seed):
        self.root, self.base, self.names, self.seed = Path(root), base, names, seed
        self.tag = hashlib.sha256(seed).hexdigest()[:16]

    def __call__(self, sha, folder):
        import numpy as np

        from . import soup_copies

        path = self.root / f"{sha}-{self.tag}.npy"
        if path.exists():
            return np.load(path)
        value = soup_copies.sketch(folder, self.base, self.names, seed=self.seed)
        self.root.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial.npy")
        np.save(partial, value)
        os.replace(partial, path)
        return value


def run(
    config,
    state_dir,
    *,
    round_id,
    snapshot,
    block_hash,
    bank,
    ledger,
    cache,
    runner_for,
    fetch_body,
    chassis,
    base=None,
    copies=None,
    twins=None,
    rates=None,
):
    """One soup round for one pool. Returns the private round record (also written to disk).

    ``config`` is the validated soup pool entry plus ``pool_id``, with ``house`` resolved to the
    registry list ({name, repo, revision, body_sha256 or null}) and an optional ``sketch_seed``
    (hex; see ``sketch_seed``). ``snapshot`` is the frozen chain view ({"members": {uid: hotkey},
    "commitments": {uid: {"block", "data"}}}) holding only commitments made before the seed block.
    ``block_hash`` is the seed block's hash. ``ledger`` is a ``soup_exam.UseLedger``, ``cache`` a
    ``soup_grade.PredictionCache``. ``runner_for(folder)`` returns a ``soup_grade`` runner for a body
    or soup folder. ``fetch_body(claim, dest)`` downloads a body repo at its revision into ``dest``
    and returns it. ``chassis`` is the pinned parent folder; ``base`` the vendor (Qwen) parent.

    Hooks, each defaulting to ``soup_copies``/``soup_rates`` when ``base`` is given:
    ``copies(sha256, folder, earlier, *, soup) -> reason | None`` rejects a weight-level copy of an
    earlier body by another hotkey (``earlier`` = {sha256: folder}) or of the current soup (``soup`` =
    its member hashes) before screening; ``twins(rows, other_rows) -> bool`` rejects a behavioural
    twin of a pool body on the screen items (default ``soup_copies.is_twin``);
    ``rates(sha256, info, house) -> 0.0 | 0.25 | 1.0`` sets each pool body's pay rate (default
    ``soup_rates.rate`` against the house sketches; without ``base``, house 0.0 and others 1.0).

    State lives under ``state_dir``: ``pool.json`` (known bodies, rejections, members), ``bodies/``,
    ``sketches/``, ``rounds/<round_id>/`` (subset, intake, private and public records, the final
    soup) and ``decision.json`` (the latest public record). The ledger is written last, so a crashed
    round reruns from its cached predictions and is recorded once.
    """
    from . import (
        base_metrics,
        bodies,
        d3,
        models,
        soup,
        soup_copies,
        soup_exam,
        soup_grade,
        soup_rates,
    )

    if not isinstance(round_id, str) or not _ROUND_ID.fullmatch(round_id):
        raise ValueError("round_id must be a safe name")
    state_dir = Path(state_dir)
    round_dir = state_dir / "rounds" / round_id
    if (round_dir / "private.json").exists():
        return _finish(state_dir, round_dir, ledger)
    pool_id = config["pool_id"]
    model = config["model"]
    model = model if isinstance(model, str) else models.validate_spec(model)
    seed = round_seed(block_hash, pool_id)
    bank_key = bank_sha256(bank)
    # Cached rows are scoped to the bank and the chassis profile (model id and pinned answer layer),
    # so a profile change never reuses rows graded on another chassis.
    scope = f"{bank_key[:16]}-{model}-{d3.pins(model)['files'][soup.READOUT][:12]}"
    scratch = state_dir / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    (state_dir / "bodies").mkdir(parents=True, exist_ok=True)
    names = list(bodies.load_index(chassis))

    def folder(sha):
        return state_dir / "bodies" / sha

    def download(claim, dest):
        shutil.rmtree(dest, ignore_errors=True)
        return Path(fetch_body(claim, dest) or dest)

    by_id = {c["id"]: c for c in bank}
    if (round_dir / "subset.json").exists():
        subset = [by_id[i] for i in json.loads((round_dir / "subset.json").read_text())]
    else:
        size = config.get("subset_size", SUBSET_SIZE)
        retired = ledger.retired()
        if sum(c["id"] not in retired for c in bank) < size:
            # Recorded with no items, so the round is not retried; the last decision keeps paying.
            print(
                f"soup round {round_id} skipped: the round bank has too few unretired items",
                flush=True,
            )
            ledger.record(round_id, [])
            return None
        subset = soup_exam.subset(bank, seed, size, exclude=retired)
        _write_json(round_dir / "subset.json", [c["id"] for c in subset])
    ids = [c["id"] for c in subset]

    state = _read_json(
        state_dir / "pool.json", {"bodies": {}, "rejected": {}, "members": [], "soup": None}
    )
    house = _house(config, state, names=names, download=download, folder=folder, scratch=scratch)
    _write_json(state_dir / "pool.json", state)
    known = state["bodies"]

    sketch = None
    if base is not None:
        sketches = _Sketches(state_dir / "sketches", base, names, sketch_seed(config, state_dir))

        def sketch(sha):
            return sketches(sha, folder(sha))

        if copies is None:

            def copies(sha, body, earlier, *, soup):
                members = [sketch(s) for s in soup]
                return soup_copies.is_copy(
                    sketches(sha, body),
                    [sketch(s) for s in earlier],
                    soup=sum(members) / len(members) if members else None,
                )

        if rates is None:
            house_known = [
                {"body_sha256": s, "sketch": sketch(s), "hotkey": None, "repo": known[s]["repo"]}
                for s in sorted(house)
            ]

            def rates(sha, info, house):
                return soup_rates.rate(
                    sha, sketch(sha), house=house, known=house_known, hotkey=info["hotkey"]
                )

    twins = soup_copies.is_twin if twins is None else twins
    rates = rates or default_rate

    intake = _read_json(round_dir / "intake.json", None)
    if intake is None:
        new, rejected, refused, failed = _intake(
            state, snapshot, chassis=chassis, base=base, download=download, copies=copies,
            folder=folder, max_new=config.get("max_new_per_round", MAX_NEW_PER_ROUND),
        )  # fmt: skip
        intake = {"new": new, "rejected": rejected, "refused": refused, "failed": failed}
        _write_json(round_dir / "intake.json", intake)
    known.update(intake["new"])
    state["rejected"].update(intake["rejected"])
    state.setdefault("failed", {}).update(intake["failed"])
    _write_json(state_dir / "pool.json", state)
    rejected = dict(intake["rejected"])

    def order(sha):
        info = known[sha]
        return (info["block"], -1 if info["uid"] is None else info["uid"], sha)

    def body_key(sha):
        return f"{sha[:32]}-{scope}"

    def graded(key, cases, members):
        """Rows for `cases` under `key`; missing rows grade the soup of `members` built on the chassis
        (a one-member soup is the body itself as a d3 candidate), deleted after grading."""
        have = cache.get(key, [c["id"] for c in cases])
        if len(have) == len(cases):
            return [have[c["id"]] for c in cases]
        out = scratch / key
        shutil.rmtree(out, ignore_errors=True)
        try:
            built = [(folder(s), _rename(folder(s))) for s in members]
            soup.build(out, chassis, built, model=model)
            return soup_grade.grade(key, cases, cache, runner_for(out))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def grade(sha, cases):
        return graded(body_key(sha), cases, (sha,))

    screen_ids = set(soup_grade.screen_subset(subset, seed, config.get("screen_size", SCREEN_SIZE)))
    screen_cases = [c for c in subset if c["id"] in screen_ids]
    incumbents = list(dict.fromkeys([s for s in state["members"] if s in known] + sorted(house)))
    screened, screen = [], {}
    for sha in sorted(intake["new"], key=order):
        try:
            rows = grade(sha, screen_cases)
            others = [  # a hotkey's newer body may behave like its own earlier one
                o
                for o in incumbents + screened
                if known[o]["hotkey"] is None or known[o]["hotkey"] != known[sha]["hotkey"]
            ]
            if any(twins(rows, grade(other, screen_cases)) for other in others):
                rejected[sha] = "behavioural twin of a pool body"
                continue
            screen[sha] = soup_grade.screen_score(screen_cases, rows)
        except OSError:
            raise  # the validator's disk or files, not the body: the round retries
        except Exception as error:  # a body that cannot be graded is rejected
            print(f"soup screen: body {sha} rejected ({type(error).__name__})", flush=True)
            rejected[sha] = _reason(error)
            continue
        screened.append(sha)
    past = [
        s
        for s, info in known.items()
        if s not in incumbents and s not in screen and s not in rejected
        if not info["house"] and info["solo"] is not None and not info.get("pruned")
    ]
    ranked = incumbents + sorted(screened, key=lambda s: (-screen[s], order(s)))
    ranked += sorted(past, key=lambda s: (-known[s]["solo"], order(s)))
    solo = {}
    for sha in dict.fromkeys(ranked):
        if len(solo) >= config.get("max_pool", MAX_POOL):
            break
        try:
            solo[sha] = soup_grade.score(subset, grade(sha, subset))
        except OSError:
            raise
        except Exception as error:  # a body that cannot be graded is rejected
            print(f"soup solo: body {sha} rejected ({type(error).__name__})", flush=True)
            rejected[sha] = _reason(error)
    pool = list(solo)
    if not pool:
        raise ValueError("soup pool has no bodies")

    soup_keys = set()

    def soup_key(members):
        digest = hashlib.sha256("+".join(members).encode()).hexdigest()
        key = f"soup-{digest[:32]}-{scope}"
        soup_keys.add(key)
        return key

    def soup_rows(members):
        if len(members) == 1:
            return grade(members[0], subset)
        return graded(soup_key(members), subset, members)

    scores = {}

    def evaluate(members):
        members = tuple(sorted(members))
        if len(members) == 1:
            return solo[members[0]]
        if members not in scores:
            scores[members] = soup_grade.score(subset, soup_rows(members))
        return scores[members]

    members, trace = greedy(solo, evaluate, order_key=order)
    baseline = None
    if len(members) == 1:
        key = f"chassis-{soup.soup_hash(chassis)[:32]}-{scope}"
        baseline = soup_grade.score(
            subset, soup_grade.grade(key, subset, cache, runner_for(chassis))
        )
    contributions = leave_one_out(members, evaluate, baseline=baseline)
    rate_map = {sha: rates(sha, known[sha], house) for sha in pool}
    seats = seat_weights(members, contributions, rate_map)
    crown = crown_weight(solo, rate_map, house, order_key=order)

    final = round_dir / "soup"
    if not final.exists():
        partial = scratch / f"final-{round_id}"
        shutil.rmtree(partial, ignore_errors=True)
        soup.build(
            partial, chassis, [(folder(s), _rename(folder(s))) for s in members], model=model
        )
        os.replace(partial, final)
    rows = soup_rows(members)
    metrics = base_metrics.score(subset, rows)
    crown_body = next(iter(crown), None)
    private = {
        "round_id": round_id,
        "pool_id": pool_id,
        "seed": seed.hex(),
        "block_hash": block_hash,
        "bank_sha256": bank_key,
        "subset_ids": ids,
        "members": [
            {
                "hotkey": known[s]["hotkey"],
                "body_sha256": s,
                "rate": rate_map[s],
                "contribution": contributions[s],
                "seat_weight": seats.get(s, 0.0),
            }
            for s in members
        ],
        "crown": None
        if crown_body is None
        else {
            "hotkey": known[crown_body]["hotkey"],
            "body_sha256": crown_body,
            "rate": rate_map[crown_body],
            "weight": crown[crown_body],
        },
        "category_scores": {c: 100 * v for c, v in metrics["categories"].items()},
        "family_scores": {f: 100 * v["skill"] for f, v in metrics["families"].items()},
        "predictions": rows,
        "soup_sha256": soup.soup_hash(final),
        "pool": pool,
        "solo": solo,
        "screen": screen,
        "rates": rate_map,
        "trace": trace,
        "rejected": rejected,
        "refused": intake["refused"],
        "failed": intake["failed"],
        "new": sorted(intake["new"]),
        "soup_keys": sorted(soup_keys),
        "round_index": state.get("rounds", 0) + 1,
    }
    _write_json(round_dir / "private.json", private)
    return _finish(state_dir, round_dir, ledger)


def _finish(state_dir, round_dir, ledger):
    """Publish a graded round: public record, pool state, decision, then the ledger (idempotent)."""
    from . import soup_publish

    private = json.loads((round_dir / "private.json").read_text())
    public = soup_publish.public_record(private)
    _write_json(round_dir / "public.json", public)
    state = _read_json(
        state_dir / "pool.json", {"bodies": {}, "rejected": {}, "members": [], "soup": None}
    )
    state["members"] = [m["body_sha256"] for m in private["members"]]
    for sha, score in private["solo"].items():
        state["bodies"][sha]["solo"] = score
    new = set(private["new"])
    for sha, reason in private["rejected"].items():
        if sha in state["members"] or (sha in state["bodies"] and sha not in new):
            continue  # a seated or earlier-admitted body is never condemned or deleted here
        state["rejected"][sha] = reason
        state["bodies"].pop(sha, None)
        shutil.rmtree(state_dir / "bodies" / sha, ignore_errors=True)
    index = private["round_index"]
    for sha in [*private["pool"], *new]:
        if sha in state["bodies"]:
            state["bodies"][sha]["pooled"] = max(state["bodies"][sha].get("pooled", 0), index)
    for sha, info in state["bodies"].items():
        # A body out of the pool for PRUNE_ROUNDS rounds is retired: its folder goes, its hash
        # (so a re-commit is not re-admitted) and its sketch (for copy checks) stay.
        if info["house"] or sha in state["members"] or info.get("pruned"):
            continue
        if index - info.get("pooled", index) >= PRUNE_ROUNDS:
            shutil.rmtree(state_dir / "bodies" / sha, ignore_errors=True)
            info["pruned"] = True
    soups = state.setdefault("soup_cache", {})
    soups.update(dict.fromkeys(private["soup_keys"], index))
    for key, last in list(soups.items()):
        if index - last >= SOUP_CACHE_ROUNDS:
            (state_dir / "cache" / f"{key}.jsonl").unlink(missing_ok=True)
            del soups[key]
    state["rounds"] = index
    state["soup"] = f"rounds/{private['round_id']}/soup"
    state["round_id"] = private["round_id"]
    _write_json(state_dir / "pool.json", state)
    _write_json(state_dir / "decision.json", public)
    for other in (state_dir / "rounds").iterdir():
        if other != round_dir:
            shutil.rmtree(other / "soup", ignore_errors=True)
    ledger.record(private["round_id"], private["subset_ids"])
    return private
