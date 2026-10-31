"""Soup rounds: greedy soup selection, leave-one-out contributions, seat and crown pay.

Each round, validators build a fresh uniform-weight average ("soup") of up to five model bodies from the
round's pool, chosen greedily (Wortsman et al. 2022, with a minimum gain per added member). The pool's
emission share splits 80% among soup seats and 20% to the round's best single body (the crown); any share
nobody earns burns. The selection and pay functions are pure: grading is passed in as ``evaluate``.
``run`` orchestrates one round on a validator; its model, download and chain dependencies are injected.

A body is any hashable id (a body sha256 in production). ``evaluate(members)`` grades the uniform soup of a
sorted tuple of bodies and returns its exam score in points.
"""

import gc
import hashlib
import http.client
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
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


def crown_weight(
    solo, rates, house, *, share=CROWN_SHARE, margin=MARGIN, soup_score=None, order_key
):
    """The crown: the best solo body that is not a house ingredient gets ``share`` times its rate,
    provided its solo score beats every house body's, and ``soup_score`` (the current soup on the
    same subset, None before the first soup), by at least ``margin``.

    Ties go to the earlier commitment (``order_key``). With no eligible body, or a best body that does
    not clear the bar, the result is {} and the crown share burns.
    """
    eligible = [body for body in solo if body not in house]
    if not eligible:
        return {}
    best = min(eligible, key=lambda body: (-solo[body], order_key(body)))
    bars = [solo[h] for h in solo if h in house]
    if soup_score is not None:
        bars.append(soup_score)
    if any(solo[best] - bar < margin - _EPS for bar in bars):
        return {}
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
DECISION_ROUNDS = 3  # seats and the crown are scored on the items of this round and the last two
ITEM_SECONDS = 0.065  # GPU time to predict one item (54-62 ms measured on a 48 GB card)
KEY_SECONDS = 35  # GPU time to build and load one body or soup before it predicts
RESERVE_SECONDS = 1800  # kept free under the round deadline for the final soup and the records
MAX_CLAIMS_PER_ROUND = 12  # commitments downloaded per round, failed ones included
MAX_DOWNLOAD_BYTES_PER_ROUND = 60 * 1024**3  # bytes downloaded per round, failed claims included
OTHERS_MAX = 0.3  # share other hotkeys' bodies may add to a body's fit beyond the house bodies
FAILED_CLAIM_BACKOFF_ROUNDS = (
    3  # rounds a hotkey sits out after a claim fails validation or download
)
TRANSPORT_STRIKES = 3  # rounds a claim's download may fail before the claim is retired
GRADE_CHUNK = 200  # items graded per cache write, so a timed-out round keeps its progress
GRADE_STRIKES = 3  # rounds a body may fail grading (with a healthy chassis) before it is banned
_GPU_FAULTS = ("cuda", "out of memory", "cublas", "cudnn", "nccl", "device-side", "accelerator")
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


def _round_fault(error):
    """True for failures of the validator rather than a body: its disk or files, or the GPU (a CUDA
    or out-of-memory error). These fail the whole round, which is retried; they never strike a body."""
    if isinstance(error, OSError | MemoryError):
        return True
    text = f"{type(error).__name__} {error}".lower()
    return isinstance(error, RuntimeError) and any(word in text for word in _GPU_FAULTS)


def _release_memory(error):
    """Drop what a failed grading still holds (the body's model and runner live on in the error's
    traceback frames) and empty the CUDA cache, so the chassis check runs on a clean GPU."""
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = None  # the frames still name their functions, and so their closures
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rename(folder):
    from . import bodies

    return "strip-model" if any(k.startswith("model.") for k in bodies.load_index(folder)) else None


def _house(config, state, *, names, download, folder, scratch):
    """Resolve the house registry to body hashes, downloading each house body once.
    Returns (house body hashes, {name: reason} for entries skipped this round).

    An entry pinned with ``body_sha256`` must hash to it; an unpinned one (null, until the registry is
    filled) is identified by the hash of the body downloaded at its pinned revision. An entry that
    cannot be downloaded or hashed sits this round out; it never stops the round.
    """
    from . import bodies

    resolved, known = state.setdefault("house", {}), state["bodies"]
    house, failures = set(), {}
    for item in config["house"]:
        ref = f"{item['repo']}@{item['revision']}"
        sha = resolved.get(ref)
        if sha is None or sha not in known:
            claim = {k: item[k] for k in ("repo", "revision")}
            claim.update(sha256=item["body_sha256"], hotkey=None)
            partial = scratch / "house.partial"
            try:
                download(claim, partial)
                sha = bodies.body_hash(partial, names)
                if item["body_sha256"] not in (None, sha):
                    raise ValueError(f"house body {item['name']} differs from its pinned hash")
            except Exception as error:
                print(f"soup house: {item['name']} skipped ({type(error).__name__})", flush=True)
                failures[item["name"]] = _reason(error)
                shutil.rmtree(partial, ignore_errors=True)
                continue
            if folder(sha).exists():
                shutil.rmtree(partial)
            else:
                os.replace(partial, folder(sha))
            resolved[ref] = sha
            known[sha] = {
                "hotkey": None, "uid": None, "repo": item["repo"], "revision": item["revision"],
                "block": -1, "house": True, "solo": None, "name": item["name"],
            }  # fmt: skip
        house.add(sha)
    return house, failures


def claim_key(claim):
    """A commitment's identity for failures that are not the weights' fault: (hotkey, repo, revision)."""
    return f"{claim['hotkey']}:{claim['repo']}@{claim['revision']}"


def _folder_bytes(path, pattern="*"):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob(pattern) if p.is_file()) if path.exists() else 0


def _intake(
    state, snapshot, *, chassis, base, download, copies, folder, limits=None, progress=None,
    save=lambda progress: None, round_id=None, residual=None,
):  # fmt: skip
    """Admit new zilsb2 commitments in commitment order. Returns (new, rejected, refused, failed).

    Only failures of weights that hash to the committed hash (``bodies.WeightError``, copies) reject
    the hash itself (``rejected``, by hash, for good). Any other failure (owner file, licence, hash
    mismatch, unreadable files, a bad repo) fails only that commitment (``failed``, by
    ``claim_key``), so another hotkey committing the same hash is still evaluated. Transport errors
    are ``refused`` for this round only and retried next round.

    Claims run in commitment-block order, then hotkey. ``limits`` caps the bodies admitted
    (``max_new_per_round``), the claims downloaded, failed ones included (``max_claims_per_round``)
    and the bytes downloaded (``max_download_bytes_per_round``); later claims wait for the next
    round. Only a claim that downloaded weight bytes counts toward the claims cap, so dead repos
    cannot starve intake; a claim whose download fails in ``TRANSPORT_STRIKES`` separate rounds
    (``round_id``) is retired as failed (the claim, never the hash). ``residual(sha256, folder, *,
    soup)``, when given, measures an admitted body's share beyond the house bodies, soup and
    release (``soup_copies.base_residual``), kept as ``base_residual`` for its rate. ``progress`` holds the outcome of each claim so far and is passed to ``save`` after every
    claim, so a round cut off by its deadline resumes without downloading a claim twice.
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
    limits = limits or {}
    max_new = limits.get("max_new_per_round", MAX_NEW_PER_ROUND)
    max_claims = limits.get("max_claims_per_round", MAX_CLAIMS_PER_ROUND)
    max_bytes = limits.get("max_download_bytes_per_round", MAX_DOWNLOAD_BYTES_PER_ROUND)
    backoff_rounds = limits.get("failed_claim_backoff_rounds", FAILED_CLAIM_BACKOFF_ROUNDS)
    index = state.get("rounds", 0) + 1  # this round's number
    prior_backoff = state.setdefault("backoff", {})
    progress = progress or {}
    for name, empty in (("new", {}), ("rejected", {}), ("failed", {}), ("refused", {})):
        progress.setdefault(name, empty)
    progress.setdefault("done", [])
    progress.setdefault("claims", 0)
    progress.setdefault("bytes", 0)
    progress.setdefault("transient", {})
    progress.setdefault("explained", {})
    progress.setdefault("backoff", {})
    prior_transient = state.setdefault("transient", {})
    prior_explained = state.setdefault("explained", {})
    new, rejected, failed = progress["new"], progress["rejected"], progress["failed"]
    refused.update(progress["refused"])
    progress["refused"] = refused
    for claim in sorted(claims, key=lambda c: (c["block"], c["hotkey"], c["uid"])):
        sha, uid, key = claim["sha256"], str(claim["uid"]), claim_key(claim)
        if uid in progress["done"]:
            continue
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
        last = progress["backoff"].get(claim["hotkey"], prior_backoff.get(claim["hotkey"]))
        if last is not None and 0 < index - last <= backoff_rounds:
            refused[uid] = (
                f"the hotkey's last claim failed; it backs off until round {last + backoff_rounds + 1}"
            )
            continue
        if len(new) >= max_new:
            refused[uid] = "the round's new-body cap is reached; retried next round"
            continue
        if progress["claims"] >= max_claims:
            refused[uid] = "the round's claims cap is reached; retried next round"
            continue
        if progress["bytes"] >= max_bytes:
            refused[uid] = "the round's download budget is spent; retried next round"
            continue
        earlier = {
            s: folder(s)
            for s, info in [*known.items(), *new.items()]
            if not info["house"] and info["hotkey"] != claim["hotkey"]
        }
        active = sorted(s for s in earlier if not {**known, **new}[s].get("pruned"))
        if key in prior_explained and set(prior_explained[key]["earlier"]) <= set(active):
            # Retried once the claim changes or one of the bodies that explained it leaves the pool.
            refused[uid] = prior_explained[key]["reason"]
            continue
        partner = state.get("twins", {}).get(key, {}).get("partner")
        if partner in known and not known[partner].get("pruned"):
            refused[uid] = state["twins"][key]["reason"]  # retried once the partner leaves
            continue
        # Checked in a partial folder, so a failed claim never leaves files under the hash's name.
        partial = folder(sha).with_name(f".{sha}.partial")
        try:
            shutil.rmtree(folder(sha), ignore_errors=True)  # never trust files of an unknown hash
            try:
                download(claim, partial)
            finally:
                progress["bytes"] += _folder_bytes(partial)
            bodies.check_layout(partial, chassis)
            progress["claims"] += 1  # only a claim with a readable chassis layout uses a slot
            bodies.validate_body(
                partial, chassis, hotkey=claim["hotkey"], base=base, sha256=sha,
                max_bytes=limits.get("max_body_bytes"),
            )  # fmt: skip
            soup = tuple(s for s in state["members"] if s in known)
            reason = copies(sha, partial, earlier, soup=soup) if copies is not None else None
            measured = None
            if not reason and residual is not None:
                measured = residual(sha, partial, soup=soup)
            if reason == "explained by earlier":
                # A judgement on today's pool, not on the weights: never a hash ban.
                refused[uid] = f"copy: {reason}; retried when the claim changes or one of the "
                refused[uid] += "explaining bodies leaves the pool"
                progress["explained"][key] = {"reason": refused[uid], "earlier": active}
            elif reason:
                raise bodies.WeightError(f"copy: {reason}")
            else:
                os.replace(partial, folder(sha))
        except (OSError, http.client.HTTPException) as error:  # retried next round, a few times
            reason = f"download failed: {_reason(error)}"[:400]
            refused[uid] = progress["transient"][key] = reason
            progress["backoff"][claim["hotkey"]] = index
            if len({*prior_transient.get(key, {}), round_id}) >= TRANSPORT_STRIKES:
                failed[key] = f"{reason} ({TRANSPORT_STRIKES} rounds)"[:400]
        except bodies.WeightError as error:
            rejected[sha] = _reason(error)
        except Exception as error:  # one hostile body must never abort the round
            print(f"soup intake: commitment {key} failed ({type(error).__name__})", flush=True)
            failed[key] = refused[uid] = _reason(error)
            progress["backoff"][claim["hotkey"]] = index
        else:
            if key not in progress["explained"]:
                new[sha] = {
                    "hotkey": claim["hotkey"], "uid": claim["uid"], "repo": claim["repo"],
                    "revision": claim["revision"], "block": claim["block"], "house": False,
                    "solo": None,
                }  # fmt: skip
                if measured is not None:
                    new[sha]["base_residual"] = measured
        finally:
            shutil.rmtree(partial, ignore_errors=True)
        progress["done"].append(uid)
        save(progress)
    return new, rejected, refused, failed


def _previous_rounds(state_dir, ledger, round_id, wanted):
    """The ``wanted`` latest finished rounds before ``round_id`` that drew items, oldest first.

    Returns ([(round id, the cases it drew)] that can be pooled, {round id: reason} for rounds in
    that window that cannot be). Cases come from the round's own record
    (``rounds/<id>/subset-cases.json``), never from the bank, which may have dropped them since. A
    round that failed or was skipped has no items in the ledger, so it never enters the window.
    """
    if wanted < 1:
        return [], {}
    window = [(r, ids) for r, ids in ledger.rounds() if r != round_id and ids][-wanted:]
    usable, skipped = [], {}
    for previous, _ in window:
        path = Path(state_dir) / "rounds" / previous / "subset-cases.json"
        if not _ROUND_ID.fullmatch(previous) or not path.exists():
            skipped[previous] = "no recorded items"
            continue
        usable.append((previous, json.loads(path.read_text())))
    return usable, skipped


def _grading_seconds(cache, body_keys, soups, ids):
    """Estimated GPU seconds to finish grading ``body_keys`` (from their cached rows) and up to
    ``soups`` soups (assumed uncached) on ``ids``."""
    items, keys = soups * len(ids), soups
    for key in body_keys:
        missing = len(ids) - len(cache.get(key, ids))
        items, keys = items + missing, keys + (missing > 0)
    return round(items * ITEM_SECONDS + keys * KEY_SECONDS, 1)


def _decision_plan(
    state_dir, ledger, cache, *, round_id, bank, subset, requested, body_keys,
    deadline, clock,
):  # fmt: skip
    """The cases this round's seats and crown are scored on: this round's subset plus the cases
    up to ``requested`` - 1 earlier rounds drew, as they recorded them (``_previous_rounds``).

    Every body is scored on the same cases. An earlier case is dropped for all of them when the
    use ledger has retired it since, or when the current bank holds a different case under its id
    (cached rows are keyed by id). A case that has only left the bank is kept. With a
    ``deadline``, the most rounds whose estimated grading fits before it (less
    ``RESERVE_SECONDS``) are used; this round's own subset is always used, as with one round."""
    from .soup_publish import canonical_json

    previous, skipped = _previous_rounds(state_dir, ledger, round_id, requested - 1)
    retired, in_bank = ledger.retired(), {c["id"]: canonical_json(c) for c in bank}
    own = {c["id"]: c for c in subset}
    options = []
    for n in range(1, len(previous) + 2):
        extra = {}
        for _, cases in previous[len(previous) - (n - 1) :]:  # oldest first: the newest wins
            extra.update((c["id"], c) for c in cases if c["id"] not in own)
        kept = {
            i: c
            for i, c in extra.items()
            if i not in retired and in_bank.get(i, canonical_json(c)) == canonical_json(c)
        }
        chosen = [r for r, _ in previous[len(previous) - (n - 1) :]] + [round_id]
        cases = sorted({**own, **kept}.values(), key=lambda c: c["id"])
        options.append((chosen, cases, len(extra) - len(kept)))
    soups = (len(body_keys) - 1) + min(len(body_keys), MAX_MEMBERS) + 1  # trials, LOO, current
    estimates = {
        str(n): _grading_seconds(cache, body_keys, soups, [c["id"] for c in cases])
        for n, (_, cases, _) in enumerate(options, 1)
    }
    n, fallback = len(options), None
    if deadline is not None:
        left = deadline - clock() - RESERVE_SECONDS
        while n > 1 and estimates[str(n)] > left:
            n -= 1
        if n < len(options):
            fallback = (
                f"deadline: {len(options)} rounds needed about {estimates[str(len(options))]:.0f} s"
                f" of grading and {max(0, left):.0f} s were left; used {n}"
            )
    rounds, cases, dropped = options[n - 1]
    return {
        "requested": requested,
        "rounds": rounds,
        "items": len(cases),
        "dropped_items": dropped,
        "skipped": skipped,
        "fallback": fallback,
        "estimates": estimates,
        "cases": cases,
    }


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


SEED_CHANGED = (
    "the sketch_seed differs from the one this pool's sketches were made with; sketches of pruned "
    "bodies cannot be remade, so every copy check would be wrong. Restore the original sketch_seed "
    "(or base-pools/<pool>/soup/sketch-seed): it must never change once set"
)


def sketch_tag(seed):
    return hashlib.sha256(seed).hexdigest()[:16]


class _Sketches:
    """Body sketches (``soup_copies.sketch``) under one fixed per-pool seed, cached on disk by body
    hash, each file holding the tag of the seed it was made with."""

    def __init__(self, root, base, names, seed):
        self.root, self.base, self.names, self.seed = Path(root), base, names, seed
        self.tag = sketch_tag(seed)

    def __call__(self, sha, folder, *, trusted=False):
        import numpy as np

        from . import soup_copies

        path = self.root / f"{sha}.npz"
        if path.exists():
            with np.load(path) as stored:
                if str(stored["tag"]) != self.tag:
                    raise ValueError(SEED_CHANGED)
                return stored["sketch"]
        value = soup_copies.sketch(folder, self.base, self.names, seed=self.seed, trusted=trusted)
        self.root.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{sha}.partial.npz")
        np.savez(partial, sketch=value, tag=np.array(self.tag))
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
    bank_source=None,
    deadline=None,
    clock=time.time,
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
    ``soup_rates.rate`` against the house sketches and the body's ``base_residual``; without
    ``base``, house 0.0 and others 1.0). The release is the chassis's own body (the pool's served
    model), so the default checks treat the house bodies, the current soup and the chassis body
    as allowed bases.

    ``bank_source`` (``soup_current.prepare``'s record of where the subset's items came from) is
    stored in the private round record when given.

    Seats, contributions and the crown are scored on the items of this round and of up to
    ``decision_rounds`` - 1 earlier rounds (``_decision_plan``); ``deadline`` (a ``clock()`` time,
    None for none) lowers that number when the grading would not fit. The use ledger counts only
    this round's own subset.

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
    names = list(bodies.load_index(chassis, trusted=True))

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
    if not (round_dir / "subset-cases.json").exists():
        # The cases as drawn, so later rounds can pool them after they leave the bank.
        _write_json(round_dir / "subset-cases.json", subset)
    ids = [c["id"] for c in subset]

    state = _read_json(
        state_dir / "pool.json", {"bodies": {}, "rejected": {}, "members": [], "soup": None}
    )
    house, house_failures = _house(
        config, state, names=names, download=download, folder=folder, scratch=scratch
    )
    _write_json(state_dir / "pool.json", state)
    known = state["bodies"]

    sketch = None
    if base is not None:
        sketches = _Sketches(state_dir / "sketches", base, names, sketch_seed(config, state_dir))
        if state.setdefault("sketch_tag", sketches.tag) != sketches.tag:
            raise ValueError(SEED_CHANGED)  # at startup, never as a silent failure of every claim
        _write_json(state_dir / "pool.json", state)

        def sketch(sha):
            return sketches(sha, folder(sha))

        # The release is the chassis's body; its sketch is keyed by the pinned files, so a new
        # release (a new pinned parent) is sketched once.
        pinned = hashlib.sha256(json.dumps(d3.pins(model)["files"], sort_keys=True).encode())
        release_key = f"release-{pinned.hexdigest()[:32]}"

        def release():
            return sketches(release_key, chassis, trusted=True)

        def soup_sketch(soup):
            members = [sketch(s) for s in soup]
            return sum(members) / len(members) if members else None

        def residual(sha, body, *, soup):
            return soup_copies.base_residual(
                sketches(sha, body),
                house=[sketch(s) for s in sorted(house)],
                soup=soup_sketch(soup),
                release=release(),
            )

        if copies is None:

            def copies(sha, body, earlier, *, soup):
                return soup_copies.is_copy(
                    sketches(sha, body),
                    [sketch(s) for s in earlier],
                    soup=soup_sketch(soup),
                    release=release(),
                    house=[sketch(s) for s in sorted(house)],
                    # Explained shares count only earlier bodies still in the pool.
                    explained=[sketch(s) for s in earlier if not known.get(s, {}).get("pruned")],
                    others_max=config.get("others_max", OTHERS_MAX),
                )

        if rates is None:
            public = []  # known public models other than the house ones (none at launch)

            def rates(sha, info, house):
                return soup_rates.rate(
                    sha,
                    sketch(sha),
                    house=house,
                    known=public,
                    hotkey=info["hotkey"],
                    house_sketches=[sketch(s) for s in sorted(house)],
                    house_mix_max=config.get("house_mix_max", soup_rates.HOUSE_MIX_MAX),
                    base_residual=info.get("base_residual"),
                )

    else:
        residual = None
    twins = soup_copies.is_twin if twins is None else twins
    rates = rates or default_rate

    intake = _read_json(round_dir / "intake.json", None)
    if intake is None:
        limits = {
            k: config[k]
            for k in (
                "max_new_per_round",
                "max_claims_per_round",
                "max_download_bytes_per_round",
                "failed_claim_backoff_rounds",
            )
            if k in config
        }
        limits["max_body_bytes"] = bodies.max_bytes(pool_id)  # each pool's own body size
        new, rejected, refused, failed = _intake(
            state, snapshot, chassis=chassis, base=base, download=download, copies=copies,
            folder=folder, limits=limits, round_id=round_id, residual=residual,
            progress=_read_json(round_dir / "intake-progress.json", None),
            save=lambda progress: _write_json(round_dir / "intake-progress.json", progress),
        )  # fmt: skip
        intake = {"new": new, "rejected": rejected, "refused": refused, "failed": failed}
        done = _read_json(round_dir / "intake-progress.json", {})
        intake["transient"] = done.get("transient", {})
        intake["explained"] = done.get("explained", {})
        intake["backoff"] = done.get("backoff", {})
        _write_json(round_dir / "intake.json", intake)
    known.update(intake["new"])
    state["rejected"].update(intake["rejected"])
    state.setdefault("failed", {}).update(intake["failed"])
    for key, reason in intake.get("transient", {}).items():
        state.setdefault("transient", {}).setdefault(key, {})[round_id] = reason
    state.setdefault("explained", {}).update(intake.get("explained", {}))
    state.setdefault("backoff", {}).update(intake.get("backoff", {}))
    _write_json(state_dir / "pool.json", state)
    rejected = dict(intake["rejected"])

    def order(sha):
        info = known[sha]
        return (info["block"], -1 if info["uid"] is None else info["uid"], sha)

    def body_key(sha):
        return f"{sha[:32]}-{scope}"

    def built(name, sources, use):
        """``use(runner)`` on the soup of ``sources`` ([(folder, rename)]) built on the chassis as a
        d3 candidate in scratch, deleted afterwards. Every grading goes through here, so no runner
        ever loads a raw body or chassis folder (they have no candidate metadata)."""
        out = scratch / name
        shutil.rmtree(out, ignore_errors=True)
        try:
            soup.build(out, chassis, sources, model=model)
            return use(runner_for(out))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def graded(key, cases, sources):
        """Rows for `cases` under `key`. Missing rows grade the soup of `sources` built once on the
        chassis (a one-member soup is that body as a d3 candidate), cached every `GRADE_CHUNK`
        items."""
        have = cache.get(key, [c["id"] for c in cases])
        missing = [c for c in cases if c["id"] not in have]
        if missing:

            def run_missing(runner):
                for start in range(0, len(missing), GRADE_CHUNK):
                    soup_grade.grade(key, missing[start : start + GRADE_CHUNK], cache, runner)

            built(key, sources, run_missing)
            have = cache.get(key, [c["id"] for c in cases])
        return [have[c["id"]] for c in cases]

    def members_of(shas):
        return [(folder(s), _rename(folder(s))) for s in shas]

    def grade(sha, cases):
        return graded(body_key(sha), cases, members_of((sha,)))

    chassis_body = [(chassis, None)]  # the chassis's own body, built like any other

    failures, checked = {}, []

    def strike(sha, stage, error):
        """A body failed grading. A GPU or disk fault, or a chassis that cannot grade either, fails
        the round instead; otherwise the body sits this round out (``_finish`` bans it after
        ``GRADE_STRIKES`` such rounds)."""
        if _round_fault(error):
            raise error
        if not checked:
            _release_memory(error)
            try:
                built("chassis-check", chassis_body, lambda runner: runner(subset[:1]))
            except Exception as chassis_error:
                raise RuntimeError(
                    f"the chassis cannot grade either ({type(chassis_error).__name__}); "
                    "the round is retried"
                ) from error
            checked.append(True)
        print(f"soup {stage}: body {sha} failed grading ({type(error).__name__})", flush=True)
        failures[sha] = _reason(error)

    screen_ids = set(soup_grade.screen_subset(subset, seed, config.get("screen_size", SCREEN_SIZE)))
    screen_cases = [c for c in subset if c["id"] in screen_ids]
    max_pool = config.get("max_pool", MAX_POOL)
    incumbents = list(dict.fromkeys([s for s in state["members"] if s in known] + sorted(house)))
    solo = {}
    for sha in incumbents[:max_pool]:  # graded first, so twin checks read their rows from the cache
        try:
            solo[sha] = soup_grade.score(subset, grade(sha, subset))
        except Exception as error:
            strike(sha, "solo", error)
    # New bodies, and bodies admitted earlier that have not been graded yet (an earlier failure).
    candidates = set(intake["new"]) | {
        s
        for s, info in known.items()
        if not info["house"] and info["solo"] is None and not info.get("pruned")
        if s not in rejected and s not in solo
    }
    screened, screen, scored, twinned = [], {}, {}, {}
    for sha in sorted(candidates, key=order):
        try:
            rows = grade(sha, subset)  # the whole subset once; the screen reads its own items
            by_id = {r["id"]: r for r in rows}
            screen_rows = [by_id[c["id"]] for c in screen_cases]
            others = [  # a hotkey's newer body may behave like its own earlier one
                o
                for o in [*solo, *screened]
                if known[o]["hotkey"] is None or known[o]["hotkey"] != known[sha]["hotkey"]
            ]
            partner = next((o for o in others if twins(screen_rows, grade(o, screen_cases))), None)
            if partner is not None:
                # A judgement on today's pool, not on the weights: the claim is refused, never the
                # hash, and is retried when re-committed or once the partner leaves the pool.
                twinned[sha] = {
                    "key": claim_key(known[sha]),
                    "partner": partner,
                    "reason": f"behavioural twin of pool body {partner}; retried when the claim"
                    " changes or that body leaves the pool",
                }
                continue
            screen[sha] = soup_grade.screen_score(screen_cases, screen_rows)
            score = soup_grade.score(subset, rows)
        except Exception as error:
            strike(sha, "screen", error)
            continue
        screened.append(sha)
        scored[sha] = score
    past = [
        s
        for s, info in known.items()
        if s not in solo and s not in screen and s not in rejected and s not in failures
        if not info["house"] and info["solo"] is not None and not info.get("pruned")
    ]
    ranked = list(solo) + sorted(screened, key=lambda s: (-screen[s], order(s)))
    ranked += sorted(past, key=lambda s: (-known[s]["solo"], order(s)))
    for sha in dict.fromkeys(ranked):
        if len(solo) >= max_pool:
            break
        if sha in solo:
            continue
        if sha in scored:
            solo[sha] = scored[sha]
            continue
        try:
            solo[sha] = soup_grade.score(subset, grade(sha, subset))
        except Exception as error:
            strike(sha, "solo", error)
    plan = _read_json(round_dir / "decision-items.json", None)
    if plan is None:  # chosen once, so a resumed round scores on the same items
        plan = _decision_plan(
            state_dir, ledger, cache, round_id=round_id, bank=bank,
            subset=subset, requested=config.get("decision_rounds", DECISION_ROUNDS),
            body_keys=[body_key(s) for s in solo], deadline=deadline, clock=clock,
        )  # fmt: skip
        _write_json(round_dir / "decision-items.json", plan)
    decision_cases = subset
    if len(plan["rounds"]) > 1:
        decision_cases = plan["cases"]
        for sha in list(solo):  # rescored on the pooled items; cached rows are never regraded
            try:
                solo[sha] = soup_grade.score(decision_cases, grade(sha, decision_cases))
            except Exception as error:
                del solo[sha]
                strike(sha, "solo", error)
    pool = list(solo)
    if not pool:
        raise ValueError("soup pool has no bodies")

    soup_keys = set()

    def soup_key(members):
        digest = hashlib.sha256("+".join(members).encode()).hexdigest()
        key = f"soup-{digest[:32]}-{scope}"
        soup_keys.add(key)
        return key

    def soup_rows(members, items):
        if len(members) == 1:
            return grade(members[0], items)
        return graded(soup_key(members), items, members_of(members))

    scores = {}

    def evaluate(members):
        members = tuple(sorted(members))
        if len(members) == 1:
            return solo[members[0]]
        if members not in scores:
            rows = soup_rows(members, decision_cases)
            scores[members] = soup_grade.score(decision_cases, rows)
        return scores[members]

    # The crown bar: the current (previous round's) soup on the decision items, if one exists.
    # If it cannot be graded the crown burns this round; only a GPU or disk fault fails the round.
    current = tuple(sorted(s for s in state["members"] if s in known))
    soup_bar, bar_missing = None, False
    if current:
        try:
            soup_bar = evaluate(current)
        except Exception as error:
            if _round_fault(error):
                raise
            print(
                f"soup round: the current soup cannot be graded ({type(error).__name__})",
                flush=True,
            )
            bar_missing = True
    members, trace = greedy(solo, evaluate, order_key=order)
    baseline = None
    if len(members) == 1:
        key = f"chassis-{soup.soup_hash(chassis)[:32]}-{scope}"
        rows = graded(key, decision_cases, chassis_body)
        baseline = soup_grade.score(decision_cases, rows)
    contributions = leave_one_out(members, evaluate, baseline=baseline)
    rate_map = {sha: rates(sha, known[sha], house) for sha in pool}
    seats = seat_weights(members, contributions, rate_map)
    crown = (
        {}
        if bar_missing
        else crown_weight(solo, rate_map, house, soup_score=soup_bar, order_key=order)
    )

    final = round_dir / "soup"
    if not final.exists():
        partial = scratch / f"final-{round_id}"
        shutil.rmtree(partial, ignore_errors=True)
        soup.build(
            partial, chassis, [(folder(s), _rename(folder(s))) for s in members], model=model
        )
        os.replace(partial, final)
    rows = soup_rows(members, subset)  # published predictions and scores: this round's subset
    metrics = base_metrics.score(subset, rows)
    crown_body = next(iter(crown), None)
    private = {
        "round_id": round_id,
        "pool_id": pool_id,
        "seed": seed.hex(),
        "block_hash": block_hash,
        "bank_sha256": bank_key,
        "subset_ids": ids,
        "decision": {k: v for k, v in plan.items() if k != "cases"},
        "decision_ids": [c["id"] for c in plan["cases"]],
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
        "soup_bar": soup_bar,
        "rejected": rejected,
        "grade_failures": failures,
        "twins": twinned,
        "house_failures": house_failures,
        "refused": intake["refused"],
        "failed": intake["failed"],
        "new": sorted(intake["new"]),
        "soup_keys": sorted(soup_keys),
        "round_index": state.get("rounds", 0) + 1,
        **({"bank_source": bank_source} if bank_source is not None else {}),
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
    banned = dict(private["rejected"])
    strikes = state.setdefault("grade_failures", {})
    for sha, reason in private.get("grade_failures", {}).items():
        strikes.setdefault(sha, {})[private["round_id"]] = reason
        if len(strikes[sha]) >= GRADE_STRIKES:
            banned[sha] = f"failed grading in {len(strikes[sha])} rounds: {reason}"[:400]
    for sha, reason in banned.items():
        info = state["bodies"].get(sha, {})
        if sha in state["members"] or info.get("house"):
            continue  # a seated or house body is never condemned or deleted here
        if sha in state["bodies"] and sha not in new and sha not in strikes:
            continue  # nor an earlier-admitted body, except after repeated grading failures
        state["rejected"][sha] = reason
        state["bodies"].pop(sha, None)
        shutil.rmtree(state_dir / "bodies" / sha, ignore_errors=True)
    for sha, twin in private.get("twins", {}).items():
        if sha in state["members"] or state["bodies"].get(sha, {}).get("house"):
            continue
        state.setdefault("twins", {})[twin["key"]] = {
            "partner": twin["partner"],
            "reason": twin["reason"],
        }
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
    # Kept at least as long as a later round can still pool the items they were graded on.
    keep = max(SOUP_CACHE_ROUNDS, private.get("decision", {}).get("requested", 1))
    for key, last in list(soups.items()):
        if index - last >= keep:
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
