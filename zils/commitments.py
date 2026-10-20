"""Base candidates by chain commitment: a miner uploads a merged candidate to Hugging Face and commits
`zils1:<checkpoint sha256>:<revision>:<repo id>` with its hotkey; validators read the commitments at
the round's preflight block and pull the pinned revision.

Copy rule: a checkpoint hash belongs to the hotkey that committed it at the earliest block. A later
commitment of the same hash is refused, and a hash two hotkeys committed in that same earliest block
belongs to nobody, so replaying another miner's commitment earns nothing. Commitments are plaintext:
the repo must be public for every validator to pull it, so sealing the pointer (timelocked
commitments) would hide nothing a copier can't find on Hugging Face, and it would not stop a re-saved
copy with a new hash either; see docs/base-candidates.md."""

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

import zils

from . import models

PREFIX = "zils1"
HF_ENDPOINT = "https://huggingface.co"
DOWNLOAD_SECONDS = 3600
# Hugging Face repo ids: owner/name, each 1-96 of [A-Za-z0-9._-], no leading or trailing - or .
REPO = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")


def check_repo(repo):
    parts = repo.split("/") if isinstance(repo, str) else []
    if len(parts) != 2 or not all(REPO.fullmatch(p) for p in parts) or "--" in repo or ".." in repo:
        raise ValueError("repo must be a Hugging Face model id, owner/name")
    return repo


def check_revision(revision):
    if not isinstance(revision, str) or not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("revision must be the full 40-hex commit sha, never a branch or tag")
    return revision


def format_commitment(sha256, revision, repo):
    if not isinstance(sha256, str) or not re.fullmatch("[a-f0-9]{64}", sha256):
        raise ValueError("invalid checkpoint hash")
    return f"{PREFIX}:{sha256}:{check_revision(revision)}:{check_repo(repo)}"


def parse_commitment(text):
    parts = text.split(":") if isinstance(text, str) else []
    if len(parts) != 4 or parts[0] != PREFIX:
        raise ValueError(f"commitment is not {PREFIX}:<sha256>:<revision>:<repo>")
    _, sha256, revision, repo = parts
    if format_commitment(sha256, revision, repo) != text:
        raise ValueError("commitment is not canonical")
    return {"sha256": sha256, "revision": revision, "repo": repo}


def candidates(snapshot, incumbent_sha256):
    """This round's base candidates from the frozen chain snapshot: uid -> claim, or uid -> {"error"}
    for a commitment that is unreadable, the incumbent, or another hotkey's earlier hash."""
    claims, out = {}, {}
    for uid, item in snapshot.get("commitments", {}).items():
        try:
            claim = parse_commitment(item["data"])
            if claim["sha256"] == incumbent_sha256:
                raise ValueError("the incumbent base is not a candidate")
            claims[int(uid)] = {**claim, "block": item["block"]}
        except ValueError as error:
            out[int(uid)] = {"error": str(error)}
    first = {}
    for uid, claim in claims.items():
        first.setdefault(claim["sha256"], []).append((claim["block"], uid))
    for uid, claim in sorted(claims.items()):
        holders = sorted(first[claim["sha256"]])
        block, owner = holders[0]
        if claim["block"] == block and len(holders) > 1 and holders[1][0] == block:
            out[uid] = {"error": f"checkpoint committed by several hotkeys in block {block}"}
        elif owner != uid:
            out[uid] = {"error": f"checkpoint committed first by uid {owner} at block {block}"}
        else:
            out[uid] = claim
    return out


class HTTPSOnly(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        if urlsplit(url).scheme != urlsplit(request.full_url).scheme:
            raise ValueError("Hugging Face redirect changed scheme")
        return super().redirect_request(request, fp, code, message, headers, url)


def opener():
    return build_opener(ProxyHandler({}), HTTPSOnly())


def revision_files(repo, revision, endpoint=None):
    """The repo's files at exactly this commit: name -> {"size", "sha256" (LFS files only)}."""
    endpoint = endpoint or HF_ENDPOINT
    url = f"{endpoint}/api/models/{check_repo(repo)}/revision/{check_revision(revision)}?blobs=true"
    with opener().open(url, timeout=30) as response:
        info = json.loads(response.read(4 * 1024**2))
    if info.get("sha") != revision:
        raise ValueError("revision is not a commit of this repo")
    return {
        s["rfilename"]: {"size": s.get("size"), "sha256": (s.get("lfs") or {}).get("sha256")}
        for s in info.get("siblings", [])
    }


def fetch(claim, destination, *, model, endpoint=None):
    """Download only the candidate files at the pinned revision under the candidate byte budget,
    then check the profile and the committed checkpoint hash."""
    files = models.candidate_files(model, "candidate")
    budget = models.artifact_budget("candidate", model)
    endpoint = endpoint or HF_ENDPOINT
    listed = revision_files(claim["repo"], claim["revision"], endpoint)
    if missing := [name for name in files if name not in listed]:
        raise ValueError(f"repo revision has no {missing[0]}")
    if sum(listed[name]["size"] or 0 for name in files) > budget:
        raise ValueError("checkpoint exceeds byte budget")
    destination = Path(destination)
    destination.mkdir()
    started, total = time.monotonic(), 0
    for name in files:
        url = f"{endpoint}/{claim['repo']}/resolve/{claim['revision']}/{quote(name)}"
        with opener().open(url, timeout=30) as response:
            size = int(response.headers.get("Content-Length", "-1"))
            if not 0 <= size <= budget - total:
                raise ValueError("invalid artifact size or checkpoint exceeds byte budget")
            received = 0
            with (destination / name).open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size or time.monotonic() - started > DOWNLOAD_SECONDS:
                        raise ValueError("artifact exceeds declared size or download deadline")
                    output.write(chunk)
            if received != size:
                raise ValueError("truncated artifact")
            total += received
    if models.checkpoint_model(destination) != model or models.artifact_files(destination) != files:
        raise ValueError("downloaded candidate differs from the required model profile")
    if zils.checkpoint_hash(destination) != claim["sha256"]:
        raise ValueError("downloaded checkpoint hash differs from the commitment")
    return {"bytes": total, "download_ms": (time.monotonic() - started) * 1000}
