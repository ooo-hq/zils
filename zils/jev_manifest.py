"""Pinned JevK5 reference identity and local file integrity verification."""

MODEL_REVISION = "c4f7fdb3aeab5582336406e78d3bef11bf98833d"
RUNTIME_REVISION = "f26426d16f59e8bbe1470e5b162cc89329e29b29"
RELEASE_ID = "zils-jevk5-v0.3-r1"
TEMPERATURE = 1.22
KNOCKOUT_TEMPERATURE = 0.93


def _sha(path):
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(manifest):
    import hashlib
    import json

    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_manifest(root):
    from pathlib import Path

    root = Path(root)
    files = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "release.json"
        and ".cache" not in path.relative_to(root).parts
    }
    manifest = {
        "release_id": RELEASE_ID,
        "model_revision": MODEL_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "temperature": TEMPERATURE,
        "knockout_temperature": KNOCKOUT_TEMPERATURE,
        "dtype": "bfloat16",
        "prompt_version": "zils-systemone-v1",
        "files": files,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    return manifest


def verify_manifest(root):
    import json
    from pathlib import Path

    root = Path(root)
    manifest = json.loads((root / "release.json").read_text())
    actual = build_manifest(root)
    if manifest != actual or not manifest["files"]:
        raise ValueError("Model release files or configuration changed")
    return manifest
