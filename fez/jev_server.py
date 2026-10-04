"""Private pinned JevK5 inference; one GPU execution with two bounded admission lanes."""

import threading
from concurrent.futures import Future, TimeoutError

from .decisions import DecisionError


class SerialEngine:
    """One pending call per lane; at most four real-time calls before a waiting bulk call."""

    def __init__(self, engine, timeout=30):
        self.engine = engine
        self.timeout = timeout
        self.condition = threading.Condition()
        self.pending = {}
        self.running = False
        self.closed = False
        self.live_count = 0
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    @property
    def pending_count(self):
        with self.condition:
            return len(self.pending)

    def evaluate(self, body, lane="realtime"):
        if lane not in ("realtime", "bulk"):
            raise DecisionError(422, "invalid_lane", "Invalid execution lane.")
        future = Future()
        with self.condition:
            if self.closed:
                raise DecisionError(503, "runtime_unavailable", "Model is unavailable.")
            if lane in self.pending:
                raise DecisionError(
                    529, "overloaded", "Model is busy; retry shortly.", retry_after=1
                )
            self.pending[lane] = (future, body)
            self.condition.notify_all()
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            # Cancellation succeeds only before execution starts. A running model retains its slot.
            future.cancel()
            raise DecisionError(
                504, "model_timeout", "Model execution deadline exceeded."
            ) from None

    def _loop(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    for future, _ in self.pending.values():
                        future.cancel()
                    self.pending.clear()
                    return
                lane = (
                    "bulk"
                    if "bulk" in self.pending
                    and ("realtime" not in self.pending or self.live_count >= 4)
                    else "realtime"
                )
                future, body = self.pending.pop(lane)
                if not future.set_running_or_notify_cancel():
                    continue
                self.live_count = 0 if lane == "bulk" else self.live_count + 1
                self.running = True
            try:
                result = self.engine.predict(self.engine.prepare(body))
            except Exception as error:
                future.set_exception(error)
            else:
                future.set_result(result)
            finally:
                with self.condition:
                    self.running = False
                    self.condition.notify_all()

    def close(self, wait=True):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if wait:
            self.worker.join()


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


class JevEngine:
    def __init__(self, model, max_pass_tokens=4096, max_request_tokens=65536):
        from jevk5.prompt import groups, prompt_text, spread

        if (
            type(max_pass_tokens) is not int
            or type(max_request_tokens) is not int
            or not 1 <= max_pass_tokens <= max_request_tokens
        ):
            raise ValueError("Invalid model token limits")
        self.model = model
        self.max_pass_tokens = max_pass_tokens
        self.max_request_tokens = max_request_tokens
        self.groups, self.prompt_text, self.spread = groups, prompt_text, spread

    def _length(self, ids):
        if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 for i in ids):
            raise DecisionError(503, "tokenizer_error", "Model tokenizer is unavailable.")
        if len(ids) > self.max_pass_tokens:
            raise DecisionError(
                413, "context_limit", "A rendered question exceeds this model context limit."
            )
        return len(ids)

    def prepare(self, body):
        from .decisions import option_descriptions, validate_request

        validate_request(body)
        prepared, reserved = {}, 0
        for qid, question in body["questions"].items():
            options = option_descriptions(question)
            keys = list(options)
            texts = [f"{key}: {value}" for key, value in options.items()]
            criterion = question.get("instructions")
            if criterion is None:
                criterion = ""
            if len(texts) <= 16:
                count = self._length(self.model.encode(body["state"], criterion, texts))
            else:
                count = sum(
                    self._length(
                        self.model.encode(body["state"], criterion, [texts[i] for i in group])
                    )
                    for group in self.groups(len(texts), (len(texts) + 15) // 16)
                )
                # Finalists depend on earlier results. UTF-8 bytes conservatively bound the
                # byte-level tokenizer's tokens for any selection of 16 descriptions.
                longest = sorted(
                    texts,
                    key=lambda text: len(self.prompt_text({}, "", [text]).encode("utf-8")),
                    reverse=True,
                )[:16]
                upper = len(self.prompt_text(body["state"], criterion, longest).encode("utf-8"))
                if upper > self.max_pass_tokens:
                    raise DecisionError(
                        413,
                        "context_limit",
                        "The large-choice final pass exceeds the conservative context budget.",
                    )
                count += upper
            reserved += count
            if reserved > self.max_request_tokens:
                raise DecisionError(
                    413, "context_limit", "Request exceeds this model total token budget."
                )
            prepared[qid] = {
                "state": body["state"],
                "criterion": criterion,
                "texts": texts,
                "keys": keys,
            }
        return {"questions": prepared, "reserved_tokens": reserved}

    def predict(self, prepared):
        import math

        predictions = {}
        for qid, question in prepared["questions"].items():
            tokens = 0

            def read(texts):
                nonlocal tokens
                ids = self.model.encode(question["state"], question["criterion"], texts)
                tokens += self._length(ids)
                logits = [float(x) / TEMPERATURE for x in self.model.letter_logits(ids, len(texts))]
                if len(logits) != len(texts) or not all(math.isfinite(v) for v in logits):
                    raise DecisionError(
                        502, "invalid_model_response", "Model returned invalid logits."
                    )
                high = max(logits)
                weights = [math.exp(v - high) for v in logits]
                total = math.fsum(weights)
                return [v / total for v in weights]

            values = self.spread(
                read, question["texts"], method="knockout", temperature=KNOCKOUT_TEMPERATURE
            )
            predictions[qid] = {
                "probabilities": dict(zip(question["keys"], values, strict=True)),
                "input_tokens": tokens,
            }
        if sum(row["input_tokens"] for row in predictions.values()) > prepared["reserved_tokens"]:
            raise DecisionError(
                502, "token_accounting_error", "Model exceeded its reserved token budget."
            )
        return predictions


def main():
    import argparse
    import hmac
    import importlib.metadata
    import json
    import os
    from pathlib import Path

    from .decision_http import Server, make_handler
    from .decisions import MAX_BODY, MAX_DEPTH, make_response

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--max-pass-tokens", type=int, default=4096)
    parser.add_argument("--max-request-tokens", type=int, default=65536)
    args = parser.parse_args()
    token = os.environ["ZILS_RUNTIME_TOKEN"]
    if len(token) < 32 or not token.isascii():
        raise ValueError(
            "ZILS_RUNTIME_TOKEN must be a strong ASCII secret of at least 32 characters"
        )
    manifest = verify_manifest(args.model_dir)
    package = importlib.metadata.distribution("jevk5")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != RUNTIME_REVISION:
        raise ValueError("Install the pinned JevK5 runtime revision")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from jevk5.runtime import JevK5

    if not torch.cuda.is_available():
        raise OSError("CUDA is required for this model release")
    model = JevK5(
        str(args.model_dir),
        graphs=False,
        device="cuda",
        dtype=torch.bfloat16,
        temperature=TEMPERATURE,
    )
    engine = JevEngine(model, args.max_pass_tokens, args.max_request_tokens)
    serial = SerialEngine(engine)
    identity = {k: manifest[k] for k in ("release_id", "fingerprint")}
    identity["limits"] = {
        "max_pass_tokens": args.max_pass_tokens,
        "max_request_tokens": args.max_request_tokens,
        "max_choice_options": 255,
        "max_score_levels": 10,
    }

    def dispatch(method, path, bearer, body, request_id):
        if not hmac.compare_digest(bearer.encode(), token.encode()):
            raise DecisionError(401, "invalid_credentials", "Invalid runtime credential.")
        if method == "GET" and path == "/health":
            return 200, identity
        if method != "POST" or path not in ("/v1/prepare", "/v1/systemone"):
            raise DecisionError(404, "not_found", "Resource not found.")
        if not isinstance(body, dict) or set(body) - {"request", "lane"} or "request" not in body:
            raise DecisionError(422, "invalid_request", "Expected an inference request envelope.")
        request = body["request"]
        if not isinstance(request, dict) or request.get("model") != RELEASE_ID:
            raise DecisionError(404, "model_not_found", "Model is unavailable.")
        if path == "/v1/prepare":
            prepared = engine.prepare(request)
            return 200, {**identity, "reserved_tokens": prepared["reserved_tokens"]}
        predictions = serial.evaluate(request, body.get("lane", "realtime"))
        make_response(RELEASE_ID, request, predictions)
        return 200, {**identity, "predictions": predictions}

    server = Server(
        ("127.0.0.1", args.port),
        make_handler(dispatch, body_limit=MAX_BODY + 1024, max_depth=MAX_DEPTH + 1),
    )
    print(f"Zils runtime ready: {RELEASE_ID}, loopback port {args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        serial.close()


if __name__ == "__main__":
    main()
