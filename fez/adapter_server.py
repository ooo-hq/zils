"""Private customer-adapter runtime compatible with the existing Zils gateway."""

import argparse
import hmac
import importlib.metadata
import json
import os
from pathlib import Path

from .adapter_releases import RUNTIME_REVISION, read_release
from .decisions import (
    MAX_BODY,
    MAX_DEPTH,
    DecisionError,
    make_response,
    option_descriptions,
    validate_request,
)
from .jev_server import SerialEngine


def catalog(root):
    root = Path(root)
    releases = {}
    for path in sorted(root.iterdir()):
        if not path.name.startswith(".") and path.is_dir() and (path / "serving.json").exists():
            release = read_release(path)
            if release["release_id"] in releases:
                raise ValueError("Duplicate adapter release")
            releases[release["release_id"]] = (path, release)
    if not releases:
        raise ValueError("Publish at least one accepted adapter before starting this runtime")
    return releases


class AdapterModel:
    """One base and one fixed LoRA slot. Only the serial execution thread switches it."""

    def __init__(self, path, release):
        from .jevk5 import DecisionModel

        self.model = DecisionModel(path, "cuda")
        self.model.peft.requires_grad_(False)
        self.active = release["fingerprint"]

    def encode(self, state, question):
        from .jevk5 import encode

        return encode(self.model.tokenizer, state, question)

    def activate(self, path, release):
        from .jevk5 import load_adapter_weights

        if self.active == release["fingerprint"]:
            return
        # Clear identity before loading. A failed switch cannot return the previous customer's output.
        self.active = None
        if read_release(path) != release:
            raise ValueError("Customer adapter release changed")
        load_adapter_weights(self.model.peft, path)
        self.model.peft.eval()
        self.active = release["fingerprint"]

    def probabilities(self, ids, keys, temperature):
        import torch

        if self.active is None:
            raise ValueError("No verified adapter is active")
        with torch.inference_mode():
            logits = self.model.logits(ids, len(keys))
            values = torch.softmax(logits.double() / temperature, dim=0).cpu().tolist()
        torch.cuda.synchronize()
        return dict(zip(keys, values, strict=True))


class AdapterEngine:
    def __init__(self, root, backend=None, max_request_tokens=65536):
        if type(max_request_tokens) is not int or max_request_tokens < 1:
            raise ValueError("Request token budget must be a positive integer")
        self.releases = catalog(root)
        self.backend = backend or AdapterModel(*next(iter(self.releases.values())))
        self.max_request_tokens = max_request_tokens

    def release(self, model):
        if not isinstance(model, str) or model not in self.releases:
            raise DecisionError(404, "model_not_found", "Model is unavailable.")
        return self.releases[model]

    def prepare(self, body):
        validate_request(body)
        path, release = self.release(body["model"])
        questions, reserved = {}, 0
        for qid, question in body["questions"].items():
            if len(option_descriptions(question)) > 16:
                raise DecisionError(
                    422, "adapter_option_limit", "Customer adapters support at most 16 outcomes."
                )
            question = dict(question)
            if question.get("instructions") is None:
                question.pop("instructions", None)
            elif not isinstance(question["instructions"], str):
                raise DecisionError(
                    422, "adapter_instructions", "Customer adapters require text instructions."
                )
            try:
                ids, keys = self.backend.encode(body["state"], question)
            except ValueError:
                raise DecisionError(
                    422,
                    "adapter_input",
                    "Input differs from the adapter's training contract or exceeds 2048 tokens.",
                ) from None
            if (
                not isinstance(ids, list)
                or not ids
                or not all(type(i) is int and i >= 0 for i in ids)
            ):
                raise DecisionError(503, "tokenizer_error", "Model tokenizer is unavailable.")
            if len(ids) > 2048:
                raise DecisionError(
                    413,
                    "context_limit",
                    "Customer adapters support at most 2048 tokens per question.",
                )
            if set(keys) != set(option_descriptions(question)):
                raise DecisionError(
                    503, "tokenizer_error", "Model options differ from the decision contract."
                )
            reserved += len(ids)
            if reserved > self.max_request_tokens:
                raise DecisionError(
                    413, "context_limit", "Request exceeds the adapter token budget."
                )
            questions[qid] = {"ids": ids, "keys": keys}
        return {
            "path": path,
            "release": release,
            "questions": questions,
            "reserved_tokens": reserved,
        }

    def predict(self, prepared):
        try:
            self.backend.activate(prepared["path"], prepared["release"])
            return {
                qid: {
                    "input_tokens": len(q["ids"]),
                    "probabilities": self.backend.probabilities(
                        q["ids"], q["keys"], prepared["release"]["temperature"]
                    ),
                }
                for qid, q in prepared["questions"].items()
            }
        except (ValueError, OSError, RuntimeError):
            raise DecisionError(
                503, "adapter_unavailable", "The verified customer adapter is unavailable."
            ) from None


class AdapterRuntime:
    def __init__(self, engine, serial, token):
        self.engine, self.serial, self.token = engine, serial, token

    def dispatch(self, method, path, bearer, body, request_id):
        if not hmac.compare_digest(bearer.encode(), self.token.encode()):
            raise DecisionError(401, "invalid_credentials", "Invalid runtime credential.")
        if method == "GET" and path == "/health":
            return 200, {
                "models": [
                    {"release_id": r["release_id"], "fingerprint": r["fingerprint"]}
                    for _, r in self.engine.releases.values()
                ]
            }
        if method != "POST" or path not in ("/v1/prepare", "/v1/systemone"):
            raise DecisionError(404, "not_found", "Resource not found.")
        if not isinstance(body, dict) or set(body) - {"request", "lane"} or "request" not in body:
            raise DecisionError(422, "invalid_request", "Expected an inference request envelope.")
        request = body["request"]
        if not isinstance(request, dict):
            raise DecisionError(422, "invalid_request", "Expected a decision request.")
        _, release = self.engine.release(request.get("model"))
        identity = {k: release[k] for k in ("release_id", "fingerprint")}
        identity["limits"] = {
            "max_pass_tokens": 2048,
            "max_request_tokens": self.engine.max_request_tokens,
            "max_choice_options": 16,
            "max_score_levels": 10,
        }
        if path == "/v1/prepare":
            prepared = self.engine.prepare(request)
            return 200, {**identity, "reserved_tokens": prepared["reserved_tokens"]}
        predictions = self.serial.evaluate(request, body.get("lane", "realtime"))
        make_response(release["release_id"], request, predictions)
        return 200, {**identity, "predictions": predictions}


def main():
    from .decision_http import Server, make_handler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--token-env", default="ZILS_ADAPTER_RUNTIME_TOKEN")
    parser.add_argument("--max-request-tokens", type=int, default=65536)
    args = parser.parse_args()
    token = os.environ[args.token_env]
    if len(token) < 32 or not token.isascii():
        raise ValueError("Use a strong ASCII runtime secret of at least 32 characters")
    package = importlib.metadata.distribution("jevk5")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != RUNTIME_REVISION:
        raise ValueError("Install the pinned JevK5 runtime revision")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine = AdapterEngine(args.releases, max_request_tokens=args.max_request_tokens)
    serial = SerialEngine(engine)
    runtime = AdapterRuntime(engine, serial, token)
    server = Server(
        ("127.0.0.1", args.port),
        make_handler(runtime.dispatch, body_limit=MAX_BODY + 1024, max_depth=MAX_DEPTH + 1),
    )
    print(
        f"Customer adapter runtime ready: {len(engine.releases)} releases, loopback port {args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        serial.close()


if __name__ == "__main__":
    main()
