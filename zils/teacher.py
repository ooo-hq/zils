"""Zils Relay tools for base miners: label a question pool with a teacher, learn from it, soup.

    python -m zils.teacher answer --teacher DIR --pool POOL.jsonl --out ANSWERS.jsonl
    python -m zils.teacher train --answers ANSWERS.jsonl --reference REF --out CANDIDATE \
        [--lora 64] [--train-answer-layer] [--start KING] [--lr LR] [--seed 1]
    python -m zils.teacher soup --model d3-mini --out CANDIDATE stock:0.5 STUDENT:0.5

A teacher is any larger d3-family checkpoint with its vendor runtime (``d3_runtime.py``) in its folder.
`answer` keeps the teacher's probability for every declared option. `train` fits the base named by
the reference (any z3 size) so its option distribution matches the teacher's (KL divergence),
scoring options exactly as validators do (last hidden state x FP32 answer layer). By default every
language-model weight trains (FP32 masters, about 100 GB for d3-mini); `--lora 64` trains a merged
LoRA that fits 20-24 GB cards, and `--train-answer-layer` also retrains readout.safetensors.
`soup` averages candidates with the pinned parent (weight averaging); in the 26 Oct 2026 pilot a
0.5 parent / 0.5 students soup kept most of the student's gain without its loss on the sealed exam.
Outputs are ordinary base candidates; commit them with ``zils-miner commit-candidate``.

Pool rows are JSON lines holding ``state`` and ``question`` (or ``questions.decision``); a ``label``
is ignored. Answer rows add ``probabilities`` keyed like ``zils.options(question)``.
"""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

from . import d3, models
from .core import options

HELD_OUT = 1000  # answer rows kept out of training to report the student's KL to the teacher


def pool_row(row):
    """(state, question) from a pool row, without any label."""
    question = row.get("question") or row.get("questions", {}).get("decision")
    if not isinstance(question, dict) or "state" not in row:
        raise ValueError("pool rows need state and question")
    return row["state"], {k: v for k, v in question.items() if k != "label"}


def teacher_probabilities(answer, question):
    """The teacher's distribution over zils.options(question); yes/no answers carry p(true) only."""
    probabilities = answer.get("probabilities")
    if probabilities is None and answer.get("type") == "noul":
        probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
    keys = options(question)
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise ValueError("teacher answer does not cover the declared options")
    values = [float(probabilities[k]) for k in keys]
    if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(
        sum(values), 1, abs_tol=1e-4
    ):
        raise ValueError("teacher answer is not a distribution")
    return dict(zip(keys, values, strict=True))


def answer(args):
    sys.path.insert(0, str(Path(args.teacher).resolve()))
    from d3_runtime import D3

    model = D3.from_pretrained(
        args.teacher,
        device="cuda:0",
        batch_size=1,
        max_length=4096,
        local_files_only=True,
        permutation_average=False,
    )
    rows = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    out = Path(args.out)
    done = sum(1 for _ in out.open()) if out.exists() else 0  # resumable
    started = time.time()
    with out.open("a") as stream:
        for i, row in enumerate(rows[done:], start=done):
            state, question = pool_row(row)
            prepared = model.prepare(state, {"decision": question})
            record = {"i": i, "state": state, "question": question}
            if prepared.errors:  # over the length limit or malformed: kept as a skipped row
                record["error"] = str(prepared.errors)[:300]
            else:
                vectors, tokens = model.run(prepared)
                reply = model.respond(prepared, vectors, tokens)["answers"]["decision"]
                record["probabilities"] = teacher_probabilities(reply, question)
            stream.write(json.dumps(record) + "\n")
            if (i + 1) % 1000 == 0:
                stream.flush()
                print(
                    f"{i + 1}/{len(rows)} answered, {(i + 1 - done) / (time.time() - started):.1f}/s",
                    flush=True,
                )


def kl(model, state, question, teacher):
    import torch

    ids, keys = model.encode(state, question)
    target = torch.tensor([teacher[k] for k in keys], dtype=torch.float32, device=model.device)
    target = target.clamp_min(1e-8) / target.clamp_min(1e-8).sum()
    student = torch.log_softmax(model.logits(ids, len(keys)), -1)
    return torch.sum(target * (target.log() - student))


def write_candidate(model, out, readout=None):
    """The pinned parent's shards with the trained tensor values, its index and contract files.
    A trained answer layer (`readout`) replaces the parent's."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    profile = model.meta["model"]
    parent = d3.pinned_base_path(model=profile)
    layout = d3.weight_map(parent)
    trained = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.base.state_dict().items()}
    if missing := sorted(set(layout) - set(trained)):
        raise RuntimeError(f"backbone has no tensor for {missing[0]}")
    out.mkdir(parents=True, exist_ok=False)
    for shard in sorted(set(layout.values())):
        with safe_open(str(parent / shard), framework="pt") as stream:
            tensors = {key: trained[key].contiguous() for key in stream.keys()}
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
    copy_contract(parent, out, readout, profile)


def copy_contract(parent, out, readout=None, profile=models.D3):
    index = "model.safetensors.index.json"
    for name in (*((index,) if (parent / index).exists() else ()), *models.D3_CONTRACT_FILES):
        shutil.copyfile(parent / name, out / name)
    models.write_metadata(out, kind="candidate", model=profile)
    if readout is not None:
        from safetensors import safe_open
        from safetensors.torch import save_file

        with safe_open(str(parent / "readout.safetensors"), framework="pt") as stream:
            (key,) = stream.keys()
            dtype = stream.get_tensor(key).dtype
        value = readout.detach().to("cpu", dtype).contiguous()
        save_file({key: value}, str(out / "readout.safetensors"), metadata={"format": "pt"})
    d3.validate_candidate(out, profile)  # a trained answer layer must keep the parent's schema


def train(args):
    import torch

    rows = []
    for line in Path(args.answers).read_text().splitlines():
        a = json.loads(line)
        if "probabilities" in a:
            rows.append((a["state"], a["question"], teacher_probabilities(a, a["question"])))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    held, data = rows[:HELD_OUT], rows[HELD_OUT:]
    random.shuffle(data)
    if args.limit:
        data = data[: args.limit]
    model = d3.DecisionModel(Path(args.reference), "cuda")
    if (
        args.start
    ):  # build on an existing candidate (e.g. the pool's king) instead of the stock base
        load_candidate(model, Path(args.start), model.meta["model"])
    model.base.requires_grad_(False)
    if args.lora:
        # LoRA on the 152 language targets: fits 20-24 GB cards; merged into the weights at the end.
        from peft import LoraConfig, get_peft_model

        targets = [
            name
            for name, _ in model.base.named_modules()
            if name.startswith("language_model.layers.") and name.rsplit(".", 1)[-1] in d3.TARGETS
        ]
        config = LoraConfig(r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.0,
                            target_modules=targets, bias="none")  # fmt: skip
        peft = get_peft_model(model.base, config)
        params = [p for n, p in peft.named_parameters() if "lora_" in n]
        for p in params:
            p.data = p.data.float()  # FP32 adapters; the backbone stays BF16
            p.requires_grad_(True)
        masters = params
    else:
        # Full fine-tune with FP32 master weights by hand: BF16 Adam states lose updates this small.
        # About 100 GB for d3-mini; lr above ~1e-6 diverged in the pilot.
        params = [p for n, p in model.base.named_parameters() if n.startswith("language_model.")]
        for p in params:
            p.requires_grad_(True)
        masters = [p.detach().float().clone() for p in params]
    # No gradient checkpointing: its recompute runs outside the runtime's sdpa context and fails
    # torch's metadata check.
    lr = args.lr or (5e-5 if args.lora else 1e-6)
    groups, trainable = [{"params": masters, "lr": lr}], list(masters)
    if args.train_answer_layer:
        # The FP32 readout (255 codes x hidden) learns too; DecisionModel.logits reads model.readout.
        model.readout = torch.nn.Parameter(model.readout.detach().clone())
        groups.append({"params": [model.readout], "lr": args.answer_layer_lr or lr})
        trainable.append(model.readout)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = math.ceil(len(data) / args.accumulate)
    warmup = max(1, steps // 30)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / steps)
    )

    def held_out():
        model.base.eval()
        with torch.no_grad():
            value = sum(kl(model, *r).item() for r in held[:300]) / max(1, len(held[:300]))
        model.base.train()
        return value

    print(json.dumps({"train": len(data), "held_out_kl": held_out(), "steps": steps}), flush=True)
    model.base.train()
    for step in range(steps):
        batch = data[step * args.accumulate : (step + 1) * args.accumulate]
        for r in batch:
            (kl(model, *r) / len(batch)).backward()
        if masters is not params:
            for p, m in zip(params, masters, strict=True):
                m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
                p.grad = None
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if masters is not params:
            with torch.no_grad():
                for p, m in zip(params, masters, strict=True):
                    p.copy_(m.to(torch.bfloat16))
        if (step + 1) % 500 == 0 or step + 1 == steps:
            print(
                json.dumps({"step": step + 1, "of": steps, "held_out_kl": held_out()}), flush=True
            )
    model.base.eval()
    if args.lora:
        for p in params:  # merge in BF16, the backbone's dtype
            p.data = p.data.to(torch.bfloat16)
        model.base = peft.merge_and_unload()
    write_candidate(model, Path(args.out), model.readout if args.train_answer_layer else None)


def load_candidate(model, path, profile=models.D3):
    """Replace the loaded base weights with a validated candidate's (same names and shapes)."""
    from safetensors.torch import load_file

    d3.validate_candidate(path, profile)
    weights = {}
    for shard in sorted(set(d3.weight_map(path).values())):
        weights.update(load_file(str(path / shard)))
    model.base.load_state_dict(weights, strict=True)


def parse_parts(specs):
    parts = []
    for spec in specs:
        path, _, weight = spec.rpartition(":")
        parts.append((path, float(weight)))
    if not parts or any(w < 0 for _, w in parts) or not math.isclose(sum(w for _, w in parts), 1):
        raise ValueError("soup weights must be nonnegative and sum to 1")
    return parts


def soup(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    profile = models.D3_POOL_MODELS.get(args.model, args.model)  # pool name or profile id
    models.spec(profile)
    parent = d3.pinned_base_path(model=profile)
    parts = [(parent if p == "stock" else Path(p), w) for p, w in parse_parts(args.parts)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for shard in sorted(set(d3.weight_map(parent).values())):
        streams = [(safe_open(str(root / shard), framework="pt"), w) for root, w in parts]
        tensors = {
            key: sum(s.get_tensor(key).float() * w for s, w in streams)
            .to(torch.bfloat16)
            .contiguous()
            for key in streams[0][0].keys()
        }
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
    # Answer layers average too; with every part on the parent's readout this is the parent's.
    readouts = []
    for root, w in parts:
        with safe_open(str(root / "readout.safetensors"), framework="pt") as stream:
            (key,) = stream.keys()
            readouts.append((stream.get_tensor(key), w))
    if all(torch.equal(r, readouts[0][0]) for r, _ in readouts):
        copy_contract(parent, out, profile=profile)
    else:
        copy_contract(parent, out, sum(r.float() * w for r, w in readouts), profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("answer", help="label a question pool with a teacher checkpoint")
    a.add_argument("--teacher", required=True, help="teacher checkpoint folder with d3_runtime.py")
    a.add_argument("--pool", required=True)
    a.add_argument("--out", required=True)
    t = commands.add_parser("train", help="fine-tune the pinned base on teacher answers")
    t.add_argument("--answers", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--reference", required=True, help="the pinned base reference (model.json)")
    t.add_argument("--lr", type=float, help="default 1e-6 full fine-tune, 5e-5 with --lora")
    t.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 trains every weight")
    t.add_argument("--start", help="candidate to build on (e.g. the current king); default stock")
    t.add_argument("--train-answer-layer", action="store_true",
                   help="also train the answer layer, readout.safetensors (same shape and dtype, finite)")  # fmt: skip
    t.add_argument("--answer-layer-lr", type=float, help="default: the main learning rate")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--accumulate", type=int, default=16)
    t.add_argument("--limit", type=int)
    s = commands.add_parser("soup", help="weight-average candidates with the pinned parent")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default=models.D3, help="z3 pool (d3-lite ... d3) or profile id")
    s.add_argument("parts", nargs="+", help="PATH:WEIGHT; PATH 'stock' is the pinned parent")
    args = parser.parse_args(argv)
    {"answer": answer, "train": train, "soup": soup}[args.command](args)


if __name__ == "__main__":
    main()
