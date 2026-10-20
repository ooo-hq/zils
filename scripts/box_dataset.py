"""Generate a seeded synthetic box draw; keep private exam seeds out of public commands."""

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

from zils.box import expand_cases


def generate(out, count, seed):
    """Write cases, expanded questions and SHA-addressed PNGs to a new directory."""
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer of at least two")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    out = Path(out)
    if out.exists():
        raise FileExistsError("draw output exists; use a new directory")
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = {
        "red": (220, 45, 50),
        "blue": (35, 85, 220),
        "green": (35, 160, 70),
        "yellow": (235, 195, 20),
        "purple": (145, 55, 190),
        "orange": (240, 125, 30),
    }
    shapes = ("rectangle", "ellipse", "triangle")
    presence = [index < (count + 1) // 2 for index in range(count)]
    rng.shuffle(presence)
    data, assets = [], {}
    out.mkdir(parents=True)
    (out / "images").mkdir()
    for present in presence:
        image = Image.new("RGB", (256, 256), tuple(rng.randint(243, 255) for _ in range(3)))
        draw = ImageDraw.Draw(image)
        scene = {}
        selected_colors = rng.sample(list(colors), 3)
        cells = rng.sample([(0, 0), (128, 0), (0, 128), (128, 128)], 3)
        for index, (color, (x, y)) in enumerate(zip(selected_colors, cells, strict=True)):
            shape = "rectangle" if index == 0 else rng.choice(shapes)
            left, top = x + rng.randint(8, 32), y + rng.randint(8, 32)
            right, bottom = x + rng.randint(80, 118), y + rng.randint(80, 118)
            rectangle = (left, top, right, bottom)
            if shape == "triangle":
                draw.polygon(
                    [((left + right) // 2, top), (right, bottom), (left, bottom)],
                    fill=colors[color],
                )
            else:
                getattr(draw, shape)(rectangle, fill=colors[color])
            scene[f"{color} {shape}"] = {
                "top": top / 256,
                "bottom": (bottom + 1) / 256,
                "left": left / 256,
                "right": (right + 1) / 256,
            }
        target = rng.choice(
            list(scene)
            if present
            else [
                f"{color} {shape}"
                for color in colors
                for shape in shapes
                if f"{color} {shape}" not in scene
            ]
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        name = digest + ".png"
        if name in assets:
            raise ValueError("duplicate generated image; choose another seed")
        (out / "images" / name).write_bytes(content)
        assets[name] = digest
        data.append(
            {
                "id": digest,
                "group_id": digest,
                "family": "synthetic",
                "state": {"instruction": "Find the target shape in the image."},
                "target": target,
                "image": name,
                "box": scene.get(target),
            }
        )
    files = {}
    for name, rows in (("cases.jsonl", data), ("expanded.jsonl", expand_cases(data))):
        content = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ).encode()
        (out / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "version": "zils-box-draw/v1",
        "scope": "synthetic-only",
        "files": files,
        "assets": assets,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    manifest = generate(args.out, args.count, args.seed)
    print(json.dumps({"scope": manifest["scope"], "cases": args.count}))


if __name__ == "__main__":
    main()
