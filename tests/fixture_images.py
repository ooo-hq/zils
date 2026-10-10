"""Image job data and fixture model computation for isolated fleet tests."""

import hashlib
import uuid

from zils import models

OWNER = "10000000-0000-4000-8000-000000000002"


JOB = "10000000-0000-4000-8000-000000000003"


POLICY = {
    "min_accuracy": 0.8,
    "min_brier_improvement": 0.01,
    "positive_class": "damaged",
    "min_positive_recall": 0.9,
    "max_false_positive_rate": 0.1,
}


def fixture():
    job = {
        "id": JOB,
        "owner_id": OWNER,
        "status": "uploading",
        "model_profile": models.spec(models.IMAJEV),
        "acceptance": POLICY,
    }
    splits, assets = {}, {}
    for split in ("train", "calibration", "test"):
        splits[split] = []
        for i, label in enumerate(("normal", "damaged")):
            aid = str(uuid.uuid4())
            sha = hashlib.sha256((split + str(i)).encode()).hexdigest()
            assets[aid] = {
                "id": aid,
                "owner_id": OWNER,
                "job_id": JOB,
                "purpose": "training",
                "state": "ready",
                "canonical_sha256": sha,
                "pixel_sha256": sha,
                "canonical_bytes": 100,
                "width": 8,
                "height": 8,
                "preprocessor": models.spec(models.IMAJEV)["preprocessor"],
                "expires_at": "2099-01-01T00:00:00Z",
                "filename": "secret-" + label + ".png",
                "canonical_path": "private/path",
            }
            splits[split].append(
                {
                    "id": split + str(i),
                    "group_id": split + str(i),
                    "family": "inspection",
                    "state": {},
                    "question": {
                        "type": "choice",
                        "instructions": "Inspect",
                        "criteria": {"normal": None, "damaged": None},
                    },
                    "label": label,
                    "image": {"asset_id": aid},
                }
            )
    return job, splits, assets


SCRIPT = """import json,sys,math,os
from pathlib import Path
from PIL import Image
from zils import models
from zils.image_metrics import probabilities_at_temperature
assert 'SUPABASE_SERVICE_ROLE_KEY' not in os.environ
checkpoint=Path(sys.argv[sys.argv.index('--checkpoint')+1])
if '--train' in sys.argv:
    rows=[json.loads(line) for line in Path(sys.argv[sys.argv.index('--cases')+1]).read_text().splitlines()]
    assert all(set(r)=={'state','questions','images'} for r in rows)
    out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()
    for name in models.IMAJEV_FILES:
        if name!='model.json':(out/name).write_text('trained')
    models.write_metadata(out,model=models.IMAJEV)
else:
    rows=json.loads(Path(sys.argv[sys.argv.index('--cases')+1]).read_text())
    assert all(set(r)=={'id','state','question','image'} for r in rows)
    images=Path(sys.argv[sys.argv.index('--images')+1])
    trained=(checkpoint/'adapter_model.safetensors').read_text()=='trained'
    temp=models.temperature(checkpoint);predictions=[]
    for row in rows:
        keys=[*row['question']['criteria'],'__unknown__']
        with Image.open(images/(row['image']['sha256']+'.png')) as photo:
            gold='normal' if photo.getpixel((0,0))[0]==0 else 'damaged'
        logits=[(3 if k==gold else 0) for k in keys] if trained else [.1,0,0]
        if os.environ.get('FIXTURE_IMAGE_REJECT')=='1' and trained:logits=[0,0,3]
        probabilities=dict(zip(keys,probabilities_at_temperature(logits,temp)))
        predictions.append({'id':row['id'],'probabilities':probabilities,'logits':logits,'elapsed_ms':1,'input_tokens':442})
    print(json.dumps({'predictions':predictions,'runtime':{'temperature':temp,'model':models.spec(models.IMAJEV),**models.profile_identity(models.IMAJEV)}}))
"""
