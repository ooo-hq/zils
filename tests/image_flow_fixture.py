"""HTTP contract rehearsal with fixture weights; this is not an accuracy benchmark."""

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from bittensor_wallet import Keypair
from PIL import Image

from miner.queue import Client, run_once
from tests.test_image_jobs import POLICY, fixture
from tests.test_queue import OWNER, Store, server
from zils import cloud, coordinator, models
from zils.runtime import run_child as real_child


class ImageStore(Store):
    def __init__(self):
        super().__init__()
        self.assets = {}

    def rpc(self, name, args):
        if name == "zils_image_get":
            row = self.assets.get(args["p_asset"])
            return row if row and row["owner_id"] == args["p_owner"] else None
        if name == "zils_submit_image_job":
            job = next(j for j in self.tables[coordinator.JOBS] if j["id"] == args["p_job"])
            assert job["owner_id"] == args["p_owner"]
            assert all(self.assets[aid]["job_id"] == job["id"] for aid in args["p_assets"])
            job["status"] = "validating"
            return job
        return super().rpc(name, args)


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


def run(test, reject=False):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        reference = root / "reference"
        reference.mkdir()
        for name in models.IMAJEV_FILES:
            if name != "model.json":
                (reference / name).write_text("stock")
        models.write_metadata(reference, model=models.IMAJEV)
        runtime = root / "fixture-runtime"
        runtime.write_text(f"#!{sys.executable}\n" + SCRIPT)
        runtime.chmod(0o700)
        store = ImageStore()
        key = Keypair.create_from_seed("0x" + "39" * 32)
        store.tables["fez_training_workers"].append(
            {"hotkey": key.ss58_address, "enabled": True, "uid": 1}
        )
        store.tables["zils_worker_profiles"].append(
            {
                "hotkey": key.ss58_address,
                "enabled": True,
                "profile_id": models.IMAJEV,
                "min_free_mib": 12288,
                "evidence": {"max_seconds": 120},
                **models.profile_identity(models.IMAJEV),
            }
        )
        service = coordinator.Service(store, "http://127.0.0.1:8910", models.IMAJEV)

        def child(command, log, device, timeout=3600, **kwargs):
            # Fixture executes on CPU; production gates are independently tested with real CUDA.
            return real_child(command, log, "cpu", timeout, check_lease=kwargs.get("check_lease"))

        with (
            server(store.handler()) as storage,
            server(coordinator.handler(service, "https://app.example")) as url,
            patch.dict(
                "os.environ",
                {
                    "ZILS_IMAGES_ENABLED": "1",
                    "ZILS_IMAGE_TRAINING_ENABLED": "1",
                    "ZILS_IMAGE_RUNTIME_PYTHON": str(runtime),
                    "FIXTURE_IMAGE_REJECT": "1" if reject else "0",
                },
            ),
            patch("zils.imajev.verify_starting_checkpoint"),
            patch("miner.queue.gpu_ready", return_value=True),
            patch("zils.coordinator.gpu_ready", return_value=True),
            patch("miner.worker.run_child", side_effect=child),
            patch("zils.validator.run_child", side_effect=child),
        ):
            store.url = storage
            service.audience = url
            headers = {"Authorization": "Bearer owner-token"}
            response = requests.post(
                url + "/v1/jobs",
                headers=headers,
                json={
                    "name": "image-test",
                    "model": models.IMAJEV,
                    "acceptance": POLICY,
                    "allow_training_data_export": True,
                },
                timeout=5,
            )
            test.assertEqual(response.status_code, 200, response.text)
            created = response.json()
            jid = created["job"]["id"]
            job_url = url + "/v1/jobs/" + jid
            _, splits, assets = fixture()
            for index, asset in enumerate(assets.values()):
                output = io.BytesIO()
                Image.new("RGB", (8, 8), (index % 2, index, 0)).save(output, format="PNG")
                data = output.getvalue()
                sha = hashlib.sha256(data).hexdigest()
                asset.update(
                    owner_id=OWNER,
                    job_id=jid,
                    canonical_sha256=sha,
                    pixel_sha256=sha,
                    canonical_bytes=len(data),
                    canonical_path=jid + "/" + asset["id"],
                )
                store.assets[asset["id"]] = asset
                store.objects["zils-images", asset["canonical_path"]] = data
            for split, rows in splits.items():
                path = root / (split + ".jsonl")
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                cloud.upload(created["uploads"][split]["url"], path)
            response = requests.post(job_url + "/submit", headers=headers, json={}, timeout=5)
            test.assertEqual(response.status_code, 200, response.text)
            engine = coordinator.Processor(
                service,
                root / "processor",
                reference,
                SimpleNamespace(runtime_python=str(runtime), device="cpu"),
            )
            test.assertTrue(engine.tick())
            test.assertEqual(service.job(jid)["status"], "awaiting_approval", service.job(jid))
            store.rpc("fez_approve_training_job", {"p_job": jid, "p_hotkeys": [key.ss58_address]})
            test.assertTrue(
                run_once(
                    Client(url, key),
                    root / "miner",
                    None,
                    str(runtime),
                    "cpu",
                    references={models.IMAJEV: reference},
                )
            )
            test.assertTrue(engine.tick())
            public = requests.get(job_url, headers=headers, timeout=5).json()["job"]
            test.assertEqual(public["status"], "completed", public)
            test.assertEqual(
                public["result"]["delivery"]["status"],
                "no_qualifying_model" if reject else "accepted",
                public,
            )
            test.assertIn("per_class", public["result"]["miners"][0])
            test.assertNotIn("predictions", json.dumps(public))
            test.assertNotIn("logits", json.dumps(public))
            test.assertNotIn("https://", json.dumps(public))
            downloaded = requests.get(job_url + "/downloads", headers=headers, timeout=5)
            test.assertEqual(downloaded.status_code, 409 if reject else 200, downloaded.text)
            return public
