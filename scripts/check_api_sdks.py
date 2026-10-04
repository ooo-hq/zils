"""Verify pinned official TypeSafe clients against loopback Zils fixtures, never TypeSafe servers."""

import argparse
import json
import subprocess
from pathlib import Path

from fez import decision_http
from fez.api import Gateway, Registry
from tests.test_api import FINGERPRINT, RELEASE, FixtureStore, GatewayTest
from tests.test_decision_http import server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--js-module", type=Path, required=True, help="Installed SDK dist/index.mjs"
    )
    args = parser.parse_args()
    from typesafe_sdk import RetryPolicy, TypeSafeAPIError, TypeSafeClient

    fixture = GatewayTest()
    fixture.setUp()
    store = FixtureStore()
    questions = {
        "n": {"type": "noul"},
        "c": {
            "type": "choice",
            "instructions": None,
            "criteria": {"match": None, "no_match": None},
        },
        "s": {"type": "score", "criteria": [{"level": "low"}, ["high"]]},
    }
    try:
        with server(decision_http, fixture.runtime) as runtime_port:
            entry = {
                "id": RELEASE,
                "fingerprint": FINGERPRINT,
                "aliases": ["zils-shared"],
                "owners": None,
                "url": f"http://127.0.0.1:{runtime_port}",
                "token_env": "ZILS_TEST_RUNTIME_TOKEN",
                "release_date": "2026-10-04",
                "description": "Test fixture",
            }
            gateway = Gateway(store, Registry([entry]))
            with server(decision_http, gateway.dispatch) as port:
                url = f"http://127.0.0.1:{port}"
                with TypeSafeClient(
                    api_key="valid-key",
                    base_url=url,
                    model="zils-shared",
                    retry=RetryPolicy(max_retries=0),
                ) as client:
                    assert {x.name for x in client.models.list().models} == {RELEASE, "zils-shared"}
                    result = client.system_one(state={"product": "USB cable"}, questions=questions)
                    assert result.model == RELEASE and result.usage.input_tokens == 30
                    assert result.answers["s"].legend == {0: {"level": "low"}, 1: ["high"]}
                    store.limit = True
                    try:
                        client.system_one(state="x", questions=questions)
                        raise AssertionError("Expected Python rate-limit error")
                    except TypeSafeAPIError as error:
                        assert error.status == 429
                    store.limit = False
                js = f"""
import assert from 'node:assert/strict';
import {{TypeSafeClient}} from {json.dumps(args.js_module.resolve().as_uri())};
const client=new TypeSafeClient({{apiKey:'valid-key',baseURL:{json.dumps(url)},defaultModel:'zils-shared',retry:{{maxRetries:0}}}});
const models=await client.models.list();
assert.deepEqual(models.map(x=>x.name).sort(),{json.dumps(sorted([RELEASE, "zils-shared"]))});
const result=await client.systemOne({{state:{{product:'USB cable'}},questions:{json.dumps(questions)}}});
assert.equal(result.model,{json.dumps(RELEASE)});
assert.equal(result.usage.input_tokens,30);
assert.deepEqual(result.answers.s.legend,{{'0':{{level:'low'}},'1':['high']}});
const bad=new TypeSafeClient({{apiKey:'invalid-key',baseURL:{json.dumps(url)},defaultModel:'zils-shared',retry:{{maxRetries:0}}}});
await assert.rejects(()=>bad.models.list(),e=>e.status===401);
console.log('Official JavaScript SDK 0.6.0: models, all answer types, structured legends, authentication passed.');
"""
                subprocess.run(["node", "--input-type=module", "-e", js], check=True)
                print(
                    "Official Python SDK 0.7.2: models, all answer types, structured legends, rate limits passed."
                )
    finally:
        fixture.tearDown()


if __name__ == "__main__":
    main()
