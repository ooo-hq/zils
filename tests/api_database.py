"""Run the actual gateway/worker against disposable PostgreSQL and a fixture model.

Only the Supabase transport and object store are substituted. RPC bodies are the
real migrations, so no queue/admission state machine is duplicated in this test.
Invoked by scripts.check_queue_db, not unittest discovery.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fez import decision_http
from fez.api import Gateway, Registry
from fez.api_store import Store
from fez.batches import BUCKET, Batches, Worker
from fez.cloud import APIError
from fez.decisions import DecisionError, option_descriptions
from tests.test_decision_http import server

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def literal(value):
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    return "'" + str(value).replace("'", "''") + "'"


def name(value):
    assert re.fullmatch("[a-z_]+", value), value
    return value


class Database:
    def __init__(self, command):
        self.command, self.objects = command, {}
        self.interrupt_commit = False

    def sql(self, sql):
        proc = subprocess.run([*self.command, "-qAt", "-c", sql], text=True, capture_output=True)
        if proc.returncode:
            raise APIError(409, "Database rejected test operation: " + proc.stderr)
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def rpc(self, function, args):
        if (
            self.interrupt_commit
            and function == "zils_api_batch_work"
            and args["p_action"] == "result"
        ):
            self.interrupt_commit = False
            raise APIError(503, "Simulated connection lost after GPU execution")
        call = (
            name(function)
            + "("
            + ",".join(name(k) + "=>" + literal(v) for k, v in args.items())
            + ")"
        )
        if function == "zils_api_auth":
            return self.sql("select coalesce(jsonb_agg(t),'[]'::jsonb) from " + call + " t")
        value = self.sql("select to_jsonb(" + call + ")")
        return value

    def select(self, table, query):
        clauses, order, limit, columns = [], "", "", "*"
        for key, values in parse_qs(query).items():
            value = values[0]
            if key == "select":
                columns = ",".join(name(x) for x in value.split(","))
            elif key == "order":
                field, _, direction = value.partition(".")
                order = " order by " + name(field) + (" desc" if direction == "desc" else "")
            elif key == "limit":
                limit = " limit " + str(int(value))
            elif value in ("is.null", "not.is.null"):
                clauses.append(name(key) + (" is null" if value == "is.null" else " is not null"))
            else:
                op, val = value.split(".", 1)
                clauses.append(name(key) + {"eq": "=", "gt": ">", "lt": "<"}[op] + literal(val))
        where = " where " + " and ".join(clauses) if clauses else ""
        return columns, where, order + limit

    def rows(self, table, query=""):
        columns, where, suffix = self.select(table, query)
        return self.sql(
            f"select coalesce(jsonb_agg(t),'[]'::jsonb) from (select {columns} from {name(table)}{where}{suffix}) t"
        )

    def patch(self, table, query, values):
        _, where, _ = self.select(table, query)
        assignments = ",".join(name(k) + "=" + literal(v) for k, v in values.items())
        return self.sql(
            f"with updated as (update {name(table)} set {assignments}{where} returning *) select coalesce(jsonb_agg(updated),'[]'::jsonb) from updated"
        )

    def user(self, token):
        if token not in ("session-owner", "session-other"):
            raise APIError(401, "Invalid session")
        return OWNER if token == "session-owner" else OTHER

    def signed(self, bucket, path, *, upload=False):
        return {"url": "https://storage.example/" + path, "method": "PUT", "headers": {}}

    def exists(self, bucket, path):
        return path in self.objects

    def download(self, bucket, path, destination, limit):
        raw = self.objects[path]
        if len(raw) > limit:
            raise ValueError("over limit")
        Path(destination).write_bytes(raw)

    def request(self, method, path, body=None):
        assert method == "DELETE"
        if path == "/storage/v1/object/" + BUCKET:
            for key in body["prefixes"]:
                self.objects.pop(key, None)
        else:
            parsed = urlsplit(path)
            _, where, _ = self.select(parsed.path.rsplit("/", 1)[-1], parsed.query)
            self.sql("delete from " + name(parsed.path.rsplit("/", 1)[-1]) + where)


def run(command):
    db, executions = Database(command), []
    store, batches = Store(db), Batches(db)
    issued = store.create_key(OWNER, "integration")
    rotated = store.create_key(OWNER, "rotation")
    store.create_key(OTHER, "other")
    token = issued["key"]
    assert {issued["id"], rotated["id"]} <= {row["id"] for row in store.list_keys(OWNER)}
    os.environ["ZILS_DB_TEST_TOKEN"] = "runtime-fixture"
    release, fingerprint = "zils-fixture-r1", "a" * 64

    def runtime(method, path, bearer, body, rid):
        assert bearer == "runtime-fixture"
        identity = {"release_id": release, "fingerprint": fingerprint}
        if path == "/v1/prepare":
            return 200, {**identity, "reserved_tokens": 100}
        executions.append(body["request"])
        predictions = {
            qid: {
                "input_tokens": 10,
                "probabilities": {
                    key: 1 / len(option_descriptions(q)) for key in option_descriptions(q)
                },
            }
            for qid, q in body["request"]["questions"].items()
        }
        return 200, {**identity, "predictions": predictions}

    with server(decision_http, runtime) as port:
        entry = {
            "id": release,
            "fingerprint": fingerprint,
            "aliases": ["shared"],
            "owners": None,
            "url": f"http://127.0.0.1:{port}",
            "token_env": "ZILS_DB_TEST_TOKEN",
            "release_date": "2026-10-04",
            "description": "Fixture only",
        }
        gateway = Gateway(store, Registry([entry]), batches)
        worker = Worker(db, gateway)

        def call(method, path, body=None, credential=None):
            return gateway.dispatch(
                method, path, credential or token, body or {}, str(uuid.uuid4())
            )

        body = {
            "state": "product",
            "model": "shared",
            "questions": {"q": {"type": "choice", "criteria": {"zebra": None, "apple": None}}},
        }
        rows = [
            {"custom_id": "one", "body": body},
            {"custom_id": "invalid", "body": {**body, "questions": {}}},
            {"custom_id": "three", "body": body},
        ]

        def create(key, records):
            _, batch = call("POST", "/v1/batches", {"idempotency_key": key})
            assert call("POST", "/v1/batches", {"idempotency_key": key})[1]["id"] == batch["id"]
            try:
                call("POST", "/v1/batches/" + batch["id"] + "/submit")
                raise AssertionError("missing upload submitted")
            except DecisionError as error:
                assert error.status == 409
            path = db.rows("zils_api_batches", "id=eq." + batch["id"])[0]["input_path"]
            db.objects[path] = "\n".join(json.dumps(x) for x in records).encode()
            assert call("POST", "/v1/batches/" + batch["id"] + "/submit")[0] == 202
            return batch["id"], path

        batch, path = create("flow", rows)
        # Freeze aliases at submit. The approved immutable release stays available.
        gateway.registry = Registry([{**entry, "aliases": []}])
        assert worker.once()  # validation
        store.revoke_key(OWNER, issued["id"])
        token = rotated["key"]
        try:
            call("GET", "/v1/batches/" + batch, credential=issued["key"])
            raise AssertionError("revoked key read batch")
        except DecisionError as error:
            assert error.status == 401
        # A new process can continue with only database/storage state.
        worker = Worker(db, gateway)
        assert worker.once()
        partial = call("GET", "/v1/batches/" + batch)[1]
        assert partial["completed"] == 1 and partial["failed"] == 1, partial
        call("POST", "/v1/batches/" + batch + "/cancel")
        results = call("GET", "/v1/batches/" + batch + "/results")[1]["data"]
        assert len(results) == 2
        assert results[0]["response"]["answers"]["q"]["choice"] == "zebra", results
        assert not worker.once()
        assert len(executions) == 1
        # Replay after a crash may recompute, but cannot double-commit results/usage.
        gateway.registry = Registry([entry])
        second, second_path = create("crash", rows[:1])
        worker.once()
        db.interrupt_commit = True
        worker.once()
        db.sql(
            "update zils_api_batches set lease_until=now()-interval '1 second' where id="
            + literal(second)
        )
        worker = Worker(db, gateway)
        worker.once()
        result = call("GET", "/v1/batches/" + second + "/results")[1]
        assert len(result["data"]) == 1 and result["status"] == "completed"
        assert len(executions) == 3
        with (
            server(decision_http, gateway.dispatch) as gateway_port,
            tempfile.TemporaryDirectory() as tmp,
        ):
            export = Path(tmp) / "results.jsonl"
            downloaded = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.download_batch",
                    "--url",
                    f"http://127.0.0.1:{gateway_port}",
                    "--batch",
                    second,
                    "--out",
                    str(export),
                ],
                env={**os.environ, "ZILS_API_KEY": token},
                capture_output=True,
                text=True,
            )
            assert downloaded.returncode == 0, downloaded.stderr
            assert json.loads(export.read_text())["custom_id"] == "one"
            assert export.stat().st_mode & 0o777 == 0o600
        assert (
            db.sql(
                "select count(*) from zils_api_usage where owner_id="
                + literal(OWNER)
                + " and status='completed' and input_tokens=10"
            )
            == 2
        )
        # Duplicate IDs fail before model execution.
        duplicate, _ = create("duplicate", [rows[0], rows[0]])
        worker.once()
        assert call("GET", "/v1/batches/" + duplicate)[1]["status"] == "failed"
        assert len(executions) == 3
        # Retention removes both input bytes and private records.
        db.sql(
            "update zils_api_batches set finished_at=now()-interval '8 days' where id="
            + literal(second)
        )
        worker.cleanup()
        assert second_path not in db.objects
        assert db.rows("zils_api_batch_items", "batch_id=eq." + second) == []
        try:
            call("GET", "/v1/batches/" + second + "/results")
            raise AssertionError("expired results exposed")
        except DecisionError as error:
            assert error.status == 410
    os.environ.pop("ZILS_DB_TEST_TOKEN")
    # Independent processes/keys share atomic budgets; assert by the actual window
    # returned under the admission lock, so a slow CI second boundary is harmless.
    another = store.create_key(OWNER, "concurrent")
    db.sql(
        "update zils_api_accounts set requests_per_second=5,tokens_per_second=50 where owner_id="
        + literal(OWNER)
    )
    db.sql("delete from zils_api_windows where owner_id=" + literal(OWNER))
    db.sql("""create function public.test_api_admit(k uuid) returns jsonb language plpgsql as $$
      declare outcome text; admission_window zils_api_windows;
      begin
        outcome:=zils_api_admit('11111111-1111-4111-8111-111111111111',k,gen_random_uuid(),10);
        select * into admission_window from zils_api_windows where owner_id='11111111-1111-4111-8111-111111111111';
        return jsonb_build_object('status',outcome,'second',admission_window.second,'requests',admission_window.requests,'tokens',admission_window.tokens);
      end $$""")

    def admission(i):
        return db.rpc("test_api_admit", {"k": rotated["id"] if i % 2 else another["id"]})

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(admission, range(20)))
    allowed = {}
    for outcome in outcomes:
        assert outcome["requests"] <= 5 and outcome["tokens"] <= 50, outcome
        if outcome["status"] == "allowed":
            second = outcome["second"]
            allowed[second] = allowed.get(second, 0) + 1
    assert all(n <= 5 for n in allowed.values()), outcomes
    assert any(x["status"] == "limited" for x in outcomes), outcomes
    print(
        "Real database gateway/worker: rotation, frozen aliases, restart, crash replay, cancellation, retention and 20-way admission passed."
    )
