"""Zils Current Gerrit stream: will this proposed code change be merged?"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlsplit

from tests.test_current import FakeHttp, at
from zils import (
    current,
    current_bank,
    current_check,
    current_gerrit,
    current_items,
    current_nightly,
)

NOW = at("2026-10-31T12:00:00Z")
DAY = "2026-10-26"
HOST = "opendev"
BASE = current_gerrit.HOSTS[HOST][0]


def account(name, username, *, service=False):
    row = {"_account_id": abs(hash(username)) % 10**6, "name": name, "username": username}
    row["email"] = f"{username}@example.org"
    if service:
        row["tags"] = ["SERVICE_USER"]
    return row


ADA = account("Ada Lovelace", "ada")


def change(
    number,
    status,
    *,
    subject="Add retry to the volume attach path",
    body="Attaching a volume fails when the backend is slow.\n\nRetry three times.",
    owner=ADA,
    created="2026-10-24 09:00:00.000000000",
    outcome="2026-10-26 15:00:00.000000000",
    reason="This is no longer needed",
    later_message=None,
):
    first = (
        f"{subject}\n\n{body}\n\n"
        f"Closes-Bug: #2101\nChange-Id: I{number:040d}\n"
        f"Signed-off-by: {owner['name']} <{owner['email']}>\n"
        "(cherry picked from commit 0123456789abcdef0123456789abcdef01234567)\n"
    )
    revisions = {
        f"sha{number}a": {
            "_number": 1,
            "created": created,
            "commit": {"subject": subject, "message": first, "author": {"name": owner["name"]}},
            "uploader": owner,
        },
        f"sha{number}b": {
            "_number": 2,
            "created": created,
            "commit": {"subject": subject, "message": later_message or first + "\nReviewed-by: X"},
            "uploader": owner,
        },
    }
    messages = [{"date": created, "message": "Uploaded patch set 1.", "_revision_number": 1}]
    row = {
        "_number": number,
        "project": "openstack/nova",
        "branch": "master",
        "status": status,
        "subject": subject,
        "owner": owner,
        "created": created,
        "updated": outcome,
        "revisions": revisions,
        "messages": messages,
    }
    if status == "MERGED":
        row["submitted"] = outcome
        messages.append({"date": outcome, "message": "Change has been successfully merged"})
    else:
        messages.append(
            {"date": outcome, "message": f"Abandoned\n\n{reason}" if reason else "Abandoned"}
        )
    return row


BOB = account("Bob Builder", "bob")
CHANGES = {
    "merged": [
        change(1, "MERGED"),
        change(2, "MERGED", owner=account("Build Worker", "builder", service=True)),
        change(3, "MERGED", outcome="2026-10-27 01:00:00.000000000"),  # another day's outcome
        change(4, "MERGED", later_message="Rewritten subject\n\nAfter review."),
    ],
    "abandoned": [
        change(10, "ABANDONED", subject="Drop the legacy scheduler filter"),
        change(11, "ABANDONED", reason="Superseded by https://review.opendev.org/c/x/+/9"),
        change(12, "ABANDONED", subject="[WIP] try a new cache"),
        change(13, "ABANDONED", subject="DNM: test the gate"),
        change(14, "ABANDONED", created="2026-10-26 14:57:00.000000000"),  # three minutes old
        *(change(20 + i, "ABANDONED", owner=BOB, subject=f"Clean up module {i}") for i in range(7)),
    ],
}


class GerritHttp(FakeHttp):
    """Answers change queries (by status, a page at a time) and change details."""

    def __init__(self, changes=None, page=None, details=None):
        super().__init__()
        self.changes = changes or CHANGES
        self.page = page
        self.details = details or {}

    def get_bytes(self, url):
        self.calls.append((url, {}))
        parts = urlsplit(url)
        if parts.path.endswith("/changes/"):
            query = parse_qs(parts.query)
            status = query["q"][0].split()[0].removeprefix("status:")
            rows = self.changes.get(status, []) if url.startswith(BASE) else []
            start, size = int(query.get("S", ["0"])[0]), self.page or len(rows) or 1
            chunk = [dict(r) for r in rows[start : start + size]]
            if chunk and start + size < len(rows):
                chunk[-1]["_more_changes"] = True
            return 200, (")]}'\n" + json.dumps(chunk)).encode()
        if "/detail" in parts.path:
            key = unquote(parts.path.split("/changes/")[1].split("/detail")[0])
            if key in self.details:
                return 200, (")]}'\n" + json.dumps(self.details[key])).encode()
        return 404, b"Not found"


def collect(tmp, http=None):
    store = current.Store(tmp)
    http = http or GerritHttp()
    with patch.object(current_gerrit, "sleep", lambda s: None):
        result = current_gerrit.collect(store, http, NOW, days=[DAY], hosts=[HOST])
    current.resolve(store, http, now=NOW)
    return store, http, result


def by_number(store):
    return {r["source"]["number"]: r for r in store.resolved()}


class CollectTest(unittest.TestCase):
    def test_merged_and_abandoned_changes_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = collect(tmp)[0]
            records = by_number(store)
            dropped = {r["source"]["number"]: r["reason"] for r in store.dropped()}
        self.assertEqual(records[1]["label"], "true")
        self.assertEqual(records[10]["label"], "false")
        self.assertEqual(records[1]["stream"], "gerrit-merge")
        self.assertEqual(records[1]["asked_at"], "2026-10-24T09:00:00Z")
        self.assertEqual(records[1]["outcome_at"], "2026-10-26T15:00:00Z")
        self.assertEqual(records[1]["source"]["url"], f"{BASE}/c/openstack/nova/+/1")
        self.assertEqual(
            dropped,
            {
                2: "bot-owner",
                11: "superseded",
                12: "not-for-merge",
                13: "not-for-merge",
                14: "too-quick",
                **{25: "owner-cap", 26: "owner-cap"},
            },
        )
        self.assertNotIn(3, records)  # its outcome belongs to another day

    def test_an_owner_gives_at_most_the_cap_per_outcome_and_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = by_number(collect(tmp)[0])
        bobs = [n for n in records if 20 <= n < 27]
        self.assertEqual(bobs, [20, 21, 22, 23, 24])
        self.assertEqual(current_gerrit.OWNER_CAP, 5)

    def test_text_is_the_first_patch_set_without_trailers_or_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = by_number(collect(tmp)[0])
        text = current_items.build(records[4])["state"]["text"]
        self.assertEqual(
            text,
            "Code review: OpenStack (OpenDev)\nProject: openstack/nova\nBranch: master\n\n"
            "Commit message:\nAdd retry to the volume attach path\n\n"
            "Attaching a volume fails when the backend is slow.\n\nRetry three times.",
        )
        for gone in (
            "Change-Id",
            "Signed-off-by",
            "Ada",
            "cherry picked",
            "Closes-Bug",
            "Rewritten",
        ):
            self.assertNotIn(gone, text)

    def test_case_is_a_yes_no_tools_question_with_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = current_items.build(by_number(collect(tmp)[0])[1])
        self.assertEqual(case["question"]["type"], "noul")
        self.assertEqual(case["category"], "tools")
        self.assertEqual(case["block"], "new")
        self.assertFalse(current_items.STREAMS["gerrit-merge"]["publishable"])
        self.assertIsNone(current_check.proof_reason(case))

    def test_queries_are_paged_and_spaced(self):
        waits = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(current_gerrit, "sleep", waits.append),
        ):
            http = GerritHttp(page=4)
            result = current_gerrit.collect(current.Store(tmp), http, NOW, days=[DAY], hosts=[HOST])
        starts = [parse_qs(urlsplit(u).query).get("S", ["0"])[0] for u, _ in http.calls]
        self.assertEqual(starts, ["0", "0", "4", "8"])
        self.assertEqual(len(waits), len(http.calls) - 1)
        self.assertTrue(all(w == current_gerrit.GAP_SECONDS for w in waits))
        self.assertEqual(result["changes"], 16)
        query = parse_qs(urlsplit(http.calls[0][0]).query)["q"][0]
        self.assertEqual(query, 'status:merged after:"2026-10-26" before:"2026-10-27"')

    def test_a_change_that_states_its_outcome_is_filtered(self):
        record = {"id": "x", "stream": "gerrit-merge", "data": {}}
        self.assertEqual(
            current_bank.leak_reason(record, "Commit message:\nThis was superseded by I12."),
            "states-outcome",
        )

    def test_days_wait_for_late_restores(self):
        self.assertEqual(current_gerrit.days(NOW, "2026-10-26"), ["2026-10-27", "2026-10-28"])
        self.assertEqual(len(current_gerrit.days(NOW, None)), current_gerrit.FIRST_DAYS)


class GerritCheckTest(unittest.TestCase):
    def test_crosscheck_reads_the_change_history_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [current_items.build(r) for r in collect(tmp)[0].resolved()]
        chosen = [c for c in cases if c["source"]["number"] in (1, 10, 20)]
        details = {
            "openstack/nova~1": change(1, "MERGED"),
            "openstack/nova~10": change(10, "ABANDONED"),
            # Restored and merged after we labelled it abandoned.
            "openstack/nova~20": change(20, "MERGED", owner=BOB),
        }
        with patch.object(current_gerrit, "sleep", lambda s: None):
            results = current_check.check_gerrit(chosen, GerritHttp(details=details))
        agree = {c["source"]["number"]: r["agree"] for c, r in zip(chosen, results, strict=True)}
        self.assertEqual(agree, {1: True, 10: True, 20: False})

    def test_crosscheck_dispatches_the_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [current_items.build(r) for r in collect(tmp)[0].resolved()]
        with patch.object(current_gerrit, "sleep", lambda s: None):
            streams = current_check.crosscheck(cases, GerritHttp(), sample=3)
        self.assertEqual(streams["gerrit-merge"]["method"], "change history (submitted, Abandoned)")


class GerritNightlyTest(unittest.TestCase):
    def test_nightly_reads_days_from_the_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, cursor = current.Store(tmp), current_nightly.Cursor(Path(tmp) / "cursor.json")
            cursor.set("gerrit", "2026-10-26")
            with patch.object(current_gerrit, "sleep", lambda s: None):
                result = current_nightly._gerrit(store, GerritHttp(), cursor, NOW, None)
            self.assertEqual(result["days"], ["2026-10-27", "2026-10-28"])
            self.assertEqual(cursor.get("gerrit"), "2026-10-28")


if __name__ == "__main__":
    unittest.main()
