"""Zils Current nightly run: collect since the cursors, resolve, export finished days.

    python -m zils.current nightly --state DIR --bank DIR --reference EXAM.jsonl ...

Cursors in `DIR/cursor.json` advance one GH Archive hour, one sampled Wikimedia hour and one
exported day at a time and are saved after each step, so a stopped run resumes and a repeated
run adds nothing. Every run appends one line to `runs.jsonl` and rewrites `stats.json`.
"""

import fcntl
import json
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import current_bank, current_github, current_wikimedia
from .current_items import iso, parse_time

DAY = "%Y-%m-%d"
WIKI_RETENTION_HOURS = 6 * 24  # the stream keeps at least 7 days; stay inside it


def _hour(moment):
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _archive_hour(moment):
    return f"{moment:%Y-%m-%d}-{moment.hour}"


def _from_archive_hour(text):
    day, hour = text.rsplit("-", 1)
    return datetime.strptime(day, DAY).replace(tzinfo=UTC) + timedelta(hours=int(hour))


class Cursor:
    def __init__(self, path):
        self.path = path
        self.values = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.values, indent=1, sort_keys=True) + "\n")
        temporary.replace(self.path)


def _gharchive(store, http, cursor, now, first_hours, max_hours):
    """Every archive hour after the cursor; the archive publishes an hour after it ends."""
    last = _hour(now) - timedelta(hours=2)
    done = cursor.get("gharchive")
    hour = (
        _from_archive_hour(done) + timedelta(hours=1)
        if done
        else _hour(now) - timedelta(hours=first_hours)
    )
    collected, added = [], 0
    while hour <= last and len(collected) < max_hours:
        name = _archive_hour(hour)
        try:
            added += current_github.collect(store, http, [name], now=now)
        except ValueError as error:
            if "not published" in str(error):
                break
            raise
        cursor.set("gharchive", name)
        collected.append(name)
        hour += timedelta(hours=1)
    return {"hours": collected, "added": added}


def _wikimedia(store, http, cursor, now, first_hours, max_hours, minutes, wikis):
    """Replay the first `minutes` of every hour after the cursor, a sample spread over the day."""
    last = _hour(now) - timedelta(hours=1)
    oldest = _hour(now) - timedelta(hours=WIKI_RETENTION_HOURS)
    done = cursor.get("wikimedia")
    hour = (
        parse_time(done) + timedelta(hours=1) if done else _hour(now) - timedelta(hours=first_hours)
    )
    hour = max(hour, oldest)
    sampled, added = [], 0
    while hour <= last and len(sampled) < max_hours:
        added += current_wikimedia.collect(
            store, http, minutes, now=now, wikis=wikis, since=iso(hour)
        )
        cursor.set("wikimedia", iso(hour))
        sampled.append(iso(hour))
        hour += timedelta(hours=1)
    return {"hours": sampled, "added": added}


def _exports(store, bank, cursor, now, options):
    """Every finished UTC day after the export cursor, oldest first (yesterday on a first run)."""
    yesterday = (now.astimezone(UTC) - timedelta(days=1)).date()
    done = cursor.get("export")
    day = datetime.strptime(done, DAY).date() + timedelta(days=1) if done else yesterday
    written = []
    while day <= yesterday:
        date = day.strftime(DAY)
        manifest = current_bank.export_bank(store, bank, date, **options)
        cursor.set("export", date)
        written.append({"date": date, "exported": manifest["files"]["items.jsonl"]["rows"]})
        day += timedelta(days=1)
    return written


def run(
    state,
    bank,
    http,
    now,
    *,
    references=(),
    check_overlap=True,
    gh_first_hours=24,
    gh_max_hours=48,
    wiki_first_hours=24,
    wiki_max_hours=48,
    wiki_minutes=2,
    wikis=None,
    max_age_days=14,
    export_options=None,
):
    from . import current

    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "nightly.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy"}
        store, cursor = current.Store(state), Cursor(state / "cursor.json")
        result = {"started_at": iso(now), "collected": {}, "errors": []}

        def step(name, action):
            try:
                return action()
            except Exception as error:  # noqa: BLE001 - logged; later steps still run
                result["errors"].append(
                    {
                        "step": name,
                        "error": f"{type(error).__name__}: {error}",
                        "trace": traceback.format_exc(limit=3),
                    }
                )
                return None

        result["collected"]["gharchive"] = step(
            "gharchive", lambda: _gharchive(store, http, cursor, now, gh_first_hours, gh_max_hours)
        )
        result["collected"]["wikimedia"] = step(
            "wikimedia",
            lambda: _wikimedia(
                store, http, cursor, now, wiki_first_hours, wiki_max_hours, wiki_minutes, wikis
            ),
        )
        result["resolve"] = step("resolve", lambda: current.resolve(store, http, now, max_age_days))
        options = {
            "references": references,
            "check_overlap": check_overlap,
            **(export_options or {}),
        }
        result["exports"] = (
            step("export", lambda: _exports(store, bank, cursor, now, options)) or []
        )
        _, streams, _ = current_bank.prepare(store, balance=False)
        stats = {"at": iso(now), "streams": streams}
        (state / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
        result["finished_at"] = iso(datetime.now(UTC))
        with (state / "runs.jsonl").open("a") as log:
            log.write(json.dumps(result, sort_keys=True) + "\n")
        return result
