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

from . import current_bank, current_check, current_github, current_items, current_wikimedia
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


def _afd(store, http, cursor, now, first_days):
    """Every Articles for deletion log day after the cursor that is old enough to be closed."""
    from . import current_afd

    days = current_afd.log_days(now, cursor.get("afd"), first_days or current_afd.FIRST_DAYS)
    added = 0
    for day in days:
        added += current_afd.collect(store, http, now, days=[day])["added"]
        cursor.set("afd", day)
    return {"days": days, "added": added}


def _days(store, http, cursor, now, first_days, module, key, days_of):
    """Every day after the cursor that a module can read; the cursor moves after each day."""
    days = days_of(now, cursor.get(key), first_days or module.FIRST_DAYS)
    added = 0
    for day in days:
        added += module.collect(store, http, now, days=[day])["added"]
        cursor.set(key, day)
    return {"days": days, "added": added}


def _afc(store, http, cursor, now, first_days):
    """Every finished Articles for Creation review day after the cursor."""
    from . import current_afc

    return _days(store, http, cursor, now, first_days, current_afc, "afc", current_afc.review_days)


def _arxiv(store, http, cursor, now, first_days):
    """Every announced arXiv submission day after the cursor."""
    from . import current_arxiv

    return _days(store, http, cursor, now, first_days, current_arxiv, "arxiv", current_arxiv.days)


def _exports(store, bank, cursor, now, options, check=None):
    """Every finished UTC day after the export cursor, oldest first (yesterday on a first run).

    `check(folder)` runs the proof and cross-check gates on each written day."""
    yesterday = (now.astimezone(UTC) - timedelta(days=1)).date()
    done = cursor.get("export")
    day = datetime.strptime(done, DAY).date() + timedelta(days=1) if done else yesterday
    written = []
    while day <= yesterday:
        date = day.strftime(DAY)
        manifest = current_bank.export_bank(store, bank, date, **options)
        entry = {"date": date, "exported": manifest["files"]["items.jsonl"]["rows"]}
        if check:
            report = check(Path(bank) / date)
            entry["crosscheck"] = {"excluded": report.get("excluded", [])}
        cursor.set("export", date)
        written.append(entry)
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
    wiki_source="stream",
    wiki_samples=48,
    negatives_per_positive=None,
    afd_first_days=None,
    afc_first_days=None,
    arxiv_first_days=None,
    se_sites=None,
    se_pages=None,
    crosscheck_sample=0,
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
        recent = {"samples": wiki_samples, "min_days": 2, "max_days": 3, "wikis": wikis}
        result["collected"]["wikimedia"] = (
            step(
                "wikimedia",
                lambda: current_wikimedia.collect_recent(store, http, now, **recent),
            )
            if wiki_source == "recentchanges"
            else step(
                "wikimedia",
                lambda: _wikimedia(
                    store, http, cursor, now, wiki_first_hours, wiki_max_hours, wiki_minutes, wikis
                ),
            )
        )
        if afd_first_days != 0:
            result["collected"]["afd"] = step(
                "afd", lambda: _afd(store, http, cursor, now, afd_first_days)
            )
        if afc_first_days != 0:
            result["collected"]["afc"] = step(
                "afc", lambda: _afc(store, http, cursor, now, afc_first_days)
            )
        if arxiv_first_days != 0:
            result["collected"]["arxiv"] = step(
                "arxiv", lambda: _arxiv(store, http, cursor, now, arxiv_first_days)
            )
        if se_pages != 0:
            result["collected"]["stackexchange"] = step(
                "stackexchange",
                lambda: current.stackexchange(store, http, now, sites=se_sites, pages=se_pages),
            )
        result["resolve"] = step(
            "resolve",
            lambda: current.resolve(
                store, http, now, max_age_days, negatives_per_positive=negatives_per_positive
            ),
        )
        options = {
            "references": references,
            "check_overlap": check_overlap,
            **(export_options or {}),
        }
        held = current_check.paused(state)
        if held:
            known = set(current_items.STREAMS) - set(held)
            if options.get("publishable"):
                known &= set(current_bank.publishable_streams())
            options["streams"] = sorted(known & set(options.get("streams") or known))

        def check(folder):
            report = current_check.run(folder, http, sample=crosscheck_sample, now=now)
            current_check.update_paused(state, report)
            return report

        result["exports"] = (
            step(
                "export",
                lambda: _exports(
                    store, bank, cursor, now, options, check if crosscheck_sample else None
                ),
            )
            or []
        )
        result["paused"] = current_check.paused(state)
        _, streams, _ = current_bank.prepare(store, balance=False)
        stats = {"at": iso(now), "streams": streams}
        (state / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
        result["finished_at"] = iso(datetime.now(UTC))
        with (state / "runs.jsonl").open("a") as log:
            log.write(json.dumps(result, sort_keys=True) + "\n")
        return result
