# Zils Current

Real-world decisions, labelled by what actually happened.

Zils Current (`python -m zils.current`) turns public events into decision questions and
labels each one with what really happened later. A GitHub issue is opened; days later a
maintainer labels it a bug or a feature request. A Wikipedia edit is saved; within days it is
reverted or it stays. The text available when the event happened is the question, and the
outcome is the label. Nothing is decided by a language model.

Because every item describes an event that happened after a model was trained, no model can have
seen it. Any validator or miner can run the tool on a laptop or server with network access and
rebuild the same items from the same public sources.

It writes local files only and never uploads or publishes anything. A soup pool can draw its
round items from the dated exports (see [Validators](#validators)). Teacher labels are not part
of it yet (see [Extension points](#extension-points)).

## Streams

| Stream | Question | Type | Label | Source of the label |
| --- | --- | --- | --- | --- |
| `github-issue-type` | Is this new issue a bug report or a feature request? | choice | `bug` / `feature` | the issue's labels (`bug`, `type: bug`, `enhancement`, `feature request`, ...) |
| `github-issue-close` | Will maintainers close it as completed rather than not planned or duplicate? | yes/no | close reason | `state_reason` of the closed issue |
| `github-pr-merge` | Will this new pull request be merged? | yes/no | merged or closed unmerged | the pull request's `merged` flag |
| `wikipedia-revert` | Will this article edit be reverted? | yes/no | `mw-reverted` tag | MediaWiki change tags after a waiting period |

The state text holds only what was known when the event happened:

- **Issues:** the repository name without its owner, the title and the body, from the
  GH Archive `opened` event. Labels are not shown.
- **Pull requests:** the repository name, the head and base branch names, the title and the
  description. GitHub's event feed no longer carries pull request text, so the title and
  description are read from the REST API at the first resolution pass (`data.text_at`). If you
  collect an hour that is days old, the text may include edits made after the outcome.
- **Wikipedia edits:** the article title, the edit summary, the size change, the minor-edit flag,
  the kind of editor (registered account, temporary account or IP address) and the removed and
  added lines of the diff.

## Run it

```bash
python -m zils.current collect --source gharchive --start 2026-10-26-15 --hours 1
python -m zils.current collect --source wikimedia --minutes 5
python -m zils.current collect --source wikimedia --minutes 5 --since 2026-10-28T12:00:00Z
GITHUB_TOKEN=... python -m zils.current resolve
python -m zils.current export --out items.jsonl --stats
python -m zils.current export --bank BANK --date 2026-10-30 --reference EXAMS/ --reference POOL.jsonl
python -m zils.current nightly --state STATE --bank BANK --reference EXAMS/
python -m zils.current grade --items BANK/2026-10-30/items.jsonl --model CHECKPOINT
```

- GH Archive hours are UTC and written without a leading zero (`2026-10-26-5`). Each hour is
  published shortly after it ends. To resolve straight away, collect hours that are several days
  old.
- `--source wikimedia` reads the live stream for `--minutes` of wall time. With `--since`, it
  replays that much stream time from an earlier moment. The stream keeps between 7 and 31 days.
  `--wiki` adds wikis (database names such as `enwiki`; the default is English Wikipedia).
- `resolve` labels items whose outcome is known and drops items still waiting after
  `--max-age-days` (default 14, or `ZILS_CURRENT_MAX_AGE_DAYS`). Wikipedia edits are checked once
  they are `--wiki-wait-hours` old (default 48). Run it again later and it continues where it
  stopped. With `GITHUB_TOKEN` (or `ZILS_GITHUB_TOKEN`), lookups go through GitHub's GraphQL API
  50 at a time, so one request replaces fifty. Without a token, they use the REST API, which
  allows 60 unauthenticated requests an hour. Each lookup resolves every stream built from the
  same issue. When GitHub signals a rate limit, the pass stops and the remaining items stay
  pending.
- `export --out FILE` writes every resolved item to one file. `export --bank DIR` writes one
  dated folder (see [Dated exports](#dated-exports)). Both apply the
  [leak filters](#leak-filters). `--max-tokens 8192` allows longer states for 8,192-token models
  (default 4,096). `--no-balance` keeps every label. `--seed` changes the downsampling. `--stats`
  prints per-stream counts.
- Set `ZILS_CURRENT_CONTACT` to your e-mail address or URL. The tool sends it in its User-Agent
  (`zils-current/v1 (<contact>)`), as Wikimedia's policy asks.

Storage is a local directory: `--dir`, `ZILS_CURRENT_DIR`, or `runs/current` in the checkout, which
Git ignores. It holds three JSON-lines files:

| File | Contents |
| --- | --- |
| `pending.jsonl` | the waiting room: `id`, `stream`, `asked_at`, `collected_at`, `source`, scrubbed `data` |
| `resolved.jsonl` | the same, plus `resolved_at`, `label` and, where the source records it, `outcome_at` |
| `dropped.jsonl` | items that will never be used, with a `reason` (`expired`, `deleted`, `bot`, `closed-by-bot`, `is-revert`, `ambiguous-label`, `no-close-reason`) |

Raw downloads are not kept. Run one process per directory at a time.

## Dated exports

`export --bank DIR --date YYYY-MM-DD` writes the items resolved on that UTC day. The default day
is yesterday.

```text
DIR/2026-10-30/items.jsonl     cases, one per line, sorted keys
DIR/2026-10-30/manifest.json   what the folder holds and how it was made
```

The manifest records:

- the export and collector versions;
- the day's window (`resolved_from`, `resolved_until`) and the span of `asked_at`;
- the mode (`local` or `publishable`, see [Licence gate](#licence-gate)) and every setting
  (streams, leak filters with their phrase lists, seed, balance, token cap);
- per stream: its licence, whether it may be published, and counts resolved, filtered (by
  reason), duplicated, overlapping, exported, and per label;
- `counts`, the exported items per stream and label;
- the overlap check: whether it ran, each reference file's name, SHA-256 and row counts, and
  every hit (item id and kind);
- the SHA-256, size and row count of `items.jsonl`.

An existing day is left untouched, so reruns are idempotent. `--force` rewrites it. Every
resolved item belongs to exactly one day, so no item appears in two exports.

## Leak filters

Leak filters are on by default; `--no-leak-filters` turns them off, and the manifest records
which. An item is dropped when:

- **`late-text`:** its text was read after the outcome (`data.text_at` later than
  `outcome_at`). This hits pull requests collected from old archive hours, whose description
  can be edited after the merge or close.
- **`label-at-open`:** it is a `github-issue-type` item whose type label was already present
  when the issue was opened. Templates set these labels, and the template headings usually give
  the answer away.
- **`states-outcome`:** its state text contains a phrase that states the outcome, such as
  "closed without merging", "was merged", "superseded by" or "won't merge" for pull requests;
  "duplicate of", "closed as", "won't fix" or "fixed in #12" for issue closes; and "reverted",
  "undid", "rv" or "rollback" for Wikipedia edits. The lists are short and specific so that
  ordinary text such as "depends on #4 being merged first" stays in. `current_bank.OUTCOME_PHRASES`
  holds them, and the manifest copies them.

## Overlap check

Every bank export compares its items with reference sets: exam draws, the round bank, private
development exams, benchmark items and training pools. Pass them with `--reference` (a file or
a folder, repeatable) or `ZILS_CURRENT_REFERENCES` (paths separated by `:`). Reference sets
usually live on the validator host, not in this repository. Without references, the export
stops unless you pass `--no-overlap-check`, and the manifest then records `"checked": false`.

References are JSON lines, or JSON lists (or objects holding `cases`, `items` or `rows`). Rows
may hold `state` with `question` or `questions`. Rows that carry only `content_sha256`, as in
rebuild-only datasets, are compared on that hash alone, and the manifest counts them as
`hash_only_rows`. An item is dropped, and listed in `overlap.hits`, when:

- **`prompt`:** its prompt fingerprint, from `base_exam.fingerprints` (the same hashes that
  exam draws use against each other), matches a reference row;
- **`hash`:** its `content_sha256` matches a reference row's;
- **`state`:** its normalized state text equals a reference row's;
- **`passage`:** it shares a passage with a reference row. Passages are compared as 16-word
  windows, one window in four chosen by hash on both sides. Shared passages of about 50 words or
  more are found almost surely. Zils Current's own template lines (field labels, size changes,
  flags) are left out, so items never match each other on template wording alone.

## Licence gate

`--publishable` exports only streams whose licence allows republishing. Today that is
`wikipedia-revert` (CC BY-SA 4.0), and every case must carry its attribution link in
`source.url`. The GitHub streams are local-only. Asking for one in a publishable export, or
any GitHub case reaching one, is an error and writes nothing. The manifest records the mode.
GitHub text needs legal review before any commercial use or release (see
[Sources, licences and terms](#sources-licences-and-terms)).

## Nightly job

`nightly --state DIR --bank DIR` runs one pass:

1. **GH Archive:** collects every hour after the `gharchive` cursor, up to two hours ago (the
   archive publishes each hour shortly after it ends). A first run starts `--gh-first-hours`
   back (default 24). A run takes at most `--gh-max-hours` (default 48). An hour not yet
   published ends the step, and the next run tries it again.
2. **Wikimedia:** for every hour after the `wikimedia` cursor, up to the last full hour,
   replays the first `--wiki-minutes` (default 2) of that hour's stream. This samples about 50
   minutes a day, spread evenly over the day, instead of every edit. It stays within the
   stream's retention. `--wiki-first-hours` and `--wiki-max-hours` bound the step, as for
   GH Archive.
3. **Resolve:** one resolution pass over the waiting room.
4. **Export:** writes every finished UTC day after the `export` cursor, oldest first. A first
   run writes yesterday. The overlap and leak options are the same as for `export`.

Each cursor in `DIR/cursor.json` is saved after each hour or day, so a stopped run resumes where
it stopped, and a repeated run downloads and exports nothing new. A failing step is logged, and
the later steps still run. Every run appends one line to `DIR/runs.jsonl` (start, finish, hours
collected, resolution counts, exports, errors) and rewrites `DIR/stats.json` with per-stream
counts. A lock file stops two runs from overlapping. The command exits 1 if any step failed.

Example units (they install nothing by themselves). `/opt/zils` stands for your checkout,
`/var/lib/zils-current` for the job's own directory, and `zils-current` for a dedicated user
that owns only that directory:

```ini
# /etc/systemd/system/zils-current.service
[Unit]
Description=Zils Current: collect, resolve and export
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=zils-current
Group=zils-current
EnvironmentFile=/etc/zils-current.env
WorkingDirectory=/opt/zils
ExecStart=/opt/zils/.venv/bin/python -m zils.current nightly \
    --state /var/lib/zils-current/state --bank /var/lib/zils-current/bank
LimitNOFILE=65536
Nice=10
IOSchedulingClass=idle
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/lib/zils-current
TimeoutStartSec=4h
```

```ini
# /etc/systemd/system/zils-current.timer
[Unit]
Description=Run Zils Current daily

[Timer]
OnCalendar=*-*-* 12:30:00 UTC
RandomizedDelaySec=15min
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
# /etc/zils-current.env (readable by the zils-current user only)
ZILS_CURRENT_CONTACT=you@example.org
GITHUB_TOKEN=github_pat_...
ZILS_CURRENT_REFERENCES=/path/to/exam-draws:/path/to/round-bank.jsonl:/path/to/benchmark
```

The job has its own unit name, user and directory, and it neither starts, stops nor depends on
any other service. The reference paths only need to be readable by that user. 12:30 UTC is
far from the 03:00 UTC soup round. It matters little anyway: collection, resolution and export
use CPU and network only, never the GPU, and they take minutes. The time matters for grading,
which needs the GPU. It also matters that the job runs after midnight UTC, so the previous day
is complete.

## Grading

`grade --items DIR/<date>/items.jsonl --model CHECKPOINT` scores one export with one d3-family
checkpoint, through the path soup rounds use: `soup_grade.production_runner` for predictions and
`soup_grade.grade` with its per-key prediction cache (`--cache`, default `predictions/` next to
the items; `--key`, default the checkpoint folder's name). A stopped run resumes, and an item is
never predicted twice for the same key. The result, `grades-<key>.json` next to the items unless
`--out` is given, holds accuracy, Brier, uniform Brier, skill, confident errors and case counts
per stream and overall, plus the SHA-256 of the items file.

GPU work holds the shared lock that training and validation use (`--lock`, `ZILS_COMPUTE_LOCK`,
default `/var/lock/zils-training/cuda.lock`). If the lock stays busy for `--lock-wait` seconds
(default 600), nothing runs and the command exits 75, so a timer can try again later. Run it on
the GPU host after the nightly export, for example:

```bash
python -m zils.current grade --items /var/lib/zils-current/bank/$(date -u -d yesterday +%F)/items.jsonl \
    --model /path/to/stock-lite --key stock-lite
```

## Item format

Exported cases use the same format as the exam and bank files that `zils.validate_cases`,
`base_metrics` and `soup_exam` read:

```json
{"id": "wikipedia-revert-…", "family": "wikipedia-revert", "category": "language", "block": "new",
 "source_id": "wikipedia-revert-…", "state": {"text": "Wiki: English Wikipedia\nArticle: …"},
 "question": {"type": "noul", "instructions": "Will this Wikipedia article edit be reverted by another edit?",
              "criteria": {"true": "Reverted", "false": "Not reverted"}},
 "label": "false", "source": {"name": "wikimedia", "wiki": "enwiki", "rev_id": 1, "url": "https://…"},
 "asked_at": "…", "resolved_at": "…", "licence": "CC-BY-SA-4.0: …", "content_sha256": "…"}
```

Choice options are shuffled per item and renamed `A`, `B`, as in authored exams. The `category`
values are provisional. `resolved_at` is when this tool saw the outcome. `outcome_at` is when the
outcome happened, where the source records it (a merge or a close time). The stats report both
medians.

States fit the model's input cap without a tokenizer. The estimate counts 1.5 ASCII characters
per token, which matches dense wiki markup and code, and one token per non-ASCII character, as in
CJK text. It leaves 384 tokens for the question. Plain English therefore uses less than half of
the cap. Long bodies are cut at a paragraph or line break and marked `[truncated]`.

## Hygiene

- **Personal data.** Collection and resolution remove e-mail addresses, IPv4 and IPv6
  addresses, `@mentions`, wiki `User:` links, and the account names attached to the event (author,
  assignees, the owner of a personal repository, the Wikipedia editor). The editor's name and IP
  address are never stored. Organisation names stay. Scrubbing is pattern-based, so a name typed
  in free text that matches no account can remain.
- **Bots.** GitHub `[bot]` accounts and `Bot` users are skipped, and so are issues closed by bots
  (for example stale-issue closers). Wikipedia edits with the bot flag are skipped, along with
  edits that are themselves reverts.
- **Self-merges.** Pull requests that owners open in their own personal repositories are
  skipped, because those are merged almost without exception.
- **Deleted sources.** Issues and pull requests that return 404/410/451, and revisions that are
  missing or have hidden text, comment or user, are dropped.
- **Duplicates.** Export keeps the first case for each content hash (stream plus normalized
  state text).
- **Balance.** Export downsamples every label of a stream to the rarest label's count, with a
  seeded random choice. A stream that has seen only one outcome exports nothing.
- **Rebuilding.** Each case keeps its source ids and links: repository, number, GH Archive event
  id and URL for GitHub; wiki, revision ids and diff URL for Wikipedia.

## Sources, licences and terms

**GH Archive** (`data.gharchive.org`) republishes GitHub's public event feed in hourly files. Its
code is MIT-licensed. The project states no licence for the dataset and warns that it "includes
material that may be subject to third party rights".

**GitHub** content belongs to its authors (Terms of Service, section D.3). Making a repository
public licenses other users to view and fork it through GitHub (section D.5). That is not a
licence to republish issue or pull request text elsewhere. GitHub's Acceptable Use Policies
(section 7) name two permitted uses of information from the service: research with public,
non-personal information, provided resulting publications are open access, and archiving. They
forbid spam, selling personal information and recruiting, and they require compliance with
GitHub's Privacy Statement. API use must stay within rate limits, and tokens must not be shared
to raise them (section H).

> **Commercial use is not cleared for the GitHub streams.** Using issue and pull request text to
> train or evaluate a commercial model, or republishing that text, is not clearly permitted by
> GitHub's terms, and the text carries no licence from its authors. Get legal review before any
> commercial use or release. Until then, keep GitHub text local, and publish only source ids
> with a rebuild script. The tool marks GitHub cases this way in their `licence` field.

**Wikipedia** text is published under CC BY-SA 4.0. Commercial use is allowed if you give
attribution (the revision link in `source.url`) and release adaptations under the same licence.
Whether a model trained on such text must be shared alike is unsettled. If edit text is ever
published, keep CC BY-SA and the attribution links. The EventStreams documentation states no
separate licence for event metadata.

**Wikimedia etiquette** is built in: an identifying User-Agent with a contact
(`ZILS_CURRENT_CONTACT`), one stream connection, a resume from `Last-Event-ID` after the server's
15-minute disconnect, replay of only the window asked for, serial Action API requests, batches of
50 revision ids, `maxlag=5` with back-off, and gzip. See the
[API etiquette](https://www.mediawiki.org/wiki/API:Etiquette),
[User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy) and
[Robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy).

## How the revert label is chosen

MediaWiki adds the `mw-reverted` change tag to an edit when a later edit reverses it through an
undo, a rollback or a manual revert (an edit that restores an exact earlier version of the page).
A background job applies the tag, so it can lag on busy wikis. It is applied only when the revert
covers at most `$wgRevertedTagMaxDepth` edits (15 by default) and the reverting edit has not itself
been reverted or deleted. On wikis that use edit patrolling, unpatrolled reverts get no tag.
Check a wiki's patrolling before adding it. The tag was chosen over revert-risk model scores
because it records an outcome rather than a prediction. It was chosen over comparing page
histories because MediaWiki already does that comparison.

The label means "reverted by the time we checked" (at least 48 hours after the edit). Reverts
after the check are missed, and so are removals that do not restore an earlier version.

## Limits

- **Pre-set labels.** Most labelled issues carry their type label from the moment they are
  opened, because issue templates apply it. The `label-at-open` filter removes those items, so
  the bug-or-feature stream keeps few items.
- **Skewed outcomes.** Most outside pull requests are merged and most closed issues are closed as
  completed, so balancing keeps a small share of these streams.
- **Pull request text** is read when resolution first sees the pull request. For old archive
  hours that is after the outcome, and the `late-text` filter drops those items. Only a job that
  collects and resolves each hour soon after it is published yields pull request items.
- **Volume.** GitHub's trimmed event feed makes GH Archive hours small (a few megabytes). Issue
  and pull request volume per hour is in the hundreds.
- **Phrase filter.** The outcome phrases are a short, conservative list. Paraphrases slip
  through ("this landed in main"), and an unrelated mention can drop an item ("not merged yet,
  depends on #4").
- **Overlap check.** It catches copies, not paraphrases or translations. Reference rows without
  text are compared on their own content hash, whose scheme may differ from the one used here.
- **Token budget.** The token estimate is conservative and approximate, and it does not replace
  the runtime's own input check.

## Validators

To grade a soup pool on Zils Current items:

1. Run the nightly job on the validator host (or anywhere that writes to a directory the
   validator can read). Pass the validator's exam draws and round bank as references, so the
   overlap check runs against them.
2. Set the pool's `exam` setting to `{"source": "current", "current_bank": "<the --bank
   directory>"}`.

Each round then reads every dated folder whose manifest checks out, and draws its Current part
under the freshness, reuse, breadth and token rules in
[model pools](model-pools.md#round-items-from-zils-current). The round falls back to the fixed
round bank when those rules cannot be met. The four streams cover two cells of the base score:
`reasoning`/`new` and `language`/`new`. The fixed bank still fills the other cells.

Every resolved item is public and can be rebuilt by anyone, labels included. A body committed
after an item resolved could have been trained on it. A short `max_age_days` narrows that
window but does not close it.

## Extension points

- **New sources.** A module with `collect(store, http, ...)` and a
  `resolve(records, http, now)` generator that yields `("pending" | "resolved" | "dropped", record)`,
  plus an entry in `current.RESOLVERS` and `current_items.STREAMS` (with `licence` and
  `publishable`). Stack Exchange closes and tags,
  and review text with ratings, fit this shape.
- **Token-free GitHub resolution.** Merge and close events also appear in later GH Archive hours,
  so a resolver could replay those hours instead of calling the API. Pull request text would
  still need the API.
- **Publishing.** A publishable dated folder (items, manifest and hashes) is the input for an
  upload with a rebuild script. Upload is not implemented. Only publishable streams may ever be
  uploaded.
- **Teacher labels.** Cases accept `target_probabilities`, so dated exports can feed teacher
  distributions without changing the format.
