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

This is a pilot collector. It writes local files only. Publishing, teacher labels and exam
integration are not part of it yet (see [Extension points](#extension-points)).

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
  stopped. Unauthenticated, the GitHub REST API allows 60 requests an hour; with `GITHUB_TOKEN`
  (or `ZILS_GITHUB_TOKEN`) it allows 5,000. One request resolves every stream built from the same
  issue. When GitHub signals a rate limit, the pass stops and the remaining items stay pending.
- `export` writes cases. `--max-tokens 8192` allows longer states for 8,192-token models
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
  opened. Issue templates apply it, and the template's headings can give the answer away.
  `data.labels_at_open` records this, so an exam can exclude such items.
- **Skewed outcomes.** Most outside pull requests are merged and most closed issues are closed as
  completed, so balancing keeps a small share of these streams.
- **Pull request text** is read when resolution first sees the pull request. That can be after
  the outcome when you collect old hours.
- **Volume.** GitHub's trimmed event feed makes GH Archive hours small (a few megabytes). Issue
  and pull request volume per hour is in the hundreds.
- **Token budget.** The token estimate is conservative and approximate, and it does not replace
  the runtime's own input check.

## Extension points

- **New sources.** A module with `collect(store, http, ...)` and a
  `resolve(records, http, now)` generator that yields `("pending" | "resolved" | "dropped", record)`,
  plus an entry in `current.RESOLVERS` and `stream_items.STREAMS`. Stack Exchange closes and tags,
  and review text with ratings, fit this shape.
- **Token-free GitHub resolution.** Merge and close events also appear in later GH Archive hours,
  so a resolver could replay those hours instead of calling the API. Pull request text would
  still need the API.
- **Publishing.** Export output, plus `source` and `content_sha256`, is the input for dated shards
  with a manifest and a rebuild script. Check overlap against existing exams and benchmarks before
  publishing.
- **Teacher labels and grading.** Cases accept `target_probabilities`, and `soup_exam.subset`
  draws family- and label-stratified subsets, so a nightly job can grade fresh items or add
  teacher distributions without changing the format.
