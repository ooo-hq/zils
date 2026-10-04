# Zils standalone benchmark preview

A read-only, single-page observatory. Native HTML, CSS, and JavaScript; no npm
dependencies, GPU, wallet, model download, or validator required.

The selected public location is `https://zils.ai/model`, implemented in the
existing [Zils website app](https://github.com/ooo-hq/zils-web)
with its shared navigation and visual identity. This folder retains the
standalone benchmark preview. The domain route must be published through that
website's normal deployment; this preview does not deploy it.

## Local preview

Install Node.js 22+ (with npm) and Python 3. From a fresh repository checkout:

```bash
npm --prefix website run preview
```

Open <http://127.0.0.1:4173>. This builds the static site and serves only
`website/dist/` on loopback. Stop with Ctrl+C. After editing, stop and restart
the command to rebuild, then refresh the browser. No `npm install` is needed.

## Checks and build

```bash
npm --prefix website test
npm --prefix website run build
```

The check covers real benchmark totals, fraction-to-percent conversion,
seconds-to-milliseconds conversion, comparison wording, unavailable values, and
invalid records. Browser verification should include 1440px and 390px widths,
keyboard navigation, expanding evidence details, source links, and a failed JSON
request followed by successful retry. There is also a no-JavaScript source link.

## Data and publication

`../docs/data/jevbench-public-001.json` is the sole data input. Edit that committed
source only when updating this recorded experiment, then rebuild. Loading and
validation failures show an error with retry; optional missing metrics display
“Unavailable.” The page records evidence timestamps in UTC, never a fabricated
live refresh time. The testnet section links to the recorded subnet 579 evidence. It does not
report live participant availability.

The build copies five explicitly named frontend files and that JSON. It does not
copy repository folders, raw runs, model artifacts, private configuration, or
participant endpoints. `dist/` is generated and ignored. Publish only a clean
build of `website/dist/`; never serve the repository root publicly.

The `/model` integration is deployed through the `ooo-hq/zils-web` GitHub repository.
Do not deploy this standalone output over the main homepage or the separate
`docs.fez.chat` manual. The website app renders public benchmark and testnet snapshots at build time
without a live feed. See the [benchmark methodology](../docs/jevbench-public.md)
and [verified testnet round](../docs/testnet-round-001.md) for scope and limitations.
