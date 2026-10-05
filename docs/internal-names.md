# Internal names and compatibility

Use `python -m zils...`, `import zils`, the `zils-finetune` skill, and `ZILS_*`
settings for new installations. The implementation lives in `zils/`. The `fez/`
package contains compatibility entry points that forward to the same modules;
existing module commands and direct-file model runners remain supported.

The canonical settings are `ZILS_TRAINING_API_URL`, `ZILS_WEB_ORIGIN`,
`ZILS_TRAINING_MODEL`, `ZILS_JEVK5_BASE_DIR`, `ZILS_GPU_MIN_FREE_MIB`,
`ZILS_NVIDIA_SMI`, `ZILS_COMPUTE_LOCK`, `ZILS_PYTHON`, and `ZILS_PG_BIN`.
Each accepts the corresponding `FEZ_*` fallback. A present canonical Python
setting takes precedence, including an empty value; required settings reject
empty values. The generated shell launcher uses a nonempty `ZILS_PYTHON`, then
`FEZ_PYTHON`, then its local Python environment.

The website uses `NEXT_PUBLIC_ZILS_TRAINING_API_URL` and server-only
`ZILS_DECISION_*` settings, with their previous names as fallbacks. New browser
sessions use `zils-training-auth`; existing sessions and pending sign-in links
keep their old key until sign-out to preserve refresh locking across open tabs.

## Persisted identifiers

This package migration does not rename database tables, RPCs, Storage buckets,
checkpoint metadata, signature domains, recorded model IDs, or experiment files.
The queue still uses `fez_training_*`, `fez_*` RPCs, `fez-training-data`, and
`fez-training-models`. Those are persisted integration identifiers, not product
names. Existing jobs, signed messages, hashes, and download URLs must remain valid.
The default local compute-lock filename is also retained so mixed-version
processes cannot run competing GPU work. Set `ZILS_COMPUTE_LOCK` consistently
across services when choosing a different shared lock path.

Rename persisted resources only through a separately reviewed data migration
with access-control, old-client, and rollback checks. Do not edit historical SQL
migrations or replace strings in saved jobs and signed reports. The old
`fez-finetune` install name remains available for existing users; its Kev workflow
and historical experiment references are retained.
