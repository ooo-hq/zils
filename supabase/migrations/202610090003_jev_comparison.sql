-- Supplemental benchmark results; never mutate the accepted release's provenance.
begin;
alter table public.fez_training_jobs add column jev_comparison jsonb;
comment on column public.fez_training_jobs.jev_comparison is
  'Server-written TypeSafe Jev benchmark status and aggregate metrics. Existing owner-only RLS applies.';
commit;
