-- Independent, opt-in evaluation. This never changes training acceptance or model aliases.
create table public.zils_training_comparisons (
  job_id uuid primary key references public.fez_training_jobs(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'uploading' check (status in ('uploading','queued','running','completed','failed')),
  consent_version text not null check (consent_version = 'typesafe-evaluation-v1'),
  consented_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  finished_at timestamptz,
  model_id text not null,
  checkpoint_sha256 text not null check (checkpoint_sha256 ~ '^[a-f0-9]{64}$'),
  jev_model text not null check (jev_model = 'jev-1.13.0'),
  lease_token uuid,
  lease_until timestamptz,
  input_sha256 text,
  result jsonb,
  error text
);
alter table public.zils_training_comparisons enable row level security;
revoke all on public.zils_training_comparisons from public, anon, authenticated;
grant all on public.zils_training_comparisons to service_role;
create index on public.zils_training_comparisons(created_at) where status in ('queued','running');
