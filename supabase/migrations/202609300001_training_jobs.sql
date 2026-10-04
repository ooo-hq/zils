-- Isolated resources for the Fez training service. Apply with a privileged migration role.
begin;

create table public.fez_training_jobs (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id),
  name text not null check (name ~ '^[a-z0-9][a-z0-9-]{0,63}$'),
  status text not null default 'uploading' check (status in
    ('uploading','validating','awaiting_approval','queued','running','evaluating','completed','failed')),
  acceptance jsonb not null,
  training_export_authorized_at timestamptz not null default now(),
  manifest jsonb,
  job_sha256 text,
  initial_sha256 text,
  lease_token uuid,
  lease_until timestamptz,
  deadline timestamptz,
  result jsonb,
  release_prefix text,
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create table public.fez_training_workers (
  hotkey text primary key,
  uid integer not null unique check (uid between 0 and 65535),
  enabled boolean not null default true
);
create table public.fez_training_assignments (
  job_id uuid not null references public.fez_training_jobs(id) on delete cascade,
  hotkey text not null references public.fez_training_workers(hotkey),
  uid integer not null check (uid between 0 and 65535),
  state text not null default 'ready' check (state in ('ready','leased','submitted','failed')),
  lease_token uuid,
  lease_until timestamptz,
  attempts integer not null default 0,
  sha256 text check (sha256 ~ '^[a-f0-9]{64}$'),
  primary key (job_id, hotkey),
  unique (job_id, uid),
  unique (job_id, sha256),
  check (state <> 'submitted' or sha256 is not null)
);
create table public.fez_training_nonces (
  hotkey text not null references public.fez_training_workers(hotkey),
  nonce uuid not null,
  created_at timestamptz not null default now(),
  primary key (hotkey, nonce)
);
create index on public.fez_training_jobs(status, created_at);
create index on public.fez_training_assignments(hotkey, state);
create index on public.fez_training_nonces(created_at);

alter table public.fez_training_jobs enable row level security;
alter table public.fez_training_workers enable row level security;
alter table public.fez_training_assignments enable row level security;
alter table public.fez_training_nonces enable row level security;
revoke all on public.fez_training_jobs, public.fez_training_workers,
  public.fez_training_assignments, public.fez_training_nonces from public, anon, authenticated;
grant select on public.fez_training_jobs to authenticated;
create policy fez_training_owner_read on public.fez_training_jobs for select to authenticated
  using ((select auth.uid()) = owner_id);
grant all on public.fez_training_jobs, public.fez_training_workers,
  public.fez_training_assignments, public.fez_training_nonces to service_role;

insert into storage.buckets (id, name, public, file_size_limit)
values ('fez-training-data', 'fez-training-data', false, 134217728),
       ('fez-training-models', 'fez-training-models', false, 536870912);
-- No client storage policies: access is issued by the coordinator using signed URLs.
-- Existing projects may have permissive storage policies. This restrictive
-- policy prevents those policies from accidentally exposing training objects.
create policy fez_training_private_objects on storage.objects as restrictive
  for all to anon, authenticated
  using (bucket_id not in ('fez-training-data','fez-training-models'))
  with check (bucket_id not in ('fez-training-data','fez-training-models'));

create function public.fez_create_training_job(p_owner uuid, p_name text, p_acceptance jsonb)
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text, 0));
  if (select count(*) from public.fez_training_jobs where owner_id = p_owner
      and status not in ('completed','failed')) >= 5 then
    raise exception 'maximum of five active jobs per account';
  end if;
  insert into public.fez_training_jobs(owner_id,name,acceptance)
    values (p_owner,p_name,p_acceptance) returning * into item;
  return item;
end $$;

create function public.fez_approve_training_job(p_job uuid, p_hotkeys text[])
returns void language plpgsql set search_path = '' as $$
begin
  perform 1 from public.fez_training_jobs where id=p_job and status='awaiting_approval' for update;
  if not found then raise exception 'job is not awaiting approval'; end if;
  if cardinality(p_hotkeys) not between 1 and 16 or
     (select count(*) from public.fez_training_workers where enabled and hotkey=any(p_hotkeys))
       <> cardinality(p_hotkeys) then raise exception 'require 1..16 distinct approved workers'; end if;
  insert into public.fez_training_assignments(job_id,hotkey,uid)
    select p_job,hotkey,uid from public.fez_training_workers where hotkey=any(p_hotkeys);
  update public.fez_training_jobs set status='queued',deadline=now()+interval '24 hours',updated_at=now()
    where id=p_job;
end $$;

create function public.fez_worker_nonce(p_hotkey text, p_nonce uuid)
returns void language plpgsql set search_path = '' as $$
begin
  if not exists(select 1 from public.fez_training_workers where hotkey=p_hotkey and enabled)
    then raise exception 'worker is not enabled'; end if;
  delete from public.fez_training_nonces where created_at < now()-interval '10 minutes';
  insert into public.fez_training_nonces(hotkey,nonce) values (p_hotkey,p_nonce);
end $$;

create function public.fez_claim_training(p_hotkey text)
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
      and (a.state='ready' or (a.state='leased' and (a.lease_until > now() or a.attempts < 3)))
    order by (a.state='leased' and a.lease_until > now()) desc, j.created_at
    limit 1 for update of a skip locked;
  if not found then return null; end if;
  if item.state <> 'leased' or item.lease_until <= now() then
    update public.fez_training_assignments set state='leased',lease_token=gen_random_uuid(),
      lease_until=least(now()+interval '20 minutes', (select deadline from public.fez_training_jobs where id=item.job_id)),
      attempts=attempts+1 where job_id=item.job_id and hotkey=p_hotkey returning * into item;
  end if;
  update public.fez_training_jobs set status='running',updated_at=now() where id=item.job_id;
  return to_jsonb(item);
end $$;

create function public.fez_renew_training(p_job uuid, p_hotkey text, p_token uuid)
returns void language plpgsql set search_path = '' as $$
begin
  update public.fez_training_assignments a
    set lease_until=least(now()+interval '20 minutes',j.deadline)
    from public.fez_training_jobs j where j.id=a.job_id and a.job_id=p_job and a.hotkey=p_hotkey
      and a.lease_token=p_token and a.state='leased' and a.lease_until>now()
      and j.status='running' and j.deadline>now();
  if not found then raise exception 'assignment lease expired'; end if;
end $$;

create function public.fez_submit_training(p_job uuid, p_hotkey text, p_token uuid, p_sha256 text)
returns void language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments;
begin
  -- Serialize with this worker's claims; lock the job before accepting a new
  -- checkpoint so a simultaneous evaluator cannot omit a committed submission.
  perform 1 from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  select * into item from public.fez_training_assignments
    where job_id=p_job and hotkey=p_hotkey for update;
  if item.state='submitted' and item.lease_token=p_token and item.sha256=p_sha256 then return; end if;
  if item.state is distinct from 'leased' or item.lease_token is distinct from p_token
    or item.lease_until<=now() or not exists(select 1 from public.fez_training_jobs
      where id=p_job and status='running' and deadline>now() for update) then
    raise exception 'assignment lease expired';
  end if;
  if exists(select 1 from public.fez_training_assignments where job_id=p_job and sha256=p_sha256)
    then raise exception 'duplicate checkpoint'; end if;
  update public.fez_training_assignments set state='submitted',sha256=p_sha256
    where job_id=p_job and hotkey=p_hotkey;
end $$;

create function public.fez_claim_processing(p_stage text)
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
  if p_stage not in ('validating','evaluating') then raise exception 'invalid stage'; end if;
  select * into item from public.fez_training_jobs j where
    (j.status=p_stage and (j.lease_until is null or j.lease_until<=now())) or
    (p_stage='evaluating' and j.status in ('queued','running') and
      (j.deadline<=now() or not exists(select 1 from public.fez_training_assignments a
        where a.job_id=j.id and a.state not in ('submitted','failed'))))
    order by created_at limit 1 for update skip locked;
  if not found then return null; end if;
  update public.fez_training_jobs set status=p_stage,lease_token=gen_random_uuid(),
    lease_until=now()+interval '20 minutes',updated_at=now() where id=item.id returning * into item;
  return item;
end $$;

create function public.fez_finish_processing(p_job uuid,p_token uuid,p_status text,p_values jsonb)
returns void language plpgsql set search_path = '' as $$
begin
  if p_status not in ('awaiting_approval','completed','failed') then raise exception 'invalid status'; end if;
  update public.fez_training_jobs set status=p_status,lease_token=null,lease_until=null,updated_at=now(),
    manifest=coalesce(p_values->'manifest',manifest),
    job_sha256=coalesce(p_values->>'job_sha256',job_sha256),
    initial_sha256=coalesce(p_values->>'initial_sha256',initial_sha256),
    result=p_values->'result',release_prefix=p_values->>'release_prefix',error=p_values->>'error'
    where id=p_job and lease_token=p_token and lease_until>now()
      and ((status='validating' and p_status in ('awaiting_approval','failed'))
        or (status='evaluating' and p_status in ('completed','failed')));
  if not found then raise exception 'processing lease expired'; end if;
end $$;

create function public.fez_fail_training(p_job uuid,p_hotkey text,p_token uuid)
returns void language plpgsql set search_path = '' as $$
begin
  update public.fez_training_assignments set state=case when attempts>=3 then 'failed' else 'ready' end,
    lease_token=null,lease_until=null where job_id=p_job and hotkey=p_hotkey
    and lease_token=p_token and state='leased';
end $$;

-- PostgREST functions are callable only by the coordinator's server credential.
revoke execute on function public.fez_create_training_job(uuid,text,jsonb),
  public.fez_approve_training_job(uuid,text[]), public.fez_worker_nonce(text,uuid),
  public.fez_claim_training(text), public.fez_renew_training(uuid,text,uuid),
  public.fez_submit_training(uuid,text,uuid,text), public.fez_claim_processing(text),
  public.fez_finish_processing(uuid,uuid,text,jsonb), public.fez_fail_training(uuid,text,uuid) from public,anon,authenticated;
grant execute on function public.fez_create_training_job(uuid,text,jsonb),
  public.fez_approve_training_job(uuid,text[]), public.fez_worker_nonce(text,uuid),
  public.fez_claim_training(text), public.fez_renew_training(uuid,text,uuid),
  public.fez_submit_training(uuid,text,uuid,text), public.fez_claim_processing(text),
  public.fez_finish_processing(uuid,uuid,text,jsonb), public.fez_fail_training(uuid,text,uuid) to service_role;
commit;
