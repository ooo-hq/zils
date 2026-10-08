-- Operator qualifications are never populated from a miner's self-report.
begin;
alter table public.zils_model_profiles add column profile_sha256 text, add column runtime_sha256 text;
update public.zils_model_profiles set profile_sha256='eb5e0a68b8af47b4a141e046cc955d2607f69b294ae170c8712c1a5a528922fa',runtime_sha256='7c098518c3924d74e26120dcf938a332c926a8d8b64715f968a7fce3022bceda' where id='kev-0.8b-v1';
update public.zils_model_profiles set profile_sha256='ebf0fcd0bce1ccd722cbfde7cfb0a30716fe1d738be10e969e9d2cfcf4ae8b97',runtime_sha256='1d0f4039d61077cc067106b9f20bac8dd28a10509784c6dbf8d17cc23010c08e' where id='jevk5-4b-v0.3';
update public.zils_model_profiles set profile_sha256='3d53e354ad72ae067f33bfa8175d281507c467780a8c6c807cc0eb41eceb459c',runtime_sha256='8b1bec22283ca78f3619d34e4b63d8e337eb6f3a51bbdd59edcb04f2809550b9' where id='imajev-4b-v1';
alter table public.zils_model_profiles alter column profile_sha256 set not null, alter column runtime_sha256 set not null;
create table public.zils_worker_profiles (
  hotkey text not null references public.fez_training_workers(hotkey),
  profile_id text not null references public.zils_model_profiles(id),
  profile_sha256 text not null check(profile_sha256 ~ '^[a-f0-9]{64}$'),
  runtime_sha256 text not null check(runtime_sha256 ~ '^[a-f0-9]{64}$'),
  verified_at timestamptz not null default now(), verified_by text not null check(length(verified_by)>0),
  enabled boolean not null default true, min_free_mib integer not null check(min_free_mib>0),
  evidence jsonb not null,
  primary key(hotkey,profile_id),
  check(profile_id <> 'imajev-4b-v1' or coalesce(
    (evidence->>'examples')::int >= 4 and (evidence->>'max_pixels')::int=400000
    and (evidence->>'max_input_tokens')::int=4096 and (evidence->>'decoder_verified')::boolean
    and (evidence->>'reload_verified')::boolean and (evidence->>'finite_gradients')::boolean
    and (evidence->>'peak_gpu_reserved_bytes')::bigint>0
    and min_free_mib * 1048576::bigint >= (evidence->>'peak_gpu_reserved_bytes')::bigint + 536870912
    and (evidence->>'max_seconds')::int between 1 and 3600
    and evidence->>'probe_sha256' ~ '^[a-f0-9]{64}$'
    and evidence->>'trainer_sha256' ~ '^[a-f0-9]{64}$',false))
);
alter table public.zils_worker_profiles enable row level security;
revoke all on public.zils_worker_profiles from public,anon,authenticated;
grant select,insert,update,delete on public.zils_worker_profiles to service_role;
-- Preserve previously approved text workers; image qualification always requires an operator probe.
insert into public.zils_worker_profiles(hotkey,profile_id,profile_sha256,runtime_sha256,verified_by,min_free_mib,evidence)
select w.hotkey,p.id,p.profile_sha256,p.runtime_sha256,'legacy-text-migration',1,'{"legacy_text":true}'::jsonb
from public.fez_training_workers w cross join public.zils_model_profiles p where p.id <> 'imajev-4b-v1';

create function public.zils_worker_qualified(p_hotkey text,p_profile text)
returns boolean language sql stable set search_path='' as $$
  select exists(select 1 from public.zils_worker_profiles q
    join public.zils_model_profiles p on p.id=q.profile_id
    join public.fez_training_workers w on w.hotkey=q.hotkey
    where q.hotkey=p_hotkey and q.profile_id=p_profile and q.enabled and w.enabled
    and q.profile_sha256=p.profile_sha256 and q.runtime_sha256=p.runtime_sha256
    and q.verified_at<=now());
$$;
create or replace function public.fez_claim_training(p_hotkey text)
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1') <> 'imajev-4b-v1' and a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
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
create function public.zils_claim_profile_training(p_hotkey text,p_supported_profiles text[])
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  if cardinality(p_supported_profiles) not between 1 and 3 or p_supported_profiles is null
    or exists(select 1 from unnest(p_supported_profiles) x where x is null or not public.zils_worker_qualified(p_hotkey,x))
    then raise exception 'installed profiles require operator qualification'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1') = any(p_supported_profiles) and public.zils_worker_qualified(p_hotkey,coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1')) and a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
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
create or replace function public.fez_renew_training(p_job uuid, p_hotkey text, p_token uuid)
returns void language plpgsql set search_path = '' as $$
begin
  if exists(select 1 from public.fez_training_jobs where id=p_job and
      coalesce(model_profile->>'id',manifest->'model'->>'id')='imajev-4b-v1')
    and not public.zils_worker_qualified(p_hotkey,'imajev-4b-v1') then
    raise exception 'worker profile qualification revoked'; end if;
  update public.fez_training_assignments a
    set lease_until=least(now()+interval '20 minutes',j.deadline)
    from public.fez_training_jobs j where j.id=a.job_id and a.job_id=p_job and a.hotkey=p_hotkey
      and a.lease_token=p_token and a.state='leased' and a.lease_until>now()
      and j.status='running' and j.deadline>now();
  if not found then raise exception 'assignment lease expired'; end if;
end $$;
create or replace function public.fez_submit_training(p_job uuid, p_hotkey text, p_token uuid, p_sha256 text)
returns void language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments;
begin
  if exists(select 1 from public.fez_training_jobs where id=p_job and
      coalesce(model_profile->>'id',manifest->'model'->>'id')='imajev-4b-v1')
    and not public.zils_worker_qualified(p_hotkey,'imajev-4b-v1') then
    raise exception 'worker profile qualification revoked'; end if;
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
create or replace function public.fez_approve_training_job(p_job uuid, p_hotkeys text[])
returns void language plpgsql set search_path = '' as $$
begin
  perform 1 from public.fez_training_jobs where id=p_job and status='awaiting_approval' for update;
  if not found then raise exception 'job is not awaiting approval'; end if;
  if cardinality(p_hotkeys) not between 1 and 16 or
     (select count(*) from public.fez_training_workers where enabled and hotkey=any(p_hotkeys))
       <> cardinality(p_hotkeys) then raise exception 'require 1..16 distinct approved workers'; end if;
  if exists(select 1 from public.fez_training_jobs j, unnest(p_hotkeys) h where j.id=p_job
    and j.model_profile->>'id'='imajev-4b-v1' and not public.zils_worker_qualified(h,'imajev-4b-v1'))
    then raise exception 'image training requires qualified workers'; end if;
  insert into public.fez_training_assignments(job_id,hotkey,uid)
    select p_job,hotkey,uid from public.fez_training_workers where hotkey=any(p_hotkeys);
  update public.fez_training_jobs set status='queued',deadline=now()+interval '24 hours',updated_at=now()
    where id=p_job;
end $$;
revoke execute on function public.zils_worker_qualified(text,text),public.zils_claim_profile_training(text,text[]) from public,anon,authenticated;
grant execute on function public.zils_worker_qualified(text,text),public.zils_claim_profile_training(text,text[]) to service_role;
commit;
