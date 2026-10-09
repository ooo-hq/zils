-- Opt-in grading. All evidence and routing configuration are operator-owned.
begin;
alter table public.fez_training_workers add column resource_id uuid;
create table public.zils_routing_policy (
  id boolean primary key default true check(id), config jsonb not null
);
create table public.zils_worker_presence (
  hotkey text primary key references public.fez_training_workers,
  received_at timestamptz not null, profiles jsonb not null
);
create table public.zils_resource_cooldowns (
  resource_id uuid primary key, until_at timestamptz not null
);
create table public.zils_worker_qualifications (
  id uuid primary key default gen_random_uuid(), hotkey text not null references public.fez_training_workers,
  resource_id uuid not null, report jsonb not null, verified_by text not null check(length(verified_by)>0),
  evidence_sha256 text not null unique check(evidence_sha256 ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now()
);
create table public.zils_qualification_benchmarks (
  context jsonb primary key, descriptor jsonb not null
);
create table public.zils_job_scheduling (
  job_id uuid primary key references public.fez_training_jobs,
  deadline timestamptz not null, purpose text not null default 'customer' check(purpose in ('customer','qualification')),
  context jsonb, benchmark jsonb, graded boolean not null default false,
  benchmark_sha256 text check(benchmark_sha256 ~ '^[a-f0-9]{64}$')
);
create table public.zils_training_attempts (
  id uuid primary key default gen_random_uuid(), job_id uuid not null references public.fez_training_jobs,
  hotkey text not null references public.fez_training_workers, resource_id uuid not null,
  lease_token uuid not null, context jsonb not null, total_tokens bigint not null check(total_tokens>0),
  started_at timestamptz not null default now(), submitted_at timestamptz, finished_at timestamptz,
  sha256 text, outcome text check(outcome in ('valid','invalid_artifact','worker_failure','abandoned',
    'capacity_deferred','validator_error','infrastructure_error','cancelled','pending_review')),
  observation jsonb, unique(job_id,hotkey,lease_token)
);
create table public.zils_worker_reservations (
  resource_id uuid primary key, job_id uuid not null references public.fez_training_jobs,
  hotkey text not null references public.fez_training_workers, expires_at timestamptz not null,
  lease_token uuid, graded boolean not null, created_at timestamptz not null default now(),
  unique(job_id,hotkey)
);
-- A fixed competition can use multiple independent hosts; graded jobs use exactly one.
create unique index zils_one_graded_assignment on public.zils_worker_reservations(job_id) where graded;
create table public.zils_assignment_decisions (
  id uuid primary key default gen_random_uuid(), job_id uuid not null references public.fez_training_jobs,
  decision jsonb not null, input_snapshot jsonb not null, created_at timestamptz not null default now()
);
create index on public.zils_training_attempts(hotkey,finished_at desc);
create index on public.zils_worker_qualifications(hotkey,created_at desc);

-- A small, approved pilot pool uses one transaction mutex. Statement triggers take
-- it BEFORE row locks, including REST cancellation, so mixed legacy/graded paths
-- cannot invert job/worker lock order. No network calls occur inside transactions.
create function public.zils_queue_mutex() returns trigger language plpgsql set search_path='' as $$
begin perform pg_advisory_xact_lock(9135172401); return null; end $$;
do $$ declare t text; begin
  foreach t in array array['fez_training_jobs','fez_training_assignments','fez_training_workers',
    'zils_worker_profiles','zils_routing_policy','zils_worker_presence','zils_worker_qualifications',
    'zils_job_scheduling','zils_training_attempts','zils_worker_reservations','zils_resource_cooldowns','zils_qualification_benchmarks'] loop
    execute format('create trigger zils_queue_mutex before insert or update or delete on public.%I for each statement execute function public.zils_queue_mutex()',t);
  end loop;
  foreach t in array array['zils_routing_policy','zils_worker_presence','zils_resource_cooldowns',
    'zils_worker_qualifications','zils_job_scheduling','zils_training_attempts','zils_worker_reservations','zils_assignment_decisions','zils_qualification_benchmarks'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('revoke all on public.%I from public,anon,authenticated',t);
    execute format('grant select,insert,update,delete on public.%I to service_role',t);
  end loop;
end $$;
create function public.zils_immutable_qualification() returns trigger language plpgsql set search_path='' as $$
begin raise exception 'qualification evidence is append-only'; end $$;
create trigger zils_immutable_qualification before update or delete on public.zils_worker_qualifications
for each row execute function public.zils_immutable_qualification();

create function public.zils_schedule_job_state() returns trigger language plpgsql set search_path='' as $$
begin
  if new.status='awaiting_approval' and new.manifest->'workload'->>'model'='jevk5-4b-v0.3' then
    insert into public.zils_job_scheduling(job_id,deadline) values(new.id,coalesce(new.deadline,now()+interval '24 hours')) on conflict do nothing;
  end if;
  if new.status in ('completed','failed') then
    update public.zils_training_attempts set finished_at=now(),outcome=case when new.error='Cancelled by customer.' then 'cancelled' else 'pending_review' end
      where job_id=new.id and finished_at is null;
    delete from public.zils_worker_reservations where job_id=new.id;
  end if;
  return new;
end $$;
create trigger zils_schedule_job_state after insert or update on public.fez_training_jobs
for each row execute function public.zils_schedule_job_state();

create function public.zils_worker_heartbeat(p_hotkey text,p_profiles jsonb) returns timestamptz
language plpgsql set search_path='' as $$
declare p jsonb;
begin
  perform pg_advisory_xact_lock(9135172401);
  if jsonb_typeof(p_profiles) is distinct from 'array' or jsonb_array_length(p_profiles) not between 1 and 3 then raise exception 'invalid presence'; end if;
  for p in select * from jsonb_array_elements(p_profiles) loop
    if (p-array['model','profile_sha256','runtime_sha256','trainer_sha256','ready']) <> '{}'::jsonb
      or jsonb_typeof(p->'ready') is distinct from 'boolean'
      or coalesce(p->>'trainer_sha256','') !~ '^[a-f0-9]{64}$'
      or not exists(select 1 from public.zils_worker_profiles q where q.hotkey=p_hotkey and q.profile_id=p->>'model'
        and q.profile_sha256=p->>'profile_sha256' and q.runtime_sha256=p->>'runtime_sha256'
        and public.zils_worker_qualified(p_hotkey,q.profile_id)) then raise exception 'unqualified presence'; end if;
  end loop;
  if (select count(distinct elem->>'model') from jsonb_array_elements(p_profiles) elem) <> jsonb_array_length(p_profiles) then raise exception 'duplicate profile'; end if;
  insert into public.zils_worker_presence values(p_hotkey,now(),p_profiles)
    on conflict(hotkey) do update set received_at=excluded.received_at,profiles=excluded.profiles;
  return now();
end $$;

create function public.zils_import_qualification(p_hotkey text,p_report jsonb,p_verified_by text,p_evidence_sha256 text) returns uuid
language plpgsql set search_path='' as $$
declare rid uuid; eid uuid:=gen_random_uuid(); c jsonb:=p_report->'context';
begin
  perform pg_advisory_xact_lock(9135172401);
  select resource_id into rid from public.fez_training_workers where hotkey=p_hotkey and enabled;
  if rid is null or c->>'model' is distinct from 'jevk5-4b-v0.3'
    or not public.zils_worker_qualified(p_hotkey,c->>'model')
    or coalesce(p_report->>'artifact_valid','false') <> 'true'
    or not ((p_report->>'verified_at')::timestamptz<=now() and (p_report->>'expires_at')::timestamptz>now())
    or not ((p_report->>'seconds_per_token')::numeric>0 and (p_report->>'seconds_per_token')::numeric<3600)
    or not ((p_report->'capacity'->>'examples')::bigint>0 and (p_report->'capacity'->>'total_tokens')::bigint>0
      and (p_report->'capacity'->>'max_tokens')::int between 1 and 2048)
    then raise exception 'invalid verified report'; end if;
  insert into public.zils_worker_qualifications(id,hotkey,resource_id,report,verified_by,evidence_sha256)
    values(eid,p_hotkey,rid,p_report||jsonb_build_object('id',eid),p_verified_by,p_evidence_sha256);
  return eid;
end $$;

-- Read-only snapshot; its digest is checked under the mutex before reservation.
create function public.zils_grading_snapshot(p_job uuid) returns jsonb language plpgsql stable set search_path='' as $$
declare j public.fez_training_jobs; s public.zils_job_scheduling; cfg jsonb; ctx jsonb; workers jsonb; state jsonb;
begin
  select * into j from public.fez_training_jobs where id=p_job;
  select * into s from public.zils_job_scheduling where job_id=p_job;
  select config into cfg from public.zils_routing_policy where id;
  ctx:=cfg->'contexts'->(j.manifest->'workload'->>'band');
  select coalesce(jsonb_agg(x order by x->>'hotkey'),'[]') into workers from (
    select jsonb_build_object('hotkey',w.hotkey,'enabled',w.enabled,'approved',true,'resource_id',w.resource_id,
      'profile',case when public.zils_worker_qualified(w.hotkey,'jevk5-4b-v0.3') and p->>'trainer_sha256'=ctx->>'trainer_sha256'
        then jsonb_build_object('profile_sha256',p->>'profile_sha256','runtime_sha256',p->>'runtime_sha256') else '{}'::jsonb end,
      'received_at',h.received_at,'ready',p->'ready','cooldown_until',cd.until_at,
      'reserved',exists(select 1 from public.zils_worker_reservations r where r.resource_id=w.resource_id),
      'last_assigned_at',(select max(created_at) from public.zils_assignment_decisions d where d.decision->>'selected_hotkey'=w.hotkey),
      'last_qualification_at',(select max(d.created_at) from public.zils_assignment_decisions d where d.decision->>'selected_hotkey'=w.hotkey and d.decision->>'purpose'='qualification'),
      'reports',(select coalesce(jsonb_agg(q.report order by q.id),'[]') from public.zils_worker_qualifications q where q.hotkey=w.hotkey and q.resource_id=w.resource_id and q.report->'context'=ctx),
      'attempts',(select coalesce(jsonb_agg(to_jsonb(a) order by a.id),'[]') from
        (select * from public.zils_training_attempts a where a.hotkey=w.hotkey and a.resource_id=w.resource_id and a.context->>'band'=ctx->>'band'
         and a.finished_at>now()-interval '30 days' order by a.finished_at desc limit 50) a)) x
    from public.fez_training_workers w left join public.zils_worker_presence h using(hotkey)
    left join lateral (select value p from jsonb_array_elements(h.profiles) where value->>'model'='jevk5-4b-v0.3') installed on true
    left join public.zils_resource_cooldowns cd on cd.resource_id=w.resource_id
    where cfg->'pool' ? w.hotkey
  ) members;
  state:=jsonb_build_object('config',cfg,'job',jsonb_build_object('id',j.id,'job_sha256',j.job_sha256,'status',j.status,
    'context',ctx,'workload',j.manifest->'workload','deadline',coalesce(j.deadline,s.deadline),
    'consent',j.training_export_authorized_at is not null and j.manifest->>'data_access'='approved-workers-training-export',
    'purpose',coalesce(s.purpose,'customer'),'benchmark_sha256',s.benchmark_sha256),'workers',workers);
  return state||jsonb_build_object('state_sha256',encode(sha256(convert_to(state::text,'UTF8')),'hex'));
end $$;

create function public.zils_release_graded_attempt(p_job uuid,p_hotkey text,p_token uuid,p_outcome text) returns jsonb
language plpgsql set search_path='' as $$
declare r public.zils_worker_reservations; n integer;
begin
  perform pg_advisory_xact_lock(9135172401);
  select * into r from public.zils_worker_reservations where job_id=p_job and hotkey=p_hotkey and graded;
  if not found or r.lease_token is distinct from p_token then return jsonb_build_object('status','stale'); end if;
  if p_outcome not in ('capacity_deferred','pending_review','abandoned','invalid_artifact') then raise exception 'invalid release outcome'; end if;
  update public.zils_training_attempts set finished_at=now(),outcome=p_outcome
    where job_id=p_job and hotkey=p_hotkey and lease_token=p_token and finished_at is null;
  update public.fez_training_assignments set state='failed',lease_token=null,lease_until=null,
    attempts=greatest(0,attempts-case when p_outcome='capacity_deferred' then 1 else 0 end)
    where job_id=p_job and hotkey=p_hotkey and state in ('ready','leased');
  if p_outcome='capacity_deferred' then
    insert into public.zils_resource_cooldowns values(r.resource_id,now()+interval '60 seconds')
      on conflict(resource_id) do update set until_at=excluded.until_at;
  end if;
  delete from public.zils_worker_reservations where resource_id=r.resource_id;
  select count(*) into n from public.zils_training_attempts where job_id=p_job and outcome in ('abandoned','worker_failure','invalid_artifact','pending_review');
  update public.fez_training_jobs set status=case when n>=3 or deadline<=now() then 'queued' else 'awaiting_approval' end,updated_at=now()
    where id=p_job and status in ('queued','running');
  return jsonb_build_object('status','released');
end $$;

create function public.zils_reap_graded() returns void language plpgsql set search_path='' as $$
declare r public.zils_worker_reservations;
begin
  perform pg_advisory_xact_lock(9135172401);
  for r in select * from public.zils_worker_reservations where graded and expires_at<=now() loop
    perform public.zils_release_graded_attempt(r.job_id,r.hotkey,r.lease_token,'abandoned');
  end loop;
  update public.fez_training_jobs j set status='failed',error='Training scheduling deadline expired.',updated_at=now()
    from public.zils_job_scheduling s, public.zils_routing_policy p where j.id=s.job_id and p.id and p.config->>'mode'='graded'
      and j.status='awaiting_approval' and coalesce(j.deadline,s.deadline)<=now()
      and p.config->'contexts' ? (j.manifest->'workload'->>'band');
end $$;

create function public.zils_reserve_graded_job(p_job uuid,p_hotkey text,p_decision jsonb) returns jsonb
language plpgsql set search_path='' as $$
declare snap jsonb; j public.fez_training_jobs; s public.zils_job_scheduling; w public.fez_training_workers;
  candidate jsonb; evidence_count integer; ctx jsonb;
begin
  perform pg_advisory_xact_lock(9135172401);
  perform public.zils_reap_graded();
  snap:=public.zils_grading_snapshot(p_job);
  select * into j from public.fez_training_jobs where id=p_job for update;
  select * into s from public.zils_job_scheduling where job_id=p_job;
  select * into w from public.fez_training_workers where hotkey=p_hotkey and enabled;
  select c into candidate from jsonb_array_elements(p_decision->'candidates') c where c->>'hotkey'=p_hotkey;
  ctx:=snap->'job'->'context';
  if p_hotkey is null or w.resource_id is null or j.status is distinct from 'awaiting_approval'
    or snap->>'state_sha256' is distinct from p_decision->>'state_sha256'
    or snap->'config'->>'mode' is distinct from 'graded'
    or p_decision->>'policy_version' is distinct from 'zils-miner-routing/v1'
    or p_decision->>'selected_hotkey' is distinct from p_hotkey
    or p_decision->>'job_sha256' is distinct from j.job_sha256
    or p_decision->'context' is distinct from ctx or candidate is null
    or not coalesce((snap->'job'->>'consent')::boolean,false)
    or not coalesce(snap->'config'->'pool' ? p_hotkey,false)
    or exists(select 1 from public.zils_worker_reservations where resource_id=w.resource_id or job_id=p_job)
    or exists(select 1 from public.zils_resource_cooldowns where resource_id=w.resource_id and until_at>now())
    or not public.zils_worker_qualified(p_hotkey,'jevk5-4b-v0.3')
    or not exists(select 1 from public.zils_worker_presence h, jsonb_array_elements(h.profiles) p where h.hotkey=p_hotkey
      and h.received_at>now()-interval '45 seconds' and p->>'model'=ctx->>'model' and p->>'profile_sha256'=ctx->>'profile_sha256'
      and p->>'runtime_sha256'=ctx->>'runtime_sha256' and p->>'trainer_sha256'=ctx->>'trainer_sha256' and p->'ready'='true'::jsonb)
    or not coalesce((candidate->>'estimated_seconds')::numeric between 0 and 3600,false)
    or coalesce(j.deadline,s.deadline)<=now()+make_interval(secs=>(candidate->>'estimated_seconds')::double precision+60)
    then return jsonb_build_object('status','retry'); end if;
  select count(*) into evidence_count from public.zils_worker_qualifications q where q.hotkey=p_hotkey and q.resource_id=w.resource_id
    and candidate->'evidence_ids' ? q.id::text and q.report->'context'=ctx
    and (q.report->>'expires_at')::timestamptz>now() and (q.report->>'verified_at')::timestamptz>now()-interval '30 days';
  if evidence_count <> jsonb_array_length(candidate->'evidence_ids') or evidence_count<(case when s.purpose='qualification' then 1 else 3 end)
    or (s.purpose='qualification' and (s.benchmark_sha256 is distinct from ctx->>'benchmark_sha256'
      or (snap->'config'->>'qualification_slots')::int<>1
      or exists(select 1 from public.zils_worker_reservations r join public.zils_job_scheduling x using(job_id) where x.purpose='qualification')))
    then return jsonb_build_object('status','retry'); end if;
  insert into public.zils_worker_reservations(resource_id,job_id,hotkey,expires_at,graded)
    values(w.resource_id,p_job,p_hotkey,least(coalesce(j.deadline,s.deadline),now()+interval '60 seconds'),true);
  update public.zils_job_scheduling set graded=true,context=ctx where job_id=p_job;
  insert into public.fez_training_assignments(job_id,hotkey,uid) values(p_job,p_hotkey,w.uid)
    on conflict(job_id,hotkey) do update set state='ready',lease_token=null,lease_until=null,sha256=null;
  update public.fez_training_jobs set status='queued',deadline=coalesce(j.deadline,s.deadline),updated_at=now() where id=p_job;
  insert into public.zils_assignment_decisions(job_id,decision,input_snapshot) values(p_job,p_decision,snap);
  return jsonb_build_object('status','reserved','hotkey',p_hotkey);
end $$;

create function public.zils_guard_resource_mapping() returns trigger language plpgsql set search_path='' as $$
begin
  if old.resource_id is distinct from new.resource_id and exists(
    select 1 from public.fez_training_assignments a join public.fez_training_jobs j on j.id=a.job_id
    where a.hotkey=new.hotkey and a.state in ('ready','leased') and j.status in ('queued','running')) then
    raise exception 'cannot remap a busy worker'; end if;
  return new;
end $$;
create trigger zils_guard_resource_mapping before update on public.fez_training_workers
for each row execute function public.zils_guard_resource_mapping();

create function public.zils_assignment_observation() returns trigger language plpgsql set search_path='' as $$
declare r public.zils_worker_reservations; s public.zils_job_scheduling;
begin
  select * into r from public.zils_worker_reservations where job_id=new.job_id and hotkey=new.hotkey;
  if not found then return new; end if;
  if new.state='leased' then
    -- Fixed assignments own the slot through their job deadline, including retries.
    update public.zils_worker_reservations set lease_token=new.lease_token,
      expires_at=case when r.graded then new.lease_until else r.expires_at end where resource_id=r.resource_id;
    if r.graded then
      select * into s from public.zils_job_scheduling where job_id=new.job_id;
      insert into public.zils_training_attempts(job_id,hotkey,resource_id,lease_token,context,total_tokens)
        select new.job_id,new.hotkey,r.resource_id,new.lease_token,s.context,(j.manifest->'workload'->>'total_tokens')::bigint
        from public.fez_training_jobs j where j.id=new.job_id on conflict do nothing;
    end if;
  elsif new.state='submitted' then
    update public.zils_training_attempts set submitted_at=now(),sha256=new.sha256
      where job_id=new.job_id and hotkey=new.hotkey and lease_token=new.lease_token and submitted_at is null;
    delete from public.zils_worker_reservations where resource_id=r.resource_id;
  end if;
  return new;
end $$;
create trigger zils_assignment_observation after insert or update on public.fez_training_assignments
for each row execute function public.zils_assignment_observation();

create function public.zils_resource_claimable(p_job uuid,p_hotkey text) returns boolean language sql stable set search_path='' as $$
  select w.resource_id is null or exists(select 1 from public.zils_worker_reservations r
    where r.resource_id=w.resource_id and r.job_id=p_job and r.hotkey=p_hotkey and r.expires_at>now()
      and (not r.graded or (public.zils_worker_qualified(p_hotkey,'jevk5-4b-v0.3') and exists(
        select 1 from public.zils_job_scheduling s,public.zils_routing_policy c where s.job_id=p_job and c.id
          and c.config->>'mode'='graded' and c.config->'pool' ? p_hotkey
          and c.config->'contexts'->(s.context->>'band')=s.context))))
  from public.fez_training_workers w where w.hotkey=p_hotkey;
$$;

create function public.zils_configure_routing(p_config jsonb) returns jsonb language plpgsql set search_path='' as $$
begin
  perform pg_advisory_xact_lock(9135172401);
  if p_config->>'mode'='graded' then
    if p_config->>'policy_version' is distinct from 'zils-miner-routing/v1'
      or jsonb_typeof(p_config->'pool') is distinct from 'array'
      or jsonb_array_length(p_config->'pool') not between 1 and 256
      or jsonb_typeof(p_config->'contexts') is distinct from 'object'
      or p_config->'contexts'='{}'::jsonb
      or not coalesce((p_config->>'qualification_slots')::int in (0,1),false)
      or exists(select 1 from jsonb_array_elements_text(p_config->'pool') k where not exists(select 1 from public.fez_training_workers w where w.hotkey=k))
      then raise exception 'invalid approved routing policy'; end if;
  elsif p_config is distinct from '{"mode":"fixed"}'::jsonb then raise exception 'invalid routing mode'; end if;
  insert into public.zils_routing_policy values(true,p_config) on conflict(id) do update set config=excluded.config;
  return p_config;
end $$;
create function public.zils_bind_resource(p_hotkey text,p_resource uuid) returns void language plpgsql set search_path='' as $$
begin
  perform pg_advisory_xact_lock(9135172401);
  if p_resource is null then raise exception 'require physical resource UUID'; end if;
  update public.fez_training_workers set resource_id=p_resource where hotkey=p_hotkey;
  if not found then raise exception 'unknown worker'; end if;
end $$;
create function public.zils_authorize_qualification(p_job uuid,p_benchmark jsonb) returns void language plpgsql set search_path='' as $$
declare j public.fez_training_jobs; ctx jsonb:=p_benchmark->'context'; cfg jsonb;
begin
  perform pg_advisory_xact_lock(9135172401);
  select * into j from public.fez_training_jobs where id=p_job and status='awaiting_approval' for update;
  select config into cfg from public.zils_routing_policy where id;
  if j.id is null or ctx is distinct from cfg->'contexts'->(j.manifest->'workload'->>'band')
    or p_benchmark->>'benchmark_sha256' is distinct from ctx->>'benchmark_sha256'
    or not coalesce((p_benchmark->>'min_accuracy')::numeric between 0 and 1,false)
    or not coalesce((p_benchmark->>'quality_floor')::numeric between -2 and 2,false)
    or exists(select 1 from public.fez_training_assignments where job_id=p_job)
    then raise exception 'invalid qualification authorization'; end if;
  insert into public.zils_qualification_benchmarks values(ctx,p_benchmark) on conflict do nothing;
  if not exists(select 1 from public.zils_qualification_benchmarks where context=ctx and descriptor=p_benchmark) then
    raise exception 'benchmark rubric changed without a new context'; end if;
  update public.zils_job_scheduling set purpose='qualification',context=ctx,benchmark=p_benchmark,
    benchmark_sha256=p_benchmark->>'benchmark_sha256' where job_id=p_job;
  if not found then raise exception 'unverified workload'; end if;
end $$;
create function public.zils_pending_qualification_jobs() returns jsonb language sql stable set search_path='' as $$
  select coalesce(jsonb_agg(jsonb_build_object('id',j.id) order by j.created_at,j.id),'[]'::jsonb)
  from public.fez_training_jobs j join public.zils_job_scheduling s on s.job_id=j.id
  where j.status='awaiting_approval' and s.purpose='qualification' and s.deadline>now()
    and not exists(select 1 from public.zils_worker_reservations r join public.zils_job_scheduling x using(job_id) where x.purpose='qualification');
$$;

create function public.zils_record_graded_evaluation(p_job uuid,p_processor_token uuid,p_observations jsonb) returns void
language plpgsql set search_path='' as $$
declare j public.fez_training_jobs; o jsonb; a public.zils_training_attempts; s public.zils_job_scheduling; q jsonb; report jsonb;
begin
  perform pg_advisory_xact_lock(9135172401);
  select * into j from public.fez_training_jobs where id=p_job and status='evaluating'
    and lease_token=p_processor_token and lease_until>now() for update;
  if not found then raise exception 'processing lease expired'; end if;
  select * into s from public.zils_job_scheduling where job_id=p_job;
  if jsonb_typeof(p_observations) is distinct from 'array' then raise exception 'invalid observations'; end if;
  for o in select * from jsonb_array_elements(p_observations) loop
    select * into a from public.zils_training_attempts where job_id=p_job and hotkey=o->>'hotkey' and lease_token=(o->>'lease_token')::uuid;
    if not found or o->>'job_sha256' is distinct from j.job_sha256 or o->>'sha256' is distinct from a.sha256
      or a.submitted_at is null or o->>'outcome' not in ('valid','invalid_artifact','pending_review','validator_error') then raise exception 'unbound observation'; end if;
    if a.finished_at is not null then
      if a.observation is distinct from o then raise exception 'terminal observation differs'; end if;
    else
      update public.zils_training_attempts set finished_at=now(),outcome=o->>'outcome',observation=o where id=a.id;
      if s.purpose='qualification' and o->>'outcome'='valid' then
        q:=o->'qualification';
        if not coalesce((q->>'baseline_brier')::numeric between 0 and 2 and (q->>'candidate_brier')::numeric between 0 and 2
          and (q->>'uniform_brier')::numeric>0 and (q->>'uniform_brier')::numeric<=2 and (q->>'accuracy')::numeric between 0 and 1
          and (q->>'cases')::int=(j.manifest->'counts'->>'test')::int
          and q->>'calibration_sha256'=j.manifest->'files'->>'calibration.jsonl'
          and q->>'evaluated_sha256' ~ '^[a-f0-9]{64}$'
          and s.context=a.context and s.benchmark_sha256=a.context->>'benchmark_sha256',false) then
          raise exception 'unbound qualification metrics'; end if;
        report:=q||jsonb_build_object('context',a.context,'artifact_valid',true,'verified_at',now(),'expires_at',now()+interval '30 days',
          'min_accuracy',s.benchmark->'min_accuracy','quality_floor',s.benchmark->'quality_floor',
          'seconds_per_token',extract(epoch from a.submitted_at-a.started_at)/a.total_tokens,
          'capacity',j.manifest->'workload','job_id',j.id,'job_sha256',j.job_sha256,'submitted_sha256',a.sha256);
        perform public.zils_import_qualification(a.hotkey,report,'queue-validator',encode(sha256(convert_to(o::text,'UTF8')),'hex'));
      end if;
    end if;
  end loop;
end $$;

create or replace function public.fez_claim_training(p_hotkey text)
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  perform pg_advisory_xact_lock(9135172401);
  perform public.zils_reap_graded();
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1') <> 'imajev-4b-v1' and public.zils_resource_claimable(j.id,p_hotkey) and a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
      and (a.state='ready' or (a.state='leased' and (a.lease_until > now() or a.attempts < 3)))
    order by (a.state='leased' and a.lease_until > now()) desc, j.created_at
    limit 1 for update of j,a skip locked;
  if not found then return null; end if;
  if item.state <> 'leased' or item.lease_until <= now() then
    update public.fez_training_assignments set state='leased',lease_token=gen_random_uuid(),
      lease_until=least(now()+interval '20 minutes', (select deadline from public.fez_training_jobs where id=item.job_id)),
      attempts=attempts+1 where job_id=item.job_id and hotkey=p_hotkey returning * into item;
  end if;
  update public.fez_training_jobs set status='running',updated_at=now() where id=item.job_id;
  return to_jsonb(item);
end $$;

create or replace function public.zils_claim_profile_training(p_hotkey text,p_supported_profiles text[])
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  perform pg_advisory_xact_lock(9135172401);
  perform public.zils_reap_graded();
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  if cardinality(p_supported_profiles) not between 1 and 3 or p_supported_profiles is null
    or exists(select 1 from unnest(p_supported_profiles) x where x is null or not public.zils_worker_qualified(p_hotkey,x))
    then raise exception 'installed profiles require operator qualification'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1') = any(p_supported_profiles) and public.zils_worker_qualified(p_hotkey,coalesce(j.model_profile->>'id', j.manifest->'model'->>'id', 'kev-0.8b-v1')) and public.zils_resource_claimable(j.id,p_hotkey) and a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
      and (a.state='ready' or (a.state='leased' and (a.lease_until > now() or a.attempts < 3)))
    order by (a.state='leased' and a.lease_until > now()) desc, j.created_at
    limit 1 for update of j,a skip locked;
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
  perform pg_advisory_xact_lock(9135172401);
  if exists(select 1 from public.fez_training_jobs where id=p_job and
      coalesce(model_profile->>'id',manifest->'model'->>'id')='imajev-4b-v1')
    and not public.zils_worker_qualified(p_hotkey,'imajev-4b-v1') then
    raise exception 'worker profile qualification revoked'; end if;
  if not public.zils_resource_claimable(p_job,p_hotkey) then raise exception 'resource reservation expired'; end if;
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
  perform pg_advisory_xact_lock(9135172401);
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
  if not public.zils_resource_claimable(p_job,p_hotkey) then raise exception 'resource reservation expired'; end if;
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
  perform pg_advisory_xact_lock(9135172401);
  perform 1 from public.fez_training_jobs where id=p_job and status='awaiting_approval' for update;
  if not found then raise exception 'job is not awaiting approval'; end if;
  if cardinality(p_hotkeys) not between 1 and 16 or
     (select count(*) from public.fez_training_workers where enabled and hotkey=any(p_hotkeys))
       <> cardinality(p_hotkeys) then raise exception 'require 1..16 distinct approved workers'; end if;
  if exists(select 1 from public.fez_training_jobs j, unnest(p_hotkeys) h where j.id=p_job
    and j.model_profile->>'id'='imajev-4b-v1' and not public.zils_worker_qualified(h,'imajev-4b-v1'))
    then raise exception 'image training requires qualified workers'; end if;
  if exists(select 1 from public.zils_job_scheduling where job_id=p_job and graded) then raise exception 'graded job requires graded reservation'; end if;
  insert into public.zils_worker_reservations(resource_id,job_id,hotkey,expires_at,graded)
    select resource_id,p_job,hotkey,now()+interval '24 hours',false from public.fez_training_workers
    where hotkey=any(p_hotkeys) and resource_id is not null;
  insert into public.fez_training_assignments(job_id,hotkey,uid)
    select p_job,hotkey,uid from public.fez_training_workers where hotkey=any(p_hotkeys);
  update public.fez_training_jobs set status='queued',deadline=now()+interval '24 hours',updated_at=now()
    where id=p_job;
end $$;

create or replace function public.fez_fail_training(p_job uuid,p_hotkey text,p_token uuid)
returns void language plpgsql set search_path = '' as $$
begin
  perform pg_advisory_xact_lock(9135172401);
  if exists(select 1 from public.zils_job_scheduling where job_id=p_job and graded) then
    perform public.zils_release_graded_attempt(p_job,p_hotkey,p_token,'pending_review'); return;
  end if;
  update public.fez_training_assignments set state=case when attempts>=3 then 'failed' else 'ready' end,
    lease_token=null,lease_until=null where job_id=p_job and hotkey=p_hotkey
    and lease_token=p_token and state='leased';
end $$;

create or replace function public.zils_claim_profile_processing(p_stage text, p_profiles text[])
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
  perform pg_advisory_xact_lock(9135172401);
  perform public.zils_reap_graded();
  if p_stage not in ('validating','evaluating') then raise exception 'invalid stage'; end if;
  if p_profiles is null or cardinality(p_profiles)=0 then return null; end if;
  if exists(select 1 from unnest(p_profiles) p where p not in ('kev-0.8b-v1','jevk5-4b-v0.3','imajev-4b-v1')) then
    raise exception 'invalid profile';
  end if;
  select * into item from public.fez_training_jobs j where
    (coalesce(j.model_profile->>'id',j.manifest->'model'->>'id','kev-0.8b-v1')=any(p_profiles)
      or (p_stage='validating' and j.model_profile is null and j.manifest is null
          and ('kev-0.8b-v1'=any(p_profiles) or 'jevk5-4b-v0.3'=any(p_profiles))))
    and ((j.status=p_stage and (j.lease_until is null or j.lease_until<=now())) or
      (p_stage='evaluating' and j.status in ('queued','running') and
        (j.deadline<=now() or not exists(select 1 from public.fez_training_assignments a
          where a.job_id=j.id and a.state not in ('submitted','failed')))))
    order by created_at limit 1 for update skip locked;
  if not found then return null; end if;
  update public.fez_training_jobs set status=p_stage,lease_token=gen_random_uuid(),
    lease_until=now()+interval '20 minutes',updated_at=now() where id=item.id returning * into item;
  return item;
end $$;

create or replace function public.fez_finish_processing(p_job uuid,p_token uuid,p_status text,p_values jsonb)
returns void language plpgsql set search_path = '' as $$
begin
  perform pg_advisory_xact_lock(9135172401);
  if p_status not in ('awaiting_approval','completed','failed') then raise exception 'invalid status'; end if;
  if exists(select 1 from public.zils_job_scheduling where job_id=p_job and purpose='qualification') then
    p_values:=p_values||jsonb_build_object('release_prefix',null,'result',coalesce(p_values->'result','{}')||
      jsonb_build_object('delivery',jsonb_build_object('status','qualification_complete')));
  end if;
  if p_values ? 'grading_observations' then
    perform public.zils_record_graded_evaluation(p_job,p_token,p_values->'grading_observations');
  end if;
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

create or replace function public.fez_create_training_job(p_owner uuid, p_name text, p_acceptance jsonb)
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
  perform pg_advisory_xact_lock(9135172401);
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text, 0));
  if (select count(*) from public.fez_training_jobs where owner_id = p_owner
      and status not in ('completed','failed')) >= 5 then
    raise exception 'maximum of five active jobs per account';
  end if;
  insert into public.fez_training_jobs(owner_id,name,acceptance)
    values (p_owner,p_name,p_acceptance) returning * into item;
  return item;
end $$;

create function public.zils_attempt_immutable() returns trigger language plpgsql set search_path='' as $$
begin
  if tg_op='DELETE' or (old.finished_at is not null and new is distinct from old) then
    raise exception 'terminal attempt evidence is immutable'; end if;
  if (new.id,new.job_id,new.hotkey,new.resource_id,new.lease_token,new.context,new.total_tokens,new.started_at)
    is distinct from (old.id,old.job_id,old.hotkey,old.resource_id,old.lease_token,old.context,old.total_tokens,old.started_at) then
    raise exception 'attempt identity is immutable'; end if;
  return new;
end $$;
create trigger zils_attempt_immutable before update or delete on public.zils_training_attempts
for each row execute function public.zils_attempt_immutable();

-- Restrict every new callable, including trigger helpers. Existing replacements retain grants.
do $$ declare f record; begin
  for f in select p.oid::regprocedure signature from pg_proc p join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='public' and p.proname in ('zils_queue_mutex','zils_immutable_qualification','zils_schedule_job_state',
      'zils_worker_heartbeat','zils_import_qualification','zils_grading_snapshot','zils_release_graded_attempt',
      'zils_reap_graded','zils_reserve_graded_job','zils_guard_resource_mapping','zils_assignment_observation',
      'zils_resource_claimable','zils_record_graded_evaluation','zils_attempt_immutable','zils_configure_routing',
      'zils_bind_resource','zils_authorize_qualification','zils_pending_qualification_jobs') loop
    execute format('revoke all on function %s from public,anon,authenticated',f.signature);
    execute format('grant execute on function %s to service_role',f.signature);
  end loop;
end $$;
commit;
