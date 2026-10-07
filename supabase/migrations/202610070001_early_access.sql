-- Zils-only admission. Review rollout: existing accounts need explicit approval.
-- Does not change project-wide Supabase Auth or unrelated applications.
begin;
create table public.zils_access_settings (
  singleton boolean primary key default true check (singleton),
  capacity integer not null default 25 check (capacity between 1 and 100000)
);
insert into public.zils_access_settings default values;
create table public.zils_access_applications (
  id uuid primary key default gen_random_uuid(),
  email text not null unique check (email=lower(btrim(email)) and length(email) between 3 and 254),
  use_case text not null default '' check (length(use_case)<=2000),
  status text not null default 'waiting' check (status in ('waiting','invited','active','paused')),
  owner_id uuid unique references auth.users(id) on delete set null,
  created_at timestamptz not null default now(),
  invited_at timestamptz,
  expires_at timestamptz,
  activated_at timestamptz,
  invited_by uuid references auth.users(id) on delete set null,
  invitation_version uuid,
  delivery_status text check (delivery_status in ('pending','sent','failed'))
);
create index on public.zils_access_applications(status,created_at,id);
create table public.zils_access_attempts (
  source text primary key check (source ~ '^[a-f0-9]{64}$'),
  window_start timestamptz not null,
  attempts integer not null
);

create function public.zils_access_apply(p_email text,p_use_case text,p_source text) returns text
language plpgsql security definer set search_path=public,pg_temp as $$
declare n integer;
begin
  if p_email is null or length(btrim(p_email)) not between 3 and 254 or p_email not like '%@%'
    or p_use_case is null or length(btrim(p_use_case)) not between 1 and 2000 then raise exception 'invalid application'; end if;
  delete from zils_access_attempts where window_start < now()-interval '2 hours';
  insert into zils_access_attempts values(p_source,date_trunc('hour',now()),1)
    on conflict(source) do update set
      attempts=case when zils_access_attempts.window_start=excluded.window_start then zils_access_attempts.attempts+1 else 1 end,
      window_start=excluded.window_start returning attempts into n;
  if n>10 then return 'limited'; end if;
  -- Never let an unauthenticated repeat overwrite someone's application/approval.
  insert into zils_access_applications(email,use_case) values(lower(btrim(p_email)),btrim(p_use_case)) on conflict(email) do nothing;
  return 'accepted';
end $$;

create function public.zils_access_allowed(p_owner uuid) returns boolean
language sql stable security definer set search_path=public,pg_temp as $$
  select exists(select 1 from zils_access_applications where owner_id=p_owner and status='active')
$$;

create function public.zils_access_claim(p_owner uuid) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare account auth.users; item zils_access_applications;
begin
  select * into account from auth.users where id=p_owner;
  if not found or account.email_confirmed_at is null then return jsonb_build_object('status','waiting'); end if;
  perform 1 from zils_access_settings where singleton for update;
  select * into item from zils_access_applications where owner_id=p_owner for update;
  if not found then
    select * into item from zils_access_applications where email=lower(btrim(account.email)) for update;
  end if;
  if not found then return jsonb_build_object('status','waiting'); end if;
  if (item.status='active' and item.owner_id is null) or (item.owner_id is not null and item.owner_id<>p_owner) then return jsonb_build_object('status','waiting'); end if;
  if item.status='invited' then
    if item.expires_at<=now() or item.expires_at is null then return jsonb_build_object('status','expired'); end if;
    update zils_access_applications set status='active',owner_id=p_owner,activated_at=now() where id=item.id returning * into item;
  end if;
  return jsonb_build_object('status',item.status);
end $$;

create function public.zils_access_invite(p_email text,p_action text,p_actor uuid) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare cap integer; used integer; item zils_access_applications;
begin
  if p_email is null or length(btrim(p_email)) not between 3 and 254 or p_email not like '%@%'
    or p_action is null or p_action not in ('approve','resend','pause') or p_actor is null then raise exception 'invalid action'; end if;
  -- Serializes ALL seat changes, including invite acceptance and simultaneous approvals.
  select capacity into cap from zils_access_settings where singleton for update;
  select a.* into item from zils_access_applications a join auth.users u on u.id=a.owner_id
    where lower(btrim(u.email))=lower(btrim(p_email)) and u.email_confirmed_at is not null for update of a;
  if not found then
    select * into item from zils_access_applications where email=lower(btrim(p_email)) for update;
  end if;
  if p_action='pause' then
    if not found then return jsonb_build_object('status','not_found','send',false); end if;
    update zils_access_applications set status='paused',expires_at=null where id=item.id returning * into item;
    return jsonb_build_object('status','ok','application',to_jsonb(item),'send',false);
  end if;
  if item.status='active' or (p_action='approve' and item.status='invited' and item.expires_at>now()) then
    return jsonb_build_object('status','ok','application',to_jsonb(item),'send',false);
  end if;
  if p_action='resend' and item.invited_at>now()-interval '1 minute' then return jsonb_build_object('status','wait','send',false); end if;
  select count(*) into used from zils_access_applications
    where (status='active' or (status='invited' and expires_at>now())) and id is distinct from item.id;
  if used>=cap then return jsonb_build_object('status','full','send',false); end if;
  if item.id is not null then
    update zils_access_applications set status='invited',invited_at=now(),expires_at=now()+interval '7 days',
      invited_by=p_actor,invitation_version=gen_random_uuid(),delivery_status='pending' where id=item.id returning * into item;
  else
    insert into zils_access_applications(email,status,invited_at,expires_at,invited_by,invitation_version,delivery_status)
      values(lower(btrim(p_email)),'invited',now(),now()+interval '7 days',p_actor,gen_random_uuid(),'pending')
      on conflict(email) do update set status='invited',invited_at=excluded.invited_at,expires_at=excluded.expires_at,
        invited_by=excluded.invited_by,invitation_version=excluded.invitation_version,delivery_status='pending'
      returning * into item;
  end if;
  return jsonb_build_object('status','ok','application',to_jsonb(item),'send',true);
end $$;

create function public.zils_access_overview(p_status text default 'all',p_offset integer default 0) returns jsonb
language plpgsql stable security definer set search_path=public,pg_temp as $$
declare result jsonb;
begin
  if p_status not in ('all','waiting','invited','active','paused','expired') or p_offset<0 then raise exception 'invalid filter'; end if;
  with visible as (
    select *,case when status='invited' and expires_at<=now() then 'expired' else status end as display_status
    from zils_access_applications
  ), matching as (select * from visible where p_status='all' or display_status=p_status),
  page as (select * from matching order by created_at,id limit 50 offset p_offset)
  select jsonb_build_object(
    'capacity',(select capacity from zils_access_settings where singleton),
    'allocated',(select count(*) from visible where display_status in ('active','invited')),
    'waiting',(select count(*) from visible where display_status='waiting'),
    'total',(select count(*) from matching),
    'applications',coalesce((select jsonb_agg(to_jsonb(page)) from page),'[]'::jsonb)) into result;
  return result;
end $$;

-- Admission is checked again inside the existing credential and usage transactions.
create or replace function public.zils_api_create_key(p_key jsonb) returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare result zils_api_keys;
begin
  if not zils_access_allowed((p_key->>'owner_id')::uuid) then raise exception 'early access required'; end if;
  insert into zils_api_accounts(owner_id) values ((p_key->>'owner_id')::uuid) on conflict do nothing;
  if not (select enabled from zils_api_accounts where owner_id=(p_key->>'owner_id')::uuid) then
    raise exception 'account disabled';
  end if;
  insert into zils_api_keys(id,owner_id,name,prefix,digest) values
    ((p_key->>'id')::uuid,(p_key->>'owner_id')::uuid,p_key->>'name',p_key->>'prefix',p_key->>'digest') returning * into result;
  return to_jsonb(result) - 'digest' - 'owner_id';
end $$;

create or replace function public.zils_api_auth(p_id uuid) returns setof public.zils_api_keys
language sql security definer set search_path = public, pg_temp as $$
  select k.* from zils_api_keys k join zils_api_accounts a using(owner_id)
  where k.id=p_id and k.revoked_at is null and a.enabled and zils_access_allowed(k.owner_id)
$$;

create or replace function public.zils_api_admit(p_owner uuid,p_key uuid,p_request uuid,p_tokens bigint) returns text
language plpgsql security definer set search_path = public, pg_temp as $$
declare a zils_api_accounts; w zils_api_windows; current_second timestamptz;
begin
  if not zils_access_allowed(p_owner) then return 'disabled'; end if;
  if p_tokens < 0 or p_tokens is null then raise exception 'invalid reservation'; end if;
  -- Every key and trusted batch worker for an account shares this lock.
  select * into a from zils_api_accounts where owner_id=p_owner for update;
  if not found or not a.enabled then return 'disabled'; end if;
  if p_key is not null and not exists(select 1 from zils_api_keys where id=p_key and owner_id=p_owner and revoked_at is null) then return 'disabled'; end if;
  if exists(select 1 from zils_api_usage where request_id=p_request) then return 'duplicate'; end if;
  current_second := date_trunc('second',clock_timestamp());
  insert into zils_api_windows values(p_owner,current_second,0,0) on conflict do nothing;
  select * into w from zils_api_windows where owner_id=p_owner;
  if w.second <> current_second then w.requests:=0; w.tokens:=0; end if;
  if (a.requests_per_second is not null and w.requests+1>a.requests_per_second)
    or (a.tokens_per_second is not null and w.tokens+p_tokens>a.tokens_per_second) then return 'limited'; end if;
  update zils_api_windows set second=current_second,requests=w.requests+1,tokens=w.tokens+p_tokens where owner_id=p_owner;
  insert into zils_api_usage(request_id,owner_id,reserved_tokens) values(p_request,p_owner,p_tokens);
  return 'allowed';
end $$;

create or replace function public.fez_create_training_job(p_owner uuid, p_name text, p_acceptance jsonb)
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
  if not public.zils_access_allowed(p_owner) then raise exception 'early access required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text, 0));
  if (select count(*) from public.fez_training_jobs where owner_id = p_owner
      and status not in ('completed','failed')) >= 5 then
    raise exception 'maximum of five active jobs per account';
  end if;
  insert into public.fez_training_jobs(owner_id,name,acceptance)
    values (p_owner,p_name,p_acceptance) returning * into item;
  return item;
end $$;


do $$ declare t text; f regprocedure;
begin
  foreach t in array array['zils_access_settings','zils_access_applications','zils_access_attempts'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('revoke all on public.%I from public,anon,authenticated',t);
    execute format('grant all on public.%I to service_role',t);
  end loop;
  for f in select oid::regprocedure from pg_proc where pronamespace='public'::regnamespace and proname like 'zils_access_%' loop
    execute format('revoke all on function %s from public,anon,authenticated',f);
    execute format('grant execute on function %s to service_role',f);
  end loop;
end $$;
commit;
