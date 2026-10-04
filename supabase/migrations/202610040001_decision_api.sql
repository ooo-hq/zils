-- Zils inference credentials/admission. Apply once; training resources are unchanged.
create table public.zils_api_accounts (
  owner_id uuid primary key references auth.users(id) on delete cascade,
  enabled boolean not null default true,
  requests_per_second integer check (requests_per_second > 0),
  tokens_per_second bigint check (tokens_per_second > 0),
  max_active_batches integer check (max_active_batches > 0),
  max_batch_storage_bytes bigint check (max_batch_storage_bytes >= 26214400),
  created_at timestamptz not null default now()
);
create table public.zils_api_keys (
  id uuid primary key,
  owner_id uuid not null references public.zils_api_accounts(owner_id) on delete cascade,
  name text not null check (length(name) between 1 and 80),
  prefix text not null,
  digest text not null check (digest ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now(),
  revoked_at timestamptz
);
create index on public.zils_api_keys(owner_id);
create table public.zils_api_windows (
  owner_id uuid primary key references public.zils_api_accounts(owner_id) on delete cascade,
  second timestamptz not null,
  requests bigint not null,
  tokens bigint not null
);
create table public.zils_api_usage (
  request_id uuid primary key,
  owner_id uuid not null references public.zils_api_accounts(owner_id) on delete cascade,
  created_at timestamptz not null default now(),
  reserved_tokens bigint not null check (reserved_tokens >= 0),
  input_tokens bigint check (input_tokens >= 0),
  status text not null default 'started' check (status in ('started','completed','failed'))
);
create index on public.zils_api_usage(created_at);

create function public.zils_api_create_key(p_key jsonb) returns jsonb
language plpgsql security definer set search_path = public, pg_temp as $$
declare result zils_api_keys;
begin
  insert into zils_api_accounts(owner_id) values ((p_key->>'owner_id')::uuid) on conflict do nothing;
  if not (select enabled from zils_api_accounts where owner_id=(p_key->>'owner_id')::uuid) then
    raise exception 'account disabled';
  end if;
  insert into zils_api_keys(id,owner_id,name,prefix,digest) values
    ((p_key->>'id')::uuid,(p_key->>'owner_id')::uuid,p_key->>'name',p_key->>'prefix',p_key->>'digest') returning * into result;
  return to_jsonb(result) - 'digest' - 'owner_id';
end $$;
create function public.zils_api_auth(p_id uuid) returns setof public.zils_api_keys
language sql security definer set search_path = public, pg_temp as $$
  select k.* from zils_api_keys k join zils_api_accounts a using(owner_id)
  where k.id=p_id and k.revoked_at is null and a.enabled
$$;
create function public.zils_api_admit(p_owner uuid,p_key uuid,p_request uuid,p_tokens bigint) returns text
language plpgsql security definer set search_path = public, pg_temp as $$
declare a zils_api_accounts; w zils_api_windows; current_second timestamptz;
begin
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
create function public.zils_api_finish_usage(p_request uuid,p_tokens bigint,p_status text) returns void
language sql security definer set search_path = public, pg_temp as $$
  update zils_api_usage set input_tokens=p_tokens,status=p_status
  where request_id=p_request and status='started' and p_status in ('completed','failed')
$$;

do $$ declare t text; f regprocedure;
begin
  foreach t in array array['zils_api_accounts','zils_api_keys','zils_api_windows','zils_api_usage'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('revoke all on public.%I from public,anon,authenticated',t);
    execute format('grant all on public.%I to service_role',t);
  end loop;
  for f in select oid::regprocedure from pg_proc where pronamespace='public'::regnamespace and proname in
    ('zils_api_create_key','zils_api_auth','zils_api_admit','zils_api_finish_usage') loop
    execute format('revoke all on function %s from public,anon,authenticated',f);
    execute format('grant execute on function %s to service_role',f);
  end loop;
end $$;
