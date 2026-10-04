-- Durable bulk inference, isolated from customer training jobs and worker credentials.
insert into storage.buckets(id,name,public,file_size_limit)
values ('zils-api-batches','zils-api-batches',false,26214400);
create policy zils_api_private_batches on storage.objects as restrictive
for all to anon,authenticated using (bucket_id <> 'zils-api-batches') with check (bucket_id <> 'zils-api-batches');
create table public.zils_api_batches (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references public.zils_api_accounts(owner_id),
  idempotency_key text not null check (length(idempotency_key) between 1 and 128),
  status text not null default 'uploading' check (status in ('uploading','queued','validating','running','completed','failed','cancelled','expired')),
  created_at timestamptz not null default now(),
  deadline timestamptz not null default now()+interval '24 hours',
  finished_at timestamptz,
  next_at timestamptz not null default now(),
  lease_token uuid,
  lease_until timestamptz,
  catalog jsonb,
  input_path text not null,
  total integer not null default 0,
  completed integer not null default 0,
  failed integer not null default 0,
  error jsonb,
  purged_at timestamptz,
  unique(owner_id,idempotency_key)
);
create index on public.zils_api_batches(next_at) where status in ('queued','validating','running');
create index on public.zils_api_batches(finished_at);
create table public.zils_api_batch_items (
  batch_id uuid not null references public.zils_api_batches(id) on delete cascade,
  line integer not null check (line > 0),
  custom_id text not null,
  body text,
  frozen jsonb,
  result jsonb,
  attempts integer not null default 0,
  primary key(batch_id,line),
  unique(batch_id,custom_id)
);
create index on public.zils_api_batch_items(batch_id,line) where result is null;

create function public.zils_api_batch_create(p_owner uuid,p_key text) returns public.zils_api_batches
language plpgsql security definer set search_path=public,pg_temp as $$
declare batch zils_api_batches; new_id uuid:=gen_random_uuid(); account zils_api_accounts; active bigint; retained bigint;
begin
  select * into account from zils_api_accounts where owner_id=p_owner for update;
  if not found or not account.enabled then raise exception 'account disabled'; end if;
  select * into batch from zils_api_batches where owner_id=p_owner and idempotency_key=p_key;
  if found then return batch; end if;
  select count(*) filter(where status in ('uploading','queued','validating','running')),
         count(*) filter(where purged_at is null) into active,retained
  from zils_api_batches where owner_id=p_owner;
  if (account.max_active_batches is not null and active>=account.max_active_batches)
    or (account.max_batch_storage_bytes is not null and (retained+1)*26214400>account.max_batch_storage_bytes) then return null; end if;
  insert into zils_api_batches(id,owner_id,idempotency_key,input_path)
  values(new_id,p_owner,p_key,p_owner::text||'/'||new_id::text||'/input.jsonl')
  on conflict(owner_id,idempotency_key) do nothing;
  select * into batch from zils_api_batches where owner_id=p_owner and idempotency_key=p_key;
  return batch;
end $$;
create function public.zils_api_batch_action(p_owner uuid,p_id uuid,p_action text,p_catalog jsonb default null) returns public.zils_api_batches
language plpgsql security definer set search_path=public,pg_temp as $$
declare batch zils_api_batches;
begin
  select * into batch from zils_api_batches where id=p_id and owner_id=p_owner for update;
  if not found then return null; end if;
  if batch.deadline<=clock_timestamp() and batch.status in ('uploading','queued','validating','running') then
    update zils_api_batches set status='expired',finished_at=clock_timestamp(),lease_token=null,lease_until=null where id=p_id;
  elsif p_action='submit' and batch.status='uploading' then
    if jsonb_typeof(p_catalog)<>'object' or p_catalog is null then raise exception 'invalid catalog'; end if;
    update zils_api_batches set status='queued',catalog=p_catalog where id=p_id;
  elsif p_action='cancel' and batch.status in ('uploading','queued','validating','running') then
    update zils_api_batches set status='cancelled',finished_at=clock_timestamp(),lease_token=null,lease_until=null where id=p_id;
  elsif p_action not in ('submit','cancel') then raise exception 'invalid action';
  end if;
  select * into batch from zils_api_batches where id=p_id;
  return batch;
end $$;
create function public.zils_api_batch_claim() returns public.zils_api_batches
language plpgsql security definer set search_path=public,pg_temp as $$
declare batch zils_api_batches;
begin
  update zils_api_batches set status='expired',finished_at=clock_timestamp(),lease_token=null,lease_until=null
  where deadline<=clock_timestamp() and status in ('uploading','queued','validating','running');
  select b.* into batch from zils_api_batches b join zils_api_accounts a using(owner_id)
  where b.status in ('queued','validating','running') and b.next_at<=clock_timestamp() and b.deadline>clock_timestamp()
    and (b.lease_until is null or b.lease_until<=clock_timestamp()) and a.enabled
  order by b.next_at,b.created_at for update of b skip locked limit 1;
  if not found then return null; end if;
  if batch.status in ('queued','validating') then
    delete from zils_api_batch_items where batch_id=batch.id;
    batch.status:='validating';
  end if;
  update zils_api_batches set status=batch.status,lease_token=gen_random_uuid(),lease_until=clock_timestamp()+interval '120 seconds'
  where id=batch.id returning * into batch;
  return batch;
end $$;
create function public.zils_api_batch_work(p_id uuid,p_lease uuid,p_action text,p_data jsonb default '{}') returns void
language plpgsql security definer set search_path=public,pg_temp as $$
declare batch zils_api_batches; item zils_api_batch_items; n integer;
begin
  select * into batch from zils_api_batches where id=p_id for update;
  if not found or batch.status not in ('validating','running') or batch.lease_token is distinct from p_lease
    or batch.lease_until<=clock_timestamp() or batch.deadline<=clock_timestamp()
    or not exists(select 1 from zils_api_accounts where owner_id=batch.owner_id and enabled) then raise exception 'batch lease expired'; end if;
  if p_action='renew' then
    update zils_api_batches set lease_until=clock_timestamp()+interval '120 seconds' where id=p_id;
  elsif p_action='load' and batch.status='validating' then
    insert into zils_api_batch_items(batch_id,line,custom_id,body,frozen,result)
    select p_id,(x->>'line')::integer,x->>'custom_id',x->>'body',x->'frozen',nullif(x->'result','null'::jsonb)
    from jsonb_array_elements(p_data) x;
  elsif p_action='ready' and batch.status='validating' then
    select count(*) into n from zils_api_batch_items where batch_id=p_id;
    if n<1 or n>10000 or n<>(p_data->>'total')::integer then raise exception 'invalid input count'; end if;
    update zils_api_batches set total=n,status='running' where id=p_id;
  elsif p_action='result' and batch.status='running' then
    select * into item from zils_api_batch_items where batch_id=p_id and line=(p_data->>'line')::integer;
    if not found then raise exception 'missing item'; end if;
    if item.result is null then
      if p_data->'result' is null or p_data->'result'='null'::jsonb then raise exception 'missing result'; end if;
      update zils_api_batch_items set result=p_data->'result' where batch_id=p_id and line=item.line;
      if p_data->>'request_id' is not null then
        update zils_api_usage set input_tokens=(p_data->>'input_tokens')::bigint,status='completed'
        where request_id=(p_data->>'request_id')::uuid and owner_id=batch.owner_id and status='started';
        if not found then raise exception 'missing usage reservation'; end if;
      end if;
    end if;
  elsif p_action='retry' and batch.status='running' then
    update zils_api_batch_items set attempts=attempts+1 where batch_id=p_id and line=(p_data->>'line')::integer and result is null;
  elsif p_action='fail' then
    update zils_api_batches set status='failed',finished_at=clock_timestamp(),error=p_data,lease_token=null,lease_until=null where id=p_id;
    return;
  elsif p_action<>'release' then raise exception 'invalid batch work action';
  end if;
  if p_action in ('ready','result','release') then
    update zils_api_batches set
      completed=(select count(*) from zils_api_batch_items where batch_id=p_id and result ? 'response'),
      failed=(select count(*) from zils_api_batch_items where batch_id=p_id and result ? 'error') where id=p_id;
    if (select status='running' and total=completed+failed from zils_api_batches where id=p_id) then
      update zils_api_batches set status='completed',finished_at=clock_timestamp(),lease_token=null,lease_until=null where id=p_id;
    end if;
  end if;
  if p_action='release' then
    update zils_api_batches set lease_token=null,lease_until=null,next_at=clock_timestamp()+make_interval(secs=>least(60,greatest(0,coalesce((p_data->>'delay')::integer,0)))) where id=p_id;
  end if;
end $$;
-- Called only after the private input object has been removed successfully.
create function public.zils_api_batch_purge(p_id uuid) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  perform 1 from zils_api_batches where id=p_id and finished_at<clock_timestamp()-interval '7 days' for update;
  if not found then raise exception 'retention period not reached'; end if;
  delete from zils_api_batch_items where batch_id=p_id;
  update zils_api_batches set purged_at=clock_timestamp(),catalog=null where id=p_id;
end $$;
do $$ declare t text; f regprocedure;
begin
  foreach t in array array['zils_api_batches','zils_api_batch_items'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('revoke all on public.%I from public,anon,authenticated',t);
    execute format('grant all on public.%I to service_role',t);
  end loop;
  for f in select oid::regprocedure from pg_proc where pronamespace='public'::regnamespace and proname in
    ('zils_api_batch_create','zils_api_batch_action','zils_api_batch_claim','zils_api_batch_work','zils_api_batch_purge') loop
    execute format('revoke all on function %s from public,anon,authenticated',f);
    execute format('grant execute on function %s to service_role',f);
  end loop;
end $$;
