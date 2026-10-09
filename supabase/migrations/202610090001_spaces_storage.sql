-- Private logical locators survive provider changes; deleted locators never fall back.
begin;
create table public.zils_storage_objects (
  bucket text not null,
  path text not null,
  provider text not null check (provider in ('supabase','spaces')),
  generation uuid not null unique default gen_random_uuid(),
  physical_bucket text not null,
  physical_key text not null,
  upload_id text,
  state text not null check (state in ('allocating','uploading','sealing','ready','deleting','deleted')),
  token uuid not null,
  lease_until timestamptz not null default now(),
  grant_expires_at timestamptz not null default now(),
  cleanup_after timestamptz not null default now(),
  max_bytes bigint not null check (max_bytes between 1 and 536870912),
  part_etag text,
  part_size bigint check (part_size > 0 and part_size <= max_bytes),
  size_bytes bigint check (size_bytes > 0 and size_bytes <= max_bytes),
  sha256 text check (sha256 ~ '^[a-f0-9]{64}$'),
  legacy_readable boolean not null default false,
  updated_at timestamptz not null default now(),
  primary key (bucket,path),
  unique(provider,physical_bucket,physical_key)
);
-- Superseded attempts retain cleanup provenance even after a replacement starts.
create table public.zils_storage_retired (
  generation uuid primary key,
  object jsonb not null,
  safe_after timestamptz not null
);
alter table public.zils_storage_objects enable row level security;
alter table public.zils_storage_retired enable row level security;
revoke all on public.zils_storage_objects,public.zils_storage_retired from public,anon,authenticated;
grant select,insert,update,delete on public.zils_storage_objects,public.zils_storage_retired to service_role;

create function public.zils_storage_object(
  p_bucket text,p_path text,p_action text,p_token uuid default null,p_values jsonb default '{}'::jsonb
) returns jsonb language plpgsql security definer set search_path=public,pg_temp as $$
declare
  r zils_storage_objects;
  stamp timestamptz := clock_timestamp();
  cap bigint;
  size bigint;
  part bigint;
  expiry timestamptz;
  generation_id uuid;
  provider_name text;
  copying boolean := false;
begin
  cap := case p_bucket when 'fez-training-data' then 134217728
    when 'fez-training-models' then 536870912 when 'zils-images' then 10485760
    when 'zils-api-batches' then 26214400 else null end;
  if cap is null or p_path is null or length(p_path) not between 1 and 1024
    or p_path ~ '(^/|/$|//|(^|/)\.{1,2}(/|$)|[[:cntrl:]]|%)'
    or position(chr(92) in p_path)>0 or jsonb_typeof(p_values)<>'object' then
    return '{"error":"invalid"}'::jsonb;
  end if;
  if p_action='get' then
    select * into r from zils_storage_objects where bucket=p_bucket and path=p_path;
    if not found then return null; end if;
    return to_jsonb(r);
  end if;
  if p_token is null then return '{"error":"invalid"}'::jsonb; end if;
  perform pg_advisory_xact_lock(hashtextextended(p_bucket||':'||p_path,914));
  select * into r from zils_storage_objects where bucket=p_bucket and path=p_path for update;

  if p_action in ('allocate','register_legacy','begin_copy') then
    if p_action='begin_copy' then
      if r.bucket is null or r.state in ('deleting','deleted') then return '{"error":"conflict"}'::jsonb; end if;
      if r.legacy_readable and r.token=p_token and r.lease_until>stamp then return to_jsonb(r); end if;
      if not (r.provider='supabase' and r.state='ready') and not
        (r.legacy_readable and r.lease_until<=stamp and r.grant_expires_at<=stamp) then
        return '{"error":"conflict"}'::jsonb;
      end if;
      if coalesce(p_values->>'sha256','') !~ '^[a-f0-9]{64}$' then return '{"error":"invalid"}'::jsonb; end if;
      copying := true;
      provider_name := 'spaces';
    elsif r.bucket is not null then
      if p_action='register_legacy' and r.provider='supabase' and r.state='ready'
        and r.size_bytes=(p_values->>'size_bytes')::bigint then return to_jsonb(r); end if;
      if p_action='allocate' and r.state in ('allocating','uploading') and r.token=p_token
        and r.lease_until>stamp and not r.legacy_readable then return to_jsonb(r); end if;
      if p_action='register_legacy' or r.state not in ('allocating','uploading')
        or r.legacy_readable or r.lease_until>stamp or r.grant_expires_at>stamp then
        return '{"error":"conflict"}'::jsonb;
      end if;
    end if;
    provider_name := coalesce(provider_name,case when p_action='register_legacy' then 'supabase' else p_values->>'provider' end);
    if provider_name is null or provider_name not in ('supabase','spaces') then return '{"error":"invalid"}'::jsonb; end if;
    if provider_name='spaces' and coalesce(p_values->>'physical_bucket','') !~ '^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$' then
      return '{"error":"invalid"}'::jsonb;
    end if;
    if p_values ? 'max_bytes' then
      if (p_values->>'max_bytes')::bigint not between 1 and cap then return '{"error":"invalid"}'::jsonb; end if;
      cap := (p_values->>'max_bytes')::bigint;
    end if;
    if p_action='register_legacy' then
      size := (p_values->>'size_bytes')::bigint;
      if size is null or size not between 1 and cap then return '{"error":"invalid"}'::jsonb; end if;
    end if;
    if r.bucket is not null and r.provider='spaces' then
      insert into zils_storage_retired values(r.generation,to_jsonb(r),
        greatest(stamp,r.lease_until,r.grant_expires_at)+interval '24 hours') on conflict do nothing;
    end if;
    generation_id := gen_random_uuid();
    insert into zils_storage_objects(bucket,path,provider,generation,physical_bucket,physical_key,state,
      token,lease_until,grant_expires_at,max_bytes,size_bytes,sha256,legacy_readable)
    values(p_bucket,p_path,provider_name,generation_id,
      case when provider_name='spaces' then p_values->>'physical_bucket' else p_bucket end,
      case when provider_name='spaces' then 'objects/'||generation_id||'/'||p_bucket||'/'||p_path else p_path end,
      case when p_action='register_legacy' then 'ready' when provider_name='spaces' then 'allocating' else 'uploading' end,
      p_token,stamp+interval '300 seconds',stamp,cap,case when copying then r.size_bytes else size end,
      p_values->>'sha256',copying)
    on conflict(bucket,path) do update set provider=excluded.provider,generation=excluded.generation,
      physical_bucket=excluded.physical_bucket,physical_key=excluded.physical_key,state=excluded.state,
      token=excluded.token,lease_until=excluded.lease_until,grant_expires_at=excluded.grant_expires_at,
      max_bytes=excluded.max_bytes,size_bytes=excluded.size_bytes,sha256=excluded.sha256,
      legacy_readable=excluded.legacy_readable,upload_id=null,part_etag=null,part_size=null,updated_at=stamp
    returning * into r;
    return to_jsonb(r);
  end if;

  if p_action='delete' then
    if r.bucket is null then
      insert into zils_storage_objects(bucket,path,provider,physical_bucket,physical_key,state,token,max_bytes)
        values(p_bucket,p_path,'supabase',p_bucket,p_path,'deleted',p_token,cap) returning * into r;
      return to_jsonb(r);
    end if;
    if r.state='deleted' or (r.state='deleting' and r.token=p_token) then return to_jsonb(r); end if;
    if r.state='deleting' and r.lease_until>stamp then return '{"error":"conflict"}'::jsonb; end if;
    update zils_storage_objects set state='deleting',token=p_token,lease_until=stamp+interval '300 seconds',
      cleanup_after=case when r.state='deleting' then r.cleanup_after
        else greatest(stamp,r.lease_until,r.grant_expires_at)+interval '300 seconds' end,
      legacy_readable=false,updated_at=stamp where bucket=p_bucket and path=p_path returning * into r;
    return to_jsonb(r);
  end if;
  if r.bucket is null or r.generation is distinct from (p_values->>'generation')::uuid then
    return '{"error":"conflict"}'::jsonb;
  end if;
  if p_action='seal' then
    if r.state='ready' then return to_jsonb(r); end if;
    if r.state not in ('uploading','sealing') or (r.state='sealing' and r.lease_until>stamp and r.token<>p_token)
      or (r.legacy_readable and r.token<>p_token) then return '{"error":"conflict"}'::jsonb; end if;
    part := (p_values->>'part_size')::bigint;
    if part is null or part not between 1 and r.max_bytes or
      (r.provider='spaces' and (r.upload_id is null or coalesce(p_values->>'part_etag','') !~ '^"[a-fA-F0-9]{32}"$')) then
      return '{"error":"invalid"}'::jsonb;
    end if;
    if r.state='sealing' and r.token=p_token and r.lease_until>stamp then
      if r.part_size=part and r.part_etag is not distinct from p_values->>'part_etag' then return to_jsonb(r); end if;
      return '{"error":"conflict"}'::jsonb;
    end if;
    update zils_storage_objects set state='sealing',token=p_token,lease_until=stamp+interval '300 seconds',
      part_etag=p_values->>'part_etag',part_size=part,updated_at=stamp
      where bucket=p_bucket and path=p_path returning * into r;
    return to_jsonb(r);
  end if;
  if p_action='grant' then
    if r.token<>p_token or r.state<>'uploading' or (r.provider='spaces' and r.upload_id is null) or r.legacy_readable then
      return '{"error":"conflict"}'::jsonb;
    end if;
    if r.provider='spaces' then
      if coalesce((p_values->>'seconds')::integer,0) not between 1 and 600 then return '{"error":"invalid"}'::jsonb; end if;
      expiry := stamp+make_interval(secs=>(p_values->>'seconds')::integer);
    else
      expiry := (p_values->>'expires_at')::timestamptz;
      if expiry is null or expiry<=stamp or expiry>stamp+interval '1 day' then return '{"error":"invalid"}'::jsonb; end if;
    end if;
    update zils_storage_objects set grant_expires_at=greatest(expiry,grant_expires_at),updated_at=stamp
      where bucket=p_bucket and path=p_path returning * into r;
    return to_jsonb(r);
  end if;
  if r.token<>p_token then return '{"error":"conflict"}'::jsonb; end if;
  if p_action='commit' and r.state='ready' then
    if r.size_bytes=(p_values->>'size_bytes')::bigint and r.sha256 is not distinct from p_values->>'sha256' then return to_jsonb(r); end if;
    return '{"error":"conflict"}'::jsonb;
  end if;
  if p_action='bind' and r.state='uploading' and r.upload_id=p_values->>'upload_id' then return to_jsonb(r); end if;
  if r.lease_until<=stamp then return '{"error":"conflict"}'::jsonb; end if;
  if p_action='bind' and r.state='allocating' and r.provider='spaces' then
    if coalesce(length(p_values->>'upload_id'),0) not between 1 and 2048 then return '{"error":"invalid"}'::jsonb; end if;
    update zils_storage_objects set upload_id=p_values->>'upload_id',state='uploading',updated_at=stamp
      where bucket=p_bucket and path=p_path returning * into r;
  elsif p_action='renew' and r.state in ('allocating','uploading','sealing','deleting') then
    update zils_storage_objects set lease_until=stamp+interval '300 seconds',updated_at=stamp
      where bucket=p_bucket and path=p_path returning * into r;
  elsif p_action='commit' and r.state='sealing' then
    size := (p_values->>'size_bytes')::bigint;
    if size is null or size not between 1 and r.max_bytes or size<>r.part_size
      or (r.legacy_readable and (p_values->>'sha256' is distinct from r.sha256 or r.sha256 is null)) then
      return '{"error":"invalid"}'::jsonb;
    end if;
    update zils_storage_objects set state='ready',size_bytes=size,sha256=p_values->>'sha256',
      legacy_readable=false,updated_at=stamp where bucket=p_bucket and path=p_path returning * into r;
  elsif p_action='deleted' and r.state='deleting' and r.cleanup_after<=stamp then
    update zils_storage_objects set state='deleted',updated_at=stamp
      where bucket=p_bucket and path=p_path returning * into r;
  else return '{"error":"conflict"}'::jsonb;
  end if;
  return to_jsonb(r);
exception when invalid_text_representation or numeric_value_out_of_range or check_violation
  or datetime_field_overflow then return '{"error":"invalid"}'::jsonb;
end $$;

create function public.zils_storage_pending(p_before timestamptz,p_limit integer default 100)
returns setof jsonb language sql security definer set search_path=public,pg_temp as $$
  select to_jsonb(o) from zils_storage_objects o
    where state not in ('ready','deleted') and lease_until<=least(p_before,clock_timestamp())
      and grant_expires_at<=least(p_before,clock_timestamp())
    order by updated_at,bucket,path limit greatest(0,least(p_limit,100));
$$;
revoke all on function public.zils_storage_object(text,text,text,uuid,jsonb),
  public.zils_storage_pending(timestamptz,integer) from public,anon,authenticated;
grant execute on function public.zils_storage_object(text,text,text,uuid,jsonb),
  public.zils_storage_pending(timestamptz,integer) to service_role;
commit;
