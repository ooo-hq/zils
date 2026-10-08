-- Private, immutable image assets. No production feature is enabled by this migration.
begin;
alter table public.fez_training_jobs add column model_profile jsonb;
create table public.zils_image_assets (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id),
  purpose text not null check (purpose in ('prediction','training')),
  job_id uuid references public.fez_training_jobs(id),
  filename text not null check (length(filename) between 1 and 255),
  state text not null default 'uploading' check (state in ('uploading','verifying','ready','failed','expired','deleted')),
  source_path text not null unique,
  canonical_path text not null unique,
  source_bytes bigint not null check (source_bytes between 1 and 10485760),
  source_sha256 text not null check (source_sha256 ~ '^[a-f0-9]{64}$'),
  canonical_sha256 text check (canonical_sha256 ~ '^[a-f0-9]{64}$'),
  pixel_sha256 text check (pixel_sha256 ~ '^[a-f0-9]{64}$'),
  canonical_bytes bigint check (canonical_bytes between 1 and 10485760),
  width integer check (width between 1 and 8192),
  height integer check (height between 1 and 8192),
  preprocessor text not null default 'zils-image-rgb-png/v1',
  referenced boolean not null default false,
  finalize_token uuid, finalize_until timestamptz,
  cleanup_token uuid, cleanup_until timestamptz, cleaned_at timestamptz,
  created_at timestamptz not null default clock_timestamp(),
  expires_at timestamptz not null default clock_timestamp()+interval '24 hours',
  -- Reserve the signed-upload window before requesting a grant; later record its actual expiry.
  grant_expires_at timestamptz not null default clock_timestamp()+interval '2 hours',
  check ((purpose='prediction' and job_id is null) or (purpose='training' and job_id is not null)),
  check (width::bigint * height <= 16000000),
  check (state <> 'ready' or (canonical_sha256 is not null and pixel_sha256 is not null
    and canonical_bytes is not null and width is not null and height is not null))
);
create index on public.zils_image_assets(owner_id,created_at);
create index on public.zils_image_assets(job_id);
alter table public.zils_image_assets enable row level security;
revoke all on public.zils_image_assets from public,anon,authenticated;
grant select on public.zils_image_assets to authenticated;
create policy zils_image_owner_read on public.zils_image_assets for select to authenticated
  using ((select auth.uid())=owner_id);
grant all on public.zils_image_assets to service_role;
insert into storage.buckets(id,name,public,file_size_limit)
values ('zils-images','zils-images',false,10485760);
create policy zils_images_private on storage.objects as restrictive for all to anon,authenticated
  using (bucket_id <> 'zils-images') with check (bucket_id <> 'zils-images');

create function public.zils_image_ensure_account(p_owner uuid) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  if not zils_access_allowed(p_owner) then raise exception 'early access required'; end if;
  insert into zils_api_accounts(owner_id) values(p_owner) on conflict do nothing;
  if not (select enabled from zils_api_accounts where owner_id=p_owner) then
    raise exception 'account disabled'; end if;
end $$;

create function public.zils_image_live(a public.zils_image_assets) returns boolean
language sql security definer set search_path=public,pg_temp as $$
  select a.state in ('uploading','verifying','ready') and case when a.purpose='prediction'
    then a.expires_at > clock_timestamp() else exists(
      select 1 from fez_training_jobs j where j.id=a.job_id and j.owner_id=a.owner_id and
       case when j.status='failed' then false
            when j.status='completed' then j.updated_at+interval '30 days'>clock_timestamp()
            when j.status='uploading' then a.expires_at>clock_timestamp() else true end) end
$$;

create function public.zils_image_create(p_owner uuid,p_purpose text,p_job uuid,
  p_filename text,p_source_bytes bigint,p_source_sha256 text) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_image_assets; ident uuid:=gen_random_uuid(); job fez_training_jobs;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text,801));
  perform zils_image_ensure_account(p_owner);
  if p_purpose='training' then
    select * into job from fez_training_jobs where id=p_job and owner_id=p_owner for update;
    if not found or job.status<>'uploading' or job.created_at+interval '24 hours'<=clock_timestamp()
      or job.model_profile->>'id' is distinct from 'imajev-4b-v1' then return null; end if;
    if (select count(*) from zils_image_assets where job_id=p_job and zils_image_live(zils_image_assets)) >=1792
       or (select coalesce(sum(coalesce(canonical_bytes,10485760)),0) from zils_image_assets
           where job_id=p_job and zils_image_live(zils_image_assets))+10485760 > 1073741824
      then return jsonb_build_object('error','limited'); end if;
  elsif p_purpose='prediction' and p_job is null then
    if (select count(*) from zils_image_assets where owner_id=p_owner and purpose='prediction'
         and zils_image_live(zils_image_assets))>=20
      or (select coalesce(sum(coalesce(canonical_bytes,10485760)),0) from zils_image_assets
         where owner_id=p_owner and purpose='prediction' and zils_image_live(zils_image_assets))+10485760>268435456
      or (select count(*) from zils_image_assets where owner_id=p_owner and purpose='prediction'
         and created_at>clock_timestamp()-interval '24 hours')>=100
      then return jsonb_build_object('error','limited'); end if;
  else raise exception 'invalid image purpose'; end if;
  insert into zils_image_assets(id,owner_id,purpose,job_id,filename,source_bytes,source_sha256,source_path,canonical_path)
   values(ident,p_owner,p_purpose,p_job,p_filename,p_source_bytes,p_source_sha256,
    p_owner::text||'/'||ident::text||'/source',p_owner::text||'/'||ident::text||'/canonical.png') returning * into a;
  return to_jsonb(a);
end $$;

create function public.zils_image_grant(p_owner uuid,p_asset uuid,p_expires timestamptz) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  -- This operation records even a concurrent cancellation so late writes remain tracked.
  if p_expires>clock_timestamp()+interval '24 hours' then raise exception 'upload grant too long'; end if;
  update zils_image_assets set grant_expires_at=greatest(grant_expires_at,p_expires)
    where id=p_asset and owner_id=p_owner;
  if not found then raise exception 'asset unavailable'; end if;
end $$;

create function public.zils_image_get(p_owner uuid,p_asset uuid) returns jsonb
language sql security definer set search_path=public,pg_temp as $$
  select to_jsonb(a) from zils_image_assets a where id=p_asset and owner_id=p_owner and zils_image_live(a)
$$;

create function public.zils_image_claim_finalize(p_owner uuid,p_asset uuid) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_image_assets;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text,801));
  select * into a from zils_image_assets where id=p_asset and owner_id=p_owner for update;
  if not found or not zils_image_live(a) then return null; end if;
  if a.state='ready' then return to_jsonb(a); end if;
  if a.state='verifying' and a.finalize_until>clock_timestamp() then return null; end if;
  update zils_image_assets set state='verifying',finalize_token=gen_random_uuid(),
    finalize_until=clock_timestamp()+interval '20 minutes' where id=p_asset returning * into a;
  return to_jsonb(a);
end $$;

create function public.zils_image_finish(p_owner uuid,p_asset uuid,p_token uuid,
  p_sha256 text,p_pixels text,p_bytes bigint,p_width integer,p_height integer) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_image_assets; job fez_training_jobs;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text,801));
  select * into a from zils_image_assets where id=p_asset and owner_id=p_owner;
  if a.job_id is not null then
    select * into job from fez_training_jobs where id=a.job_id for update;
    if job.status<>'uploading' then return null; end if;
    if (select coalesce(sum(coalesce(canonical_bytes,10485760)),0) from zils_image_assets
        where job_id=a.job_id and id<>p_asset and zils_image_live(zils_image_assets))+p_bytes>1073741824
       then raise exception 'image byte budget exceeded'; end if;
  end if;
  update zils_image_assets set state='ready',canonical_sha256=p_sha256,pixel_sha256=p_pixels,
     canonical_bytes=p_bytes,width=p_width,height=p_height
    where id=p_asset and owner_id=p_owner and state='verifying' and finalize_token=p_token
      and finalize_until>clock_timestamp() and zils_image_live(zils_image_assets) returning * into a;
  if not found then return null; end if;
  return to_jsonb(a);
end $$;

create function public.zils_image_fail(p_owner uuid,p_asset uuid,p_token uuid) returns void
language sql security definer set search_path=public,pg_temp as $$
  update zils_image_assets set state='failed' where id=p_asset and owner_id=p_owner
   and state='verifying' and finalize_token=p_token
$$;

create function public.zils_image_delete(p_owner uuid,p_asset uuid) returns boolean
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_image_assets;
begin
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text,801));
  select * into a from zils_image_assets where id=p_asset and owner_id=p_owner for update;
  if not found or not zils_image_live(a) then return false; end if;
  if a.referenced or (a.job_id is not null and exists(select 1 from fez_training_jobs
    where id=a.job_id and status<>'uploading')) then raise exception 'image is in use'; end if;
  update zils_image_assets set state='deleted' where id=p_asset;
  return true;
end $$;

create function public.zils_image_cleanup_claim(p_limit integer default 100) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare result jsonb;
begin
  if p_limit not between 1 and 100 then raise exception 'invalid cleanup limit'; end if;
  with candidates as (
    select a.id from zils_image_assets a left join fez_training_jobs j on j.id=a.job_id
    where a.cleaned_at is null and (a.cleanup_until is null or a.cleanup_until<=clock_timestamp())
      and a.grant_expires_at<=clock_timestamp()
      and (a.finalize_until is null or a.finalize_until<=clock_timestamp())
      and (a.state in ('failed','deleted','expired') or
           (a.purpose='prediction' and a.expires_at<=clock_timestamp()) or
           (a.purpose='training' and ((j.status='uploading' and a.expires_at<=clock_timestamp())
              or (j.status in ('completed','failed') and j.updated_at+interval '30 days'<=clock_timestamp()))))
    order by a.created_at limit p_limit for update of a skip locked
  ), claimed as (
    update zils_image_assets set cleanup_token=gen_random_uuid(),cleanup_until=clock_timestamp()+interval '10 minutes',
      state=case when state in ('uploading','verifying','ready') then 'expired' else state end
    where id in (select id from candidates) returning *)
    select coalesce(jsonb_agg(to_jsonb(claimed)),'[]'::jsonb) into result from claimed;
  return result;
end $$;

create function public.zils_image_cleanup_finish(p_asset uuid,p_token uuid) returns boolean
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  update zils_image_assets set cleaned_at=clock_timestamp(),cleanup_until=null
   where id=p_asset and cleanup_token=p_token and cleanup_until>clock_timestamp()
     and grant_expires_at<=clock_timestamp();
  return found;
end $$;

-- Service-role RPCs only; owner RLS grants read access to metadata, never object access.
do $$ declare f record; begin
  for f in select oid::regprocedure as signature from pg_proc
    where pronamespace='public'::regnamespace and proname like 'zils_image_%' loop
    execute format('revoke all on function %s from public,anon,authenticated',f.signature);
    execute format('grant execute on function %s to service_role',f.signature);
  end loop;
end $$;
commit;
