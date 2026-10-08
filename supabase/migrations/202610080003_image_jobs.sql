begin;
alter table public.fez_training_jobs add column image_intake jsonb;
-- Server-owned, exact profiles. Clients cannot add models or mutate jobs after creation.
create table public.zils_model_profiles(id text primary key, profile jsonb not null);
insert into public.zils_model_profiles values
('kev-0.8b-v1', '{"id": "kev-0.8b-v1", "name": "Kev 0.8B", "base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"}'::jsonb),
('jevk5-4b-v0.3', '{"id": "jevk5-4b-v0.3", "name": "JevK5 4B", "base": "alibiserikbay/JevK5", "base_revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d"}'::jsonb),
('imajev-4b-v1', '{"id": "imajev-4b-v1", "name": "Imajev 4B", "base": "Qwen/Qwen3.5-4B", "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "adapter": "mohit67890/imajev-4b", "adapter_revision": "f8d8234cebc6c99065c07731e59716dc0a6e27ab", "starting_adapter_sha256": "88c2c44361e0c469352495abcfee789ff73a4deae0811168d9402cfc2b6e749c", "starting_head_sha256": "52ceafd7d824bf3ea5ce55276b48cc08ba9dd6b2c98a55d9bd6a72ed1643a427", "runtime_revision": "ccf586d43d2a580319b6535c893668904d909eb9", "preprocessor": "zils-image-rgb-png/v1", "prompt": "imajev-readout/v1", "option_order": "declared_then_unknown", "min_pixels": 65536, "max_pixels": 400000, "max_input_tokens": 4096, "recipe": "imajev-lora64-readout-adamw/v1", "calibration": "full-native-temperature/v1"}'::jsonb);
alter table public.zils_model_profiles enable row level security;
revoke all on public.zils_model_profiles from public,anon,authenticated,service_role;
grant select on public.zils_model_profiles to service_role;

create function public.zils_job_profile_guard() returns trigger
language plpgsql set search_path='' as $$
begin
  if tg_op='UPDATE' and old.image_intake is not null and old.image_intake is distinct from new.image_intake then
    raise exception 'image intake is immutable';
  end if;
  if tg_op='UPDATE' and old.model_profile is distinct from new.model_profile then
    if old.model_profile is not null or old.status<>'uploading' or old.manifest is not null then
      raise exception 'job model profile is immutable';
    end if;
  end if;
  if new.model_profile is not null and not exists(select 1 from public.zils_model_profiles
    where id=new.model_profile->>'id' and profile=new.model_profile) then
    raise exception 'model profile is not registered';
  end if;
  if new.model_profile is not null and new.manifest is not null and
    new.manifest->'model' is distinct from new.model_profile then
    raise exception 'manifest differs from frozen job profile';
  end if;
  return new;
end $$;
create trigger zils_job_profile_guard before insert or update on public.fez_training_jobs
for each row execute function public.zils_job_profile_guard();

create function public.zils_create_profile_job(p_owner uuid,p_name text,p_acceptance jsonb,p_model jsonb)
returns public.fez_training_jobs language plpgsql set search_path='' as $$
declare item public.fez_training_jobs;
begin
  if p_model is null or not exists(select 1 from public.zils_model_profiles
    where id=p_model->>'id' and profile=p_model) then raise exception 'model profile is not registered'; end if;
  item:=public.fez_create_training_job(p_owner,p_name,p_acceptance);
  update public.fez_training_jobs set model_profile=p_model where id=item.id returning * into item;
  return item;
end $$;
revoke execute on function public.zils_create_profile_job(uuid,text,jsonb,jsonb),public.zils_job_profile_guard() from public,anon,authenticated;
grant execute on function public.zils_create_profile_job(uuid,text,jsonb,jsonb),public.zils_job_profile_guard() to service_role;
create function public.zils_submit_image_job(p_owner uuid,p_job uuid,p_assets jsonb)
returns public.fez_training_jobs language plpgsql set search_path='' as $$
declare item public.fez_training_jobs; ids uuid[];
begin
  select * into item from public.fez_training_jobs where id=p_job and owner_id=p_owner for update;
  if not found or item.model_profile->>'id' is distinct from 'imajev-4b-v1' then raise exception 'image job unavailable'; end if;
  if item.status<>'uploading' then return item; end if;
  if item.created_at+interval '24 hours'<=clock_timestamp() then raise exception 'image draft expired'; end if;
  if p_assets is null or jsonb_typeof(p_assets)<>'array' or jsonb_array_length(p_assets) not between 1 and 1792 then
    raise exception 'invalid image count';
  end if;
  select array_agg(distinct value::uuid) into ids from jsonb_array_elements_text(p_assets);
  if cardinality(ids)<>jsonb_array_length(p_assets) then raise exception 'duplicate asset IDs'; end if;
  perform 1 from public.zils_image_assets where id=any(ids) order by id for update;
  if (select count(*) from public.zils_image_assets where id=any(ids) and job_id=p_job and owner_id=p_owner
      and purpose='training' and state='ready' and public.zils_image_live(zils_image_assets))<>cardinality(ids) then
    raise exception 'images must be complete and live';
  end if;
  update public.zils_image_assets set referenced=true where id=any(ids);
  update public.fez_training_jobs set status='validating',updated_at=clock_timestamp() where id=p_job returning * into item;
  return item;
end $$;
revoke execute on function public.zils_submit_image_job(uuid,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.zils_submit_image_job(uuid,uuid,jsonb) to service_role;
create function public.zils_create_image_job(p_owner uuid,p_name text,p_acceptance jsonb,p_model jsonb,p_intake jsonb)
returns public.fez_training_jobs language plpgsql set search_path='' as $$
declare item public.fez_training_jobs;
begin
  if jsonb_typeof(p_intake) is distinct from 'object' or not (p_intake ?& array['version','seed','snapshot_sha256'])
     or p_model->>'id' is distinct from 'imajev-4b-v1' or p_intake->>'version' is distinct from 'zils-image-intake/v1'
     or p_intake->>'snapshot_sha256' !~ '^[a-f0-9]{64}$' or length(p_intake->>'seed') not between 1 and 100 then
    raise exception 'invalid image intake';
  end if;
  item:=public.zils_create_profile_job(p_owner,p_name,p_acceptance,p_model);
  update public.fez_training_jobs set image_intake=p_intake where id=item.id returning * into item;
  return item;
end $$;
revoke execute on function public.zils_create_image_job(uuid,text,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.zils_create_image_job(uuid,text,jsonb,jsonb,jsonb) to service_role;
commit;
