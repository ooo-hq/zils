-- Select work only for installed profiles with current capacity. No image head-of-line blocking.
create function public.zils_claim_profile_processing(p_stage text, p_profiles text[])
returns public.fez_training_jobs language plpgsql set search_path = '' as $$
declare item public.fez_training_jobs;
begin
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
-- Old processors remain text-only during a rolling migration.
create or replace function public.fez_claim_processing(p_stage text)
returns public.fez_training_jobs language sql set search_path = '' as $$
  select public.zils_claim_profile_processing(p_stage,array['kev-0.8b-v1','jevk5-4b-v0.3']);
$$;
revoke all on function public.zils_claim_profile_processing(text,text[]) from public,anon,authenticated;
grant execute on function public.zils_claim_profile_processing(text,text[]) to service_role;
