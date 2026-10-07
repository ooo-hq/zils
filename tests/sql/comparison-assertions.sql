\set ON_ERROR_STOP on
set role service_role;
select id as comparison_job from public.fez_create_training_job('11111111-1111-4111-8111-111111111111','comparison-job','{"min_accuracy":0.8,"min_brier_improvement":0.01}') \gset
insert into public.zils_training_comparisons(job_id,owner_id,consent_version,model_id,checkpoint_sha256,jev_model)
values (:'comparison_job','11111111-1111-4111-8111-111111111111','typesafe-evaluation-v1','model',repeat('a',64),'jev-1.13.0');
-- Duplicate create must preserve the frozen consent and model.
insert into public.zils_training_comparisons(job_id,owner_id,consent_version,model_id,checkpoint_sha256,jev_model)
values (:'comparison_job','11111111-1111-4111-8111-111111111111','typesafe-evaluation-v1','changed',repeat('b',64),'jev-1.13.0')
on conflict(job_id) do nothing;
do $$ begin
  if (select count(*) from public.zils_training_comparisons where model_id='model') <> 1 then raise exception 'frozen model overwritten'; end if;
end $$;
update public.zils_training_comparisons set status='queued' where job_id=:'comparison_job';
update public.zils_training_comparisons set status='running',lease_token=gen_random_uuid(),lease_until=now()+interval '1 hour' where job_id=:'comparison_job' and status='queued';
do $$ begin
  update public.zils_training_comparisons set lease_token=gen_random_uuid() where status='queued';
  if found then raise exception 'paid comparison claimed twice'; end if;
end $$;
reset role;
set role authenticated;
do $$ begin
  begin perform 1 from public.zils_training_comparisons; raise exception 'direct comparison read allowed';
  exception when insufficient_privilege then null; end;
  begin update public.zils_training_comparisons set status='completed'; raise exception 'direct comparison mutation allowed';
  exception when insufficient_privilege then null; end;
end $$;
reset role;
set role anon;
do $$ begin
  begin perform 1 from public.zils_training_comparisons; raise exception 'anonymous comparison read allowed';
  exception when insufficient_privilege then null; end;
end $$;
reset role;
