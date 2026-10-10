\set ON_ERROR_STOP on
insert into auth.users values ('11111111-1111-4111-8111-111111111111'),('22222222-2222-4222-8222-222222222222');
insert into public.fez_training_workers values ('worker-a',1,true),('worker-b',2,true),('worker-disabled',3,false);
insert into storage.objects(bucket_id) values ('fez-training-data'),('fez-training-models'),('other-bucket');
set role service_role;
select id as job from public.fez_create_training_job('11111111-1111-4111-8111-111111111111','test-job','{"min_accuracy":0.8,"min_brier_improvement":0.01}') \gset
select id as other from public.fez_create_training_job('22222222-2222-4222-8222-222222222222','other-job','{"min_accuracy":0.8,"min_brier_improvement":0.01}') \gset
update public.fez_training_jobs set jev_comparison='{"status":"completed","accuracy":0.75}' where id=:'job';
reset role;
set role authenticated;
select set_config('request.jwt.claim.sub','11111111-1111-4111-8111-111111111111',false);
do $$ begin
  if (select count(*) from public.fez_training_jobs)<>1 then raise exception 'tenant isolation failed'; end if;
  if (select jev_comparison->>'accuracy' from public.fez_training_jobs) <> '0.75' then raise exception 'owner comparison read failed'; end if;
  begin
    update public.fez_training_jobs set jev_comparison='{"status":"completed","accuracy":1}';
    raise exception 'customer forged Jev results';
  exception when insufficient_privilege then null; end;
  if (select count(*) from storage.objects)<>1 then raise exception 'private storage restriction failed'; end if;
  begin
    perform public.fez_claim_training('worker-a');
    raise exception 'authenticated could claim a worker job';
  exception when insufficient_privilege then null; end;
  begin
    update public.fez_training_jobs set status='completed';
    raise exception 'customer changed a job state directly';
  exception when insufficient_privilege then null; end;
end $$;
reset role;
set role anon;
do $$ begin
  if (select count(*) from storage.objects)<>0 then raise exception 'anonymous storage access'; end if;
  begin perform public.fez_claim_processing('validating'); raise exception 'anonymous processor access';
  exception when insufficient_privilege then null; end;
end $$;
reset role;
set role service_role;
update public.fez_training_jobs set status='validating' where id=:'job';
select lease_token as prepare_token from public.fez_claim_processing('validating') \gset
-- No second processor may acquire a live lease.
do $$ begin
  if (public.fez_claim_processing('validating')).id is not null then raise exception 'double processing claim'; end if;
end $$;
select public.fez_finish_processing(:'job',:'prepare_token','awaiting_approval','{"job_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}');
select public.fez_approve_training_job(:'job',array['worker-a','worker-b']);
select public.fez_worker_nonce('worker-a','aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa');
do $$ begin
  begin perform public.fez_worker_nonce('worker-a','aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'); raise exception 'replayed nonce';
  exception when unique_violation then null; end;
  begin perform public.fez_worker_nonce('worker-disabled','bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'); raise exception 'disabled worker accepted';
  exception when raise_exception then if sqlerrm <> 'worker is not enabled' then raise; end if; end;
end $$;
select public.fez_claim_training('worker-a')->>'lease_token' as token_a \gset
select public.fez_claim_training('worker-b')->>'lease_token' as token_b \gset
select (public.fez_claim_training('worker-a')->>'lease_token' = :'token_a') as repeated_claim_same \gset
\if :repeated_claim_same
\else
  \quit 1
\endif
select public.fez_renew_training(:'job','worker-a',:'token_a');
-- Lease expiry changes the token, blocking stale writes.
update public.fez_training_assignments set lease_until=now()-interval '1 second' where job_id=:'job' and hotkey='worker-a';
select public.fez_claim_training('worker-a')->>'lease_token' as fresh_a \gset
select set_config('test.job',:'job',false),set_config('test.old_token',:'token_a',false);
do $$ begin
  begin perform public.fez_submit_training(current_setting('test.job')::uuid,'worker-a',current_setting('test.old_token')::uuid,repeat('a',64));
    raise exception 'stale lease submitted';
  exception when raise_exception then if sqlerrm<>'assignment lease expired' then raise; end if; end;
end $$;
select public.fez_submit_training(:'job','worker-a',:'fresh_a',repeat('a',64));
select public.fez_submit_training(:'job','worker-a',:'fresh_a',repeat('a',64));
-- A lost HTTP response followed by fail cannot discard a submitted checkpoint.
select public.fez_fail_training(:'job','worker-a',:'fresh_a');
select public.fez_submit_training(:'job','worker-b',:'token_b',repeat('b',64));
select lease_token as evaluate_token from public.fez_claim_processing('evaluating') \gset
select public.fez_finish_processing(:'job',:'evaluate_token','completed','{"result":{"delivery":{"status":"accepted"}}}');
do $$ begin
  if (select count(*) from public.fez_training_jobs where status='completed')<>1 then raise exception 'completion failed'; end if;
  if (public.fez_claim_training('worker-a')) is not null then raise exception 'completed job reclaimed'; end if;
end $$;
-- Check bounded retries and final failed-assignment readiness.
update public.fez_training_jobs set status='awaiting_approval' where id=:'other';
select public.fez_approve_training_job(:'other',array['worker-a']);
select public.fez_claim_training('worker-a')->>'lease_token' as retry_token \gset
select public.fez_fail_training(:'other','worker-a',:'retry_token');
select public.fez_claim_training('worker-a')->>'lease_token' as retry_token \gset
select public.fez_fail_training(:'other','worker-a',:'retry_token');
select public.fez_claim_training('worker-a')->>'lease_token' as retry_token \gset
select public.fez_fail_training(:'other','worker-a',:'retry_token');
do $$ begin
  if public.fez_claim_training('worker-a') is not null then raise exception 'unbounded attempts'; end if;
  if (public.fez_claim_processing('evaluating')).id is null then raise exception 'failed job never evaluated'; end if;
end $$;
reset role;
