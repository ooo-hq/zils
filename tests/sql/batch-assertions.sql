\set ON_ERROR_STOP on
set role service_role;
update public.zils_api_accounts set requests_per_second=null,tokens_per_second=null;
select id as batch from zils_api_batch_create('11111111-1111-4111-8111-111111111111','batch-test') \gset
select set_config('test.batch',:'batch',false);
do $$ begin
  if (zils_api_batch_create('11111111-1111-4111-8111-111111111111','batch-test')).id<>current_setting('test.batch')::uuid then raise exception 'duplicate create'; end if;
  if (zils_api_batch_action('22222222-2222-4222-8222-222222222222',current_setting('test.batch')::uuid,'cancel')).id is not null then raise exception 'cross tenant cancel'; end if;
end $$;
select zils_api_batch_action('11111111-1111-4111-8111-111111111111',:'batch','submit','{"alias":{"id":"r1","fingerprint":"a"}}');
select zils_api_batch_action('11111111-1111-4111-8111-111111111111',:'batch','submit','{"alias":{"id":"r2","fingerprint":"b"}}');
select lease_token as lease from zils_api_batch_claim() \gset
select set_config('test.lease',:'lease',false);
do $$ begin
  if (select catalog#>>'{alias,id}' from zils_api_batches where id=current_setting('test.batch')::uuid)<>'r1' then raise exception 'submit changed pinned version'; end if;
  if (zils_api_batch_claim()).id is not null then raise exception 'concurrent lease accepted'; end if;
end $$;
select zils_api_batch_work(:'batch',:'lease','load','[{"line":1,"custom_id":"a","body":"{}","frozen":{},"result":null},{"line":2,"custom_id":"b","body":null,"frozen":null,"result":{"custom_id":"b","error":{"status":422}}}]');
-- Crash during validation discards partial preparation and starts from immutable input.
update zils_api_batches set lease_until=now()-interval '1 second' where id=:'batch';
select lease_token as recovered from zils_api_batch_claim() \gset
do $$ begin
  if exists(select 1 from zils_api_batch_items where batch_id=current_setting('test.batch')::uuid) then raise exception 'partial preparation not reset'; end if;
  begin perform zils_api_batch_work(current_setting('test.batch')::uuid,current_setting('test.lease')::uuid,'ready','{"total":2}'); raise exception 'old lease accepted';
  exception when raise_exception then if sqlerrm<>'batch lease expired' then raise; end if; end;
end $$;
select zils_api_batch_work(:'batch',:'recovered','load','[{"line":1,"custom_id":"a","body":"{}","frozen":{},"result":null},{"line":2,"custom_id":"b","body":null,"frozen":null,"result":{"custom_id":"b","error":{"status":422}}}]');
select zils_api_batch_work(:'batch',:'recovered','ready','{"total":2}');
select zils_api_batch_work(:'batch',:'recovered','release');
select lease_token as active from zils_api_batch_claim() \gset
select gen_random_uuid() as request \gset
select zils_api_admit('11111111-1111-4111-8111-111111111111',null,:'request',50);
select zils_api_batch_work(:'batch',:'active','result',jsonb_build_object('line',1,'request_id',:'request','input_tokens',40,'result','{"custom_id":"a","response":{"model":"r1"}}'::jsonb));
select set_config('test.request',:'request',false);
do $$ begin
  if not exists(select 1 from zils_api_batches where id=current_setting('test.batch')::uuid and status='completed' and completed=1 and failed=1 and total=2) then raise exception 'mixed result counts wrong'; end if;
  if (select input_tokens from zils_api_usage where request_id=current_setting('test.request')::uuid)<>40 then raise exception 'usage not atomically committed'; end if;
end $$;
-- Cancellation fences already-running inference; completed results remain retrievable.
select id as cancelled from zils_api_batch_create('11111111-1111-4111-8111-111111111111','cancel-test') \gset
select zils_api_batch_action('11111111-1111-4111-8111-111111111111',:'cancelled','submit','{}');
select lease_token as cancelled_lease from zils_api_batch_claim() \gset
select zils_api_batch_action('11111111-1111-4111-8111-111111111111',:'cancelled','cancel');
select set_config('test.cancelled',:'cancelled',false),set_config('test.cancelled_lease',:'cancelled_lease',false);
do $$ begin
  begin perform zils_api_batch_work(current_setting('test.cancelled')::uuid,current_setting('test.cancelled_lease')::uuid,'load','[]'); raise exception 'cancelled work committed';
  exception when raise_exception then if sqlerrm<>'batch lease expired' then raise; end if; end;
  begin perform zils_api_batch_purge(current_setting('test.batch')::uuid); raise exception 'early deletion';
  exception when raise_exception then if sqlerrm<>'retention period not reached' then raise; end if; end;
end $$;
update zils_api_batches set finished_at=now()-interval '8 days' where id=:'batch';
select zils_api_batch_purge(:'batch');
do $$ begin
  if exists(select 1 from zils_api_batch_items where batch_id=current_setting('test.batch')::uuid) then raise exception 'private results survived purge'; end if;
end $$;
select id as expired from zils_api_batch_create('11111111-1111-4111-8111-111111111111','expiry-test') \gset
update zils_api_batches set deadline=now()-interval '1 second' where id=:'expired';
select zils_api_batch_claim();
do $$ begin
  if exists(select 1 from zils_api_batches where idempotency_key='expiry-test' and status<>'expired') then raise exception 'upload deadline not enforced'; end if;
end $$;
reset role;
insert into storage.objects(bucket_id) values ('zils-api-batches');
set role authenticated;
do $$ begin
  if exists(select 1 from storage.objects where bucket_id='zils-api-batches') then raise exception 'client sees bulk inputs'; end if;
  begin perform * from zils_api_batch_items; raise exception 'client reads private records'; exception when insufficient_privilege then null; end;
  begin perform zils_api_batch_claim(); raise exception 'client claims work'; exception when insufficient_privilege then null; end;
end $$;
reset role;
-- Creation budgets include retained terminal jobs, and repeated creates are idempotent.
set role service_role;
update zils_api_accounts set max_active_batches=1,max_batch_storage_bytes=26214400 where owner_id='22222222-2222-4222-8222-222222222222';
-- Create this account if running only the SQL assertions (the Python flow follows).
insert into zils_api_accounts(owner_id,max_active_batches,max_batch_storage_bytes) values ('22222222-2222-4222-8222-222222222222',1,26214400) on conflict do nothing;
select id as capped from zils_api_batch_create('22222222-2222-4222-8222-222222222222','capacity-one') \gset
select set_config('test.capped',:'capped',false);
do $$ begin
  if (zils_api_batch_create('22222222-2222-4222-8222-222222222222','capacity-one')).id<>current_setting('test.capped')::uuid then raise exception 'idempotent create charged twice'; end if;
  if (zils_api_batch_create('22222222-2222-4222-8222-222222222222','capacity-two')).id is not null then raise exception 'active capacity ignored'; end if;
  perform zils_api_batch_action('22222222-2222-4222-8222-222222222222',current_setting('test.capped')::uuid,'cancel');
  if (zils_api_batch_create('22222222-2222-4222-8222-222222222222','capacity-two')).id is not null then raise exception 'retained capacity ignored'; end if;
end $$;
update zils_api_batches set finished_at=now()-interval '8 days' where id=:'capped';
select zils_api_batch_purge(:'capped');
do $$ begin
  if (zils_api_batch_create('22222222-2222-4222-8222-222222222222','capacity-two')).id is null then raise exception 'purge did not release capacity'; end if;
end $$;
reset role;
