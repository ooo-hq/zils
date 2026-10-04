\set ON_ERROR_STOP on
set role service_role;
select public.zils_api_create_key('{"id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa","owner_id":"11111111-1111-4111-8111-111111111111","name":"test","prefix":"zils_sk_a","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}');
select public.zils_api_create_key('{"id":"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb","owner_id":"11111111-1111-4111-8111-111111111111","name":"second","prefix":"zils_sk_b","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}');
update public.zils_api_accounts set requests_per_second=2,tokens_per_second=100;
begin;
-- Freeze the window in this transaction for deterministic accounting assertions.
do $$ declare owner uuid := '11111111-1111-4111-8111-111111111111'; first_key uuid := 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'; second_key uuid := 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'; req uuid := gen_random_uuid();
begin
  if zils_api_admit(owner,first_key,req,60)<>'allowed' then raise exception 'first request rejected'; end if;
  if zils_api_admit(owner,first_key,req,60)<>'duplicate' then raise exception 'duplicate admitted'; end if;
  if zils_api_admit(owner,second_key,gen_random_uuid(),41)<>'limited' then raise exception 'keys bypass token budget'; end if;
  if zils_api_admit(owner,second_key,gen_random_uuid(),40)<>'allowed' then raise exception 'exact token budget rejected'; end if;
  if zils_api_admit(owner,first_key,gen_random_uuid(),0)<>'limited' then raise exception 'request budget bypassed'; end if;
  perform zils_api_finish_usage(req,50,'completed');
  perform zils_api_finish_usage(req,99,'completed');
  if (select input_tokens from zils_api_usage where request_id=req)<>50 then raise exception 'duplicate usage update'; end if;
  update zils_api_keys set revoked_at=now() where id=first_key;
  if exists(select 1 from zils_api_auth(first_key)) then raise exception 'revoked credential authenticated'; end if;
  if zils_api_admit(owner,first_key,gen_random_uuid(),1)<>'disabled' then raise exception 'revocation raced admission'; end if;
  if zils_api_admit('22222222-2222-4222-8222-222222222222',second_key,gen_random_uuid(),1)<>'disabled' then raise exception 'cross tenant admission'; end if;
  update zils_api_accounts set enabled=false where owner_id=owner;
  if exists(select 1 from zils_api_auth(second_key)) then raise exception 'disabled account authenticated'; end if;
end $$;
rollback;
reset role;
set role authenticated;
do $$ begin
  begin perform * from public.zils_api_keys; raise exception 'customer read key digests'; exception when insufficient_privilege then null; end;
  begin perform public.zils_api_auth('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'); raise exception 'customer called privileged auth'; exception when insufficient_privilege then null; end;
  begin update public.zils_api_accounts set enabled=true; raise exception 'customer changed budgets'; exception when insufficient_privilege then null; end;
end $$;
reset role;
set role anon;
do $$ begin
  begin perform * from public.zils_api_usage; raise exception 'anonymous usage access'; exception when insufficient_privilege then null; end;
end $$;
reset role;
