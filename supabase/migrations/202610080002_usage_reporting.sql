-- Account usage reporting. Existing balances and price calculations are unchanged.
begin;
alter table public.zils_api_usage
  add column model_id text check(length(model_id) between 1 and 256),
  add column model_name text check(length(model_name) between 1 and 256);
create index on public.zils_api_usage(owner_id,billing_mode,created_at);
create index on public.zils_billing_reservations(owner_id,mode,created_at);

-- Optional arguments keep existing gateway/worker callers compatible during rollout.
drop function public.zils_api_admit(uuid,uuid,uuid,bigint,bigint);
create function public.zils_api_admit(p_owner uuid,p_key uuid,p_request uuid,p_tokens bigint,
  p_billable_tokens bigint default null, p_model_id text default null, p_model_name text default null) returns text
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_api_accounts; w zils_api_windows; current_second timestamptz;
begin
  if p_tokens < 0 or p_tokens is null then raise exception 'invalid reservation'; end if;
  select * into a from zils_api_accounts where owner_id=p_owner for update;
  if not found or not a.enabled then return 'disabled'; end if;
  if p_key is not null and not exists(select 1 from zils_api_keys where id=p_key and owner_id=p_owner and revoked_at is null) then return 'disabled'; end if;
  if exists(select 1 from zils_api_usage where request_id=p_request) then return 'duplicate'; end if;
  current_second:=date_trunc('second',clock_timestamp());
  insert into zils_api_windows values(p_owner,current_second,0,0) on conflict do nothing;
  select * into w from zils_api_windows where owner_id=p_owner;
  if w.second<>current_second then w.requests:=0; w.tokens:=0; end if;
  if (a.requests_per_second is not null and w.requests+1>a.requests_per_second)
    or (a.tokens_per_second is not null and w.tokens+p_tokens>a.tokens_per_second) then return 'limited'; end if;
  update zils_api_windows set second=current_second,requests=w.requests+1,tokens=w.tokens+p_tokens where owner_id=p_owner;
  insert into zils_api_usage(request_id,owner_id,reserved_tokens,billable_tokens,model_id,model_name) values(p_request,p_owner,p_tokens,p_billable_tokens,p_model_id,p_model_name);
  return 'allowed';
exception when sqlstate 'P0402' then return 'insufficient_credit';
  when sqlstate 'P0422' then return 'billing_meter_unavailable';
end $$;

create function public.zils_billing_usage_summary(p_owner uuid,p_mode text) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare report jsonb; until_at timestamptz:=now(); since_at timestamptz:=now()-interval '30 days';
begin
  perform zils_billing_require_mode(p_mode);
  with usage as (
    select * from zils_api_usage where owner_id=p_owner and billing_mode=p_mode
      and created_at>=since_at and created_at<until_at
  ), charges as (
    select * from zils_billing_ledger where owner_id=p_owner and mode=p_mode
      and kind in ('inference','training') and created_at>=since_at and created_at<until_at
  ), model_rows as (
    select model_id,max(model_name) as model_name,
      count(*) filter(where status='completed') as calls,
      count(*) filter(where status='failed') as failed_calls,
      count(*) filter(where status='started') as active_calls,
      coalesce(sum(billable_tokens) filter(where status='completed'),0) as input_tokens,
      0::numeric as spend_nanos
    from usage group by model_id
    union all
    select u.model_id,max(u.model_name),0,0,0,0,-sum(c.amount_nanos)
    from charges c left join zils_api_usage u on u.request_id::text=c.reference
      and u.owner_id=p_owner and u.billing_mode=p_mode
    where c.kind='inference' group by u.model_id
  ), models as (
    select model_id,max(model_name) as model_name,sum(calls)::text as calls,
      sum(failed_calls)::text as failed_calls,sum(active_calls)::text as active_calls,
      sum(input_tokens)::text as input_tokens,sum(spend_nanos)::text as spend_nanos
    from model_rows group by model_id
  ), training as (
    select j.status from zils_billing_reservations r join fez_training_jobs j on j.id=r.id and j.owner_id=r.owner_id
    where r.owner_id=p_owner and r.mode=p_mode and r.kind='training'
      and r.created_at>=since_at and r.created_at<until_at
  )
  select jsonb_build_object(
    'since',since_at,'until',until_at,
    'calls',(select count(*)::text from usage where status='completed'),
    'failed_calls',(select count(*)::text from usage where status='failed'),
    'active_calls',(select count(*)::text from usage where status='started'),
    'input_tokens',(select coalesce(sum(billable_tokens),0)::text from usage where status='completed'),
    'training_runs',(select count(*)::text from training where status='completed'),
    'failed_training_runs',(select count(*)::text from training where status='failed'),
    'active_training_runs',(select count(*)::text from training where status not in ('completed','failed')),
    'inference_spend_nanos',(select (-coalesce(sum(amount_nanos),0))::text from charges where kind='inference'),
    'training_spend_nanos',(select (-coalesce(sum(amount_nanos),0))::text from charges where kind='training'),
    'models',(select coalesce(jsonb_agg(m order by m.model_id nulls last),'[]'::jsonb) from models m)
  ) into report;
  return report;
end $$;

create or replace function public.zils_billing_summary(p_owner uuid,p_mode text) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_billing_accounts; transactions jsonb; payments jsonb;
begin
  if p_mode='off' and (select mode from zils_billing_settings where singleton)='off' then
    return jsonb_build_object('mode','off','currency','usd','balance_nanos','0',
      'reserved_nanos','0','available_nanos','0','free_training_runs',0,
      'topup_amounts_cents',jsonb_build_array(500,2000,5000,10000),
      'transactions','[]'::jsonb,'payments','[]'::jsonb);
  end if;
  perform zils_billing_require_mode(p_mode);
  insert into zils_billing_accounts(owner_id,mode) values(p_owner,p_mode) on conflict do nothing;
  select * into a from zils_billing_accounts where owner_id=p_owner and mode=p_mode;
  select coalesce(jsonb_agg(t),'[]'::jsonb) into transactions from
    (select id,kind,amount_nanos::text,created_at,reference from zils_billing_ledger
     where owner_id=p_owner and mode=p_mode order by created_at desc,id desc limit 100) t;
  select coalesce(jsonb_agg(t),'[]'::jsonb) into payments from
    (select id,amount_cents,status,created_at,receipt_url from zils_billing_purchases
     where owner_id=p_owner and mode=p_mode order by created_at desc,id desc limit 100) t;
  return jsonb_build_object('mode',p_mode,'currency','usd','balance_nanos',a.balance_nanos::text,
    'reserved_nanos',a.reserved_nanos::text,'available_nanos',(a.balance_nanos-a.reserved_nanos)::text,
    'free_training_runs',a.free_training_runs,'topup_amounts_cents',jsonb_build_array(500,2000,5000,10000),
    'transactions',transactions,'payments',payments,
    'usage',zils_billing_usage_summary(p_owner,p_mode));
end $$;

revoke all on function public.zils_api_admit(uuid,uuid,uuid,bigint,bigint,text,text) from public,anon,authenticated;
grant execute on function public.zils_api_admit(uuid,uuid,uuid,bigint,bigint,text,text) to service_role;
revoke all on function public.zils_billing_usage_summary(uuid,text) from public,anon,authenticated;
grant execute on function public.zils_billing_usage_summary(uuid,text) to service_role;
notify pgrst, 'reload schema';
commit;
