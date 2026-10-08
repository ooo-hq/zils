-- Apply once. Existing access is unchanged until the operator enables billing.
-- Use a separate database and runtime for test mode. Never enable test on production.
begin;
create table public.zils_billing_settings (
  singleton boolean primary key default true check(singleton),
  mode text not null default 'off' check(mode in ('off','test','live'))
);
insert into public.zils_billing_settings default values;
create table public.zils_billing_accounts (
  owner_id uuid not null references auth.users(id),
  mode text not null check(mode in ('test','live')),
  balance_nanos bigint not null default 0,
  reserved_nanos bigint not null default 0 check(reserved_nanos >= 0),
  free_training_runs integer not null default 0 check(free_training_runs between 0 and 1),
  first_purchase_id uuid,
  bonus_eligible boolean not null default false,
  primary key(owner_id,mode)
);
create table public.zils_billing_purchases (
  id uuid primary key,
  owner_id uuid not null references auth.users(id),
  mode text not null check(mode in ('test','live')),
  amount_cents integer not null check(amount_cents in (500,2000,5000,10000)),
  status text not null default 'pending' check(status in ('pending','paid','refunded','expired')),
  session_id text unique,
  checkout_url text,
  payment_id text unique,
  refunded_cents integer not null default 0 check(refunded_cents >= 0 and refunded_cents <= amount_cents),
  disputed boolean not null default false,
  dispute_updated bigint not null default 0,
  receipt_url text,
  created_at timestamptz not null default clock_timestamp(),
  credited_at timestamptz
);
create index on public.zils_billing_purchases(owner_id,mode,created_at desc);
create table public.zils_billing_events (
  mode text not null,
  event_id text not null,
  created_at timestamptz not null default clock_timestamp(),
  primary key(mode,event_id)
);
create table public.zils_billing_ledger (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  mode text not null,
  kind text not null check(kind in ('topup','inference','training','refund')),
  amount_nanos bigint not null,
  reference text not null,
  created_at timestamptz not null default clock_timestamp(),
  foreign key(owner_id,mode) references public.zils_billing_accounts(owner_id,mode),
  unique(mode,kind,reference)
);
create index on public.zils_billing_ledger(owner_id,mode,created_at desc);
create table public.zils_billing_reservations (
  id uuid primary key,
  owner_id uuid not null,
  mode text not null,
  kind text not null check(kind in ('inference','training')),
  amount_nanos bigint not null check(amount_nanos >= 0),
  bonus boolean not null default false,
  status text not null default 'reserved' check(status in ('reserved','settled','released')),
  created_at timestamptz not null default clock_timestamp(),
  foreign key(owner_id,mode) references public.zils_billing_accounts(owner_id,mode)
);
create index on public.zils_billing_reservations(owner_id,mode,status);

create function public.zils_billing_require_mode(p_mode text) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  if p_mode not in ('test','live') or p_mode is null
    or p_mode is distinct from (select mode from zils_billing_settings where singleton) then
    raise exception 'billing mode unavailable';
  end if;
end $$;

create function public.zils_billing_summary(p_owner uuid,p_mode text) returns jsonb
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
    'transactions',transactions,'payments',payments);
end $$;

create function public.zils_billing_checkout(p_owner uuid,p_mode text,p_purchase uuid,p_amount integer)
returns public.zils_billing_purchases language plpgsql security definer set search_path=public,pg_temp as $$
declare p zils_billing_purchases;
begin
  perform zils_billing_require_mode(p_mode);
  perform pg_advisory_xact_lock(hashtextextended('billing-checkout:'||p_owner::text,0));
  select * into p from zils_billing_purchases where id=p_purchase;
  if found then
    if p.owner_id<>p_owner or p.mode<>p_mode or p.amount_cents<>p_amount then raise exception 'checkout conflict'; end if;
    return p;
  end if;
  if (select count(*) from zils_billing_purchases where owner_id=p_owner and mode=p_mode
      and created_at>clock_timestamp()-interval '10 minutes')>=10 then raise exception 'checkout limit'; end if;
  insert into zils_billing_purchases(id,owner_id,mode,amount_cents)
    values(p_purchase,p_owner,p_mode,p_amount) returning * into p;
  return p;
end $$;
create function public.zils_billing_attach_checkout(p_purchase uuid,p_mode text,p_session text,p_url text)
returns void language plpgsql security definer set search_path=public,pg_temp as $$
begin
  perform zils_billing_require_mode(p_mode);
  if p_session is null or length(p_session)>255 or p_session not like 'cs_%'
    or p_url is null or p_url not like 'https://checkout.stripe.com/%' then raise exception 'invalid checkout'; end if;
  update zils_billing_purchases set session_id=p_session,checkout_url=p_url
    where id=p_purchase and mode=p_mode and (session_id is null or session_id=p_session);
  if not found then raise exception 'checkout conflict'; end if;
end $$;
create function public.zils_billing_expire(p_purchase uuid,p_mode text) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
begin
  perform zils_billing_require_mode(p_mode);
  update zils_billing_purchases set status='expired' where id=p_purchase and mode=p_mode and status='pending';
end $$;

create function public.zils_billing_fulfill(p_purchase uuid,p_mode text,p_session text,p_payment text,
  p_amount integer,p_event text,p_receipt text default null,p_refunded integer default 0,
  p_disputed boolean default false,p_event_created bigint default 0) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
declare p zils_billing_purchases; a zils_billing_accounts; reversal integer;
begin
  perform zils_billing_require_mode(p_mode);
  select * into p from zils_billing_purchases where id=p_purchase and mode=p_mode;
  if not found or p.amount_cents<>p_amount or p_session is null or p_session not like 'cs_%'
    or (p.session_id is not null and p_session<>p.session_id)
    or p_payment is null or p_payment not like 'pi_%' or length(p_payment)>255
    or (p.payment_id is not null and p.payment_id<>p_payment)
    or p_refunded is null or p_refunded<0 or p_refunded>p.amount_cents
    or p_disputed is null or p_event is null or p_event='' then raise exception 'payment conflict'; end if;
  insert into zils_billing_accounts(owner_id,mode) values(p.owner_id,p_mode) on conflict do nothing;
  select * into a from zils_billing_accounts where owner_id=p.owner_id and mode=p_mode for update;
  select * into p from zils_billing_purchases where id=p_purchase for update;
  if (p.payment_id is not null and p.payment_id<>p_payment)
    or (p.session_id is not null and p.session_id<>p_session) then raise exception 'payment conflict'; end if;
  insert into zils_billing_events(mode,event_id) values(p_mode,p_event) on conflict do nothing;
  -- Purchase identity, not event identity, is the exactly-once financial boundary.
  if p.credited_at is null then
    update zils_billing_accounts set balance_nanos=balance_nanos+p.amount_cents::bigint*10000000,
      first_purchase_id=coalesce(first_purchase_id,p.id),
      free_training_runs=case when first_purchase_id is null and p_refunded<p.amount_cents then 1 else free_training_runs end,
      bonus_eligible=case when first_purchase_id is null then p_refunded<p.amount_cents else bonus_eligible end
      where owner_id=p.owner_id and mode=p_mode;
    insert into zils_billing_ledger(owner_id,mode,kind,amount_nanos,reference)
      values(p.owner_id,p_mode,'topup',p.amount_cents::bigint*10000000,p.id::text);
    update zils_billing_purchases set credited_at=clock_timestamp(),payment_id=p_payment,session_id=coalesce(session_id,p_session),status='paid'
      where id=p.id;
  end if;
  reversal:=greatest(p.refunded_cents,p_refunded)-p.refunded_cents;
  if reversal>0 then
    update zils_billing_accounts set balance_nanos=balance_nanos-reversal::bigint*10000000
      where owner_id=p.owner_id and mode=p_mode;
    insert into zils_billing_ledger(owner_id,mode,kind,amount_nanos,reference)
      values(p.owner_id,p_mode,'refund',-reversal::bigint*10000000,p.id::text||':'||p_refunded::text);
    update zils_billing_purchases set refunded_cents=p_refunded,status='refunded' where id=p.id;
  end if;
  if greatest(p.refunded_cents,p_refunded)=p.amount_cents then
    update zils_billing_accounts set free_training_runs=0,bonus_eligible=false
      where owner_id=p.owner_id and mode=p_mode and first_purchase_id=p.id;
  end if;
  update zils_billing_purchases set receipt_url=coalesce(p_receipt,receipt_url),
    disputed=case when p_event_created>dispute_updated then p_disputed
      when p_event_created=dispute_updated then disputed or p_disputed else disputed end,
    dispute_updated=greatest(dispute_updated,p_event_created) where id=p.id;
end $$;

create function public.zils_billing_refund(p_mode text,p_payment text,p_refunded integer,p_event text)
returns void language plpgsql security definer set search_path=public,pg_temp as $$
declare p zils_billing_purchases;
begin
  select * into p from zils_billing_purchases where payment_id=p_payment and mode=p_mode;
  if not found or p.credited_at is null then raise exception 'payment not fulfilled'; end if;
  perform zils_billing_fulfill(p.id,p_mode,p.session_id,p.payment_id,p.amount_cents,p_event,
    p.receipt_url,p_refunded,p.disputed,p.dispute_updated);
end $$;
create function public.zils_billing_dispute(p_mode text,p_payment text,p_disputed boolean,p_event text,
  p_event_created bigint default 0) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
declare p zils_billing_purchases;
begin
  select * into p from zils_billing_purchases where payment_id=p_payment and mode=p_mode;
  if not found or p.credited_at is null then raise exception 'payment not fulfilled'; end if;
  perform zils_billing_fulfill(p.id,p_mode,p.session_id,p.payment_id,p.amount_cents,p_event,
    p.receipt_url,p.refunded_cents,p_disputed,p_event_created);
end $$;

create function public.zils_billing_reserve(p_id uuid,p_owner uuid,p_mode text,p_kind text,p_nanos bigint)
returns void language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_billing_accounts; bonus boolean := false; cost bigint := p_nanos;
begin
  if p_nanos is null or p_nanos<0 or p_kind not in ('inference','training') then raise exception 'invalid reservation'; end if;
  insert into zils_billing_accounts(owner_id,mode) values(p_owner,p_mode) on conflict do nothing;
  select * into a from zils_billing_accounts where owner_id=p_owner and mode=p_mode for update;
  if exists(select 1 from zils_billing_purchases where owner_id=p_owner and mode=p_mode and disputed) then
    raise exception using errcode='P0402',message='Payment requires attention';
  end if;
  if p_kind='training' and a.free_training_runs>0 and a.bonus_eligible then bonus:=true; cost:=0; end if;
  if a.balance_nanos-a.reserved_nanos<cost then
    raise exception using errcode='P0402',message='Insufficient prepaid credit';
  end if;
  insert into zils_billing_reservations(id,owner_id,mode,kind,amount_nanos,bonus)
    values(p_id,p_owner,p_mode,p_kind,cost,bonus);
  update zils_billing_accounts set reserved_nanos=reserved_nanos+cost,
    free_training_runs=free_training_runs-case when bonus then 1 else 0 end
    where owner_id=p_owner and mode=p_mode;
end $$;
create function public.zils_billing_settle(p_id uuid,p_charge boolean) returns void
language plpgsql security definer set search_path=public,pg_temp as $$
declare r zils_billing_reservations;
begin
  select * into r from zils_billing_reservations where id=p_id for update;
  if not found or r.status<>'reserved' then return; end if;
  update zils_billing_accounts set reserved_nanos=reserved_nanos-r.amount_nanos,
    balance_nanos=balance_nanos-case when p_charge then r.amount_nanos else 0 end,
    free_training_runs=case when not p_charge and r.bonus and bonus_eligible then 1 else free_training_runs end
    where owner_id=r.owner_id and mode=r.mode;
  if p_charge then
    insert into zils_billing_ledger(owner_id,mode,kind,amount_nanos,reference)
      values(r.owner_id,r.mode,r.kind,-r.amount_nanos,r.id::text);
  end if;
  update zils_billing_reservations set status=case when p_charge then 'settled' else 'released' end where id=p_id;
end $$;

alter table public.zils_api_usage add column billable_tokens bigint check(billable_tokens between 0 and 2147483648);
alter table public.zils_api_usage add column billing_mode text check(billing_mode in ('test','live'));
create function public.zils_billing_usage() returns trigger
language plpgsql security definer set search_path=public,pg_temp as $$
declare active_mode text;
begin
  if TG_OP='INSERT' then
    select mode into active_mode from zils_billing_settings where singleton;
    if active_mode='off' then return new; end if;
    if new.billable_tokens is null then raise exception using errcode='P0422',message='Billable meter unavailable'; end if;
    new.billing_mode:=active_mode;
    perform zils_billing_reserve(new.request_id,new.owner_id,active_mode,'inference',new.billable_tokens*42);
  elsif old.status='started' and new.status in ('completed','failed') and old.billing_mode is not null then
    if new.billable_tokens is distinct from old.billable_tokens or new.billing_mode is distinct from old.billing_mode then
      raise exception 'usage price is immutable';
    end if;
    perform zils_billing_settle(new.request_id,new.status='completed');
  end if;
  return new;
end $$;
create trigger zils_billing_usage before insert or update on public.zils_api_usage
for each row execute function public.zils_billing_usage();

drop function public.zils_api_admit(uuid,uuid,uuid,bigint);
create function public.zils_api_admit(p_owner uuid,p_key uuid,p_request uuid,p_tokens bigint,
  p_billable_tokens bigint default null) returns text
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
  insert into zils_api_usage(request_id,owner_id,reserved_tokens,billable_tokens) values(p_request,p_owner,p_tokens,p_billable_tokens);
  return 'allowed';
exception when sqlstate 'P0402' then return 'insufficient_credit';
  when sqlstate 'P0422' then return 'billing_meter_unavailable';
end $$;

create function public.zils_billing_training() returns trigger
language plpgsql security definer set search_path=public,pg_temp as $$
declare active_mode text; started boolean;
begin
  if old.status='uploading' and new.status='validating' then
    select mode into active_mode from zils_billing_settings where singleton;
    if active_mode<>'off' then perform zils_billing_reserve(new.id,new.owner_id,active_mode,'training',2000000000); end if;
  elsif old.status not in ('completed','failed') and new.status in ('completed','failed') then
    started:=old.status='evaluating' or exists
      (select 1 from fez_training_assignments where job_id=new.id and (attempts>0 or state in ('leased','submitted')));
    perform zils_billing_settle(new.id,new.status='completed' or (new.error='Cancelled by customer.' and started));
  end if;
  return new;
end $$;
create trigger zils_billing_training before update on public.fez_training_jobs
for each row execute function public.zils_billing_training();

-- Fence the assignment/job race: a cancelled job must not be revived after its refund.
create or replace function public.fez_claim_training(p_hotkey text)
returns jsonb language plpgsql set search_path = '' as $$
declare item public.fez_training_assignments; worker_uid integer;
begin
  select uid into worker_uid from public.fez_training_workers where hotkey=p_hotkey and enabled for update;
  if not found then raise exception 'worker is not enabled'; end if;
  select a.* into item from public.fez_training_assignments a
    join public.fez_training_jobs j on j.id=a.job_id
    where a.hotkey=p_hotkey and j.status in ('queued','running') and j.deadline > now()
      and (a.state='ready' or (a.state='leased' and (a.lease_until > now() or a.attempts < 3)))
    order by (a.state='leased' and a.lease_until > now()) desc, j.created_at
    limit 1 for update of a skip locked;
  if not found then return null; end if;
  if item.state <> 'leased' or item.lease_until <= now() then
    update public.fez_training_assignments set state='leased',lease_token=gen_random_uuid(),
      lease_until=least(now()+interval '20 minutes', (select deadline from public.fez_training_jobs where id=item.job_id)),
      attempts=attempts+1 where job_id=item.job_id and hotkey=p_hotkey returning * into item;
  end if;
  update public.fez_training_jobs set status='running',updated_at=now() where id=item.job_id and status in ('queued','running');
  if not found then raise exception 'job no longer accepts training'; end if;
  return to_jsonb(item);
end $$;

-- An abandoned inference cannot silently tie up credit forever. Only an operator
-- invokes this after confirming the gateway/batch worker is stopped or recovered.
create function public.zils_billing_release_abandoned(p_before timestamptz) returns integer
language plpgsql security definer set search_path=public,pg_temp as $$
declare n integer;
begin
  if p_before>clock_timestamp()-interval '24 hours' then raise exception 'reconciliation cutoff too recent'; end if;
  update zils_api_usage set status='failed',input_tokens=null where status='started' and billing_mode is not null and created_at<p_before;
  get diagnostics n=row_count;
  return n;
end $$;

do $$ declare t text; f regprocedure;
begin
  foreach t in array array['zils_billing_settings','zils_billing_accounts','zils_billing_purchases',
    'zils_billing_events','zils_billing_ledger','zils_billing_reservations'] loop
    execute format('alter table public.%I enable row level security',t);
    execute format('revoke all on public.%I from public,anon,authenticated',t);
    execute format('grant all on public.%I to service_role',t);
  end loop;
  for f in select oid::regprocedure from pg_proc where pronamespace='public'::regnamespace
    and (proname like 'zils_billing_%' or proname='zils_api_admit') loop
    execute format('revoke all on function %s from public,anon,authenticated',f);
    execute format('grant execute on function %s to service_role',f);
  end loop;
end $$;
commit;
