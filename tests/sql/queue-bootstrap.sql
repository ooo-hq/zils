-- Run only inside an empty, disposable test cluster/database.
create role anon;
create role authenticated;
create role service_role bypassrls;
create schema auth;
create table auth.users(id uuid primary key,email text,email_confirmed_at timestamptz);
create function auth.uid() returns uuid language sql stable as
  $$ select nullif(current_setting('request.jwt.claim.sub', true),'')::uuid $$;
grant usage on schema auth to authenticated;
grant execute on function auth.uid() to authenticated;
create schema storage;
create table storage.buckets(id text primary key,name text,public boolean,file_size_limit bigint);
create table storage.objects(id uuid default gen_random_uuid(),bucket_id text);
alter table storage.objects enable row level security;
grant usage on schema storage to anon,authenticated,service_role;
grant all on storage.objects to anon,authenticated,service_role;
create policy existing_permissive_policy on storage.objects for all to authenticated using (true) with check (true);
