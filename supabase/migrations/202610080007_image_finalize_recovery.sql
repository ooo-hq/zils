-- Keep the original claim RPC available for workers running the previous release.
-- A request-owned token makes cleanup safe even when the claim reply is lost.
begin;
create function public.zils_image_claim_finalize_request(
  p_owner uuid, p_asset uuid, p_token uuid
) returns jsonb
language plpgsql security definer set search_path=public,pg_temp as $$
declare a zils_image_assets;
begin
  if p_token is null then raise exception 'finalization token required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(p_owner::text,801));
  select * into a from zils_image_assets where id=p_asset and owner_id=p_owner for update;
  if not found or not zils_image_live(a) then return null; end if;
  if a.state='ready' then return to_jsonb(a); end if;
  if a.state='verifying' and a.finalize_until>clock_timestamp() then
    if a.finalize_token=p_token then return to_jsonb(a); end if;
    return null;
  end if;
  update zils_image_assets set state='verifying',finalize_token=p_token,
    finalize_until=clock_timestamp()+interval '20 minutes'
    where id=p_asset returning * into a;
  return to_jsonb(a);
end $$;

create function public.zils_image_release_finalize(
  p_owner uuid, p_asset uuid, p_token uuid
) returns void
language sql security definer set search_path=public,pg_temp as $$
  update zils_image_assets set state='uploading',finalize_token=null,finalize_until=null
    where id=p_asset and owner_id=p_owner and state='verifying' and finalize_token=p_token;
$$;

revoke all on function public.zils_image_claim_finalize_request(uuid,uuid,uuid),
  public.zils_image_release_finalize(uuid,uuid,uuid) from public,anon,authenticated;
grant execute on function public.zils_image_claim_finalize_request(uuid,uuid,uuid),
  public.zils_image_release_finalize(uuid,uuid,uuid) to service_role;
commit;
