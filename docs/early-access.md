# Early-access admission

Zils customer admission is separate from Supabase authentication. An approved,
activated account is required for training, API-key management, and inference.
The website provides the public application and private administrator interface.

Apply `supabase/migrations/202610070001_early_access.sql` once, after the existing
training and decision API/batch migrations, to the same project. Deploy the
matching coordinator and website changes together. The migration blocks existing
customer API keys until their owners are explicitly approved and activated;
prepare the rollout before applying it to a live service. A missing admission
function fails closed. It does not change global Auth signup settings.

The website setup is documented in
[the website repository](https://github.com/ooo-hq/zils-web/blob/main/docs/early-access.md).
Its server verifies admin email identity before invoking service-role-only RPCs.
Neither customers nor miners may call those RPCs directly. Applications and email
addresses have RLS enabled and no anonymous/authenticated table privileges.

The default capacity is 25. Active customers and unexpired invitations each
reserve one spot; Zil count is independent. Seven-day invitation expiry releases
unused capacity. All reservations, acceptance, and pauses serialize on the single
settings record, preventing concurrent over-allocation. Email delivery occurs
after reservation; a failed email is recoverable by resending in the admin page.

Pausing blocks future customer training requests, key management, existing API
keys, and new inference admissions (including batch items). It does not abort
work already admitted, terminate GPU jobs, or revoke storage URLs already issued.
The public playground's service-key owner also needs explicit approval. No miner
credentials, rewards, billing rates, or automatic financial credits are changed.

Run `make check-queue-db` with PostgreSQL 16+ to test the real migration and its
capacity race, expiry, identity binding, pause, old-key denial, and privileges.
`tests.test_access` covers coordinator/key-service enforcement and failure closure.
The checks create disposable local state and never contact production.
