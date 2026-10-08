# Prepaid billing

Billing is disabled by default. This implementation supports manual Stripe
Checkout top-ups of **$5, $20, $50, or $100 USD**. All existing API keys for an
account share its balance. The first paid top-up grants one standard training
run without reducing purchased credit; later runs cost $2. Inference costs
$0.042 per million logical input tokens, with no output charge. There are no
subscriptions or automatic recharges.

The ledger uses integer nano-USD: one dollar is 1,000,000,000 units and an input
token costs 42 units. HTTP responses encode monetary ledger values as strings.
The runtime's [logical input token count](decision-api.md) includes the shared
state once and each question once. Usage reserves available funds before work,
then settles or releases the reservation. Insufficient funds or an unresolved
payment dispute blocks new work across the account's keys. Existing throughput
and concurrency limits still apply.

Training reserves its free run or $2 when a job leaves uploading. Completed
evaluation consumes it; infrastructure or validation failure releases it.
Cancellation after execution begins consumes the reservation. Jobs approved
before billing was enabled remain grandfathered.

Bulk inference cancelled or expired during execution releases its reservation
when the worker's result is rejected. A worker crash or an ambiguous database
outage can leave a reservation pending. Unfinished usage is retained for
reconciliation even after normal usage retention expires. Operators can inspect
those requests and call `zils_billing_release_abandoned(p_before)` through a
privileged database connection after confirming no work remains in flight.
The cutoff must be at least 24 hours old; it marks unfinished inference failed
and releases its credit. Completed charges are unchanged. This recovery is
manual, and does not cancel or reconcile training jobs.

## Isolated test setup

Use Python 3.13, PostgreSQL's `psql` client, and the existing
[gateway setup](decision-api.md#start-from-a-fresh-clone) on macOS, Linux, or WSL 2.
Create a separate Supabase test project, gateway, worker, and private runtime.
Test credits must never buy production GPU work. Apply the training, decision
API, and batch migrations to that test project before the billing migration.
The service-role key and database connection must belong to the same project.

From a fresh clone at the repository root:

```bash
python3.13 -m venv .venv-api
.venv-api/bin/python -m pip install -r requirements/api.txt
mkdir -p .private/api
cp examples/zils-api/service.env.example .private/api/service.env
chmod 600 .private/api/service.env
```

Replace the placeholders in that file, including these server-only settings:

```dotenv
ZILS_BILLING_MODE=test
STRIPE_SECRET_KEY=sk_test_REPLACE_WITH_TEST_SECRET
STRIPE_WEBHOOK_SECRET=whsec_REPLACE_WITH_ENDPOINT_SIGNING_SECRET
ZILS_BILLING_RETURN_ORIGIN=https://YOUR_TEST_DASHBOARD.example
ZILS_BILLING_ALLOW_LIVE=false
```

The return origin must be a fixed exact origin, without a path or query. Local
test dashboards can use `http://localhost:3000`. Checkout always returns to
`/billing?checkout=success` or `/billing?checkout=cancelled` on that origin.
Return URLs are never accepted from a customer request. A success redirect
does not establish payment; the dashboard waits for the verified webhook.

Supply a privileged test-project connection as `SUPABASE_DB_URL` and apply:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/202610080001_prepaid_billing.sql
```

Load the environment when starting the gateway as described in its setup guide.
The database mode and gateway mode must match. The migration starts in `off`
mode and uses service-role-only tables and RPCs. Billing endpoints report 503
when payment settings are missing or incompatible. Test and live account rows
are isolated, but separate test infrastructure is still required.

Create a Stripe **test-mode snapshot webhook for your own Stripe account** at
`https://YOUR_TEST_API.example/v1/billing/webhook`, using API version
`2026-09-30.endive`, and select:

- `checkout.session.completed`
- `checkout.session.async_payment_succeeded`
- `charge.refunded`
- `charge.dispute.created`, `charge.dispute.updated`, `charge.dispute.closed`
- `charge.dispute.funds_withdrawn`, `charge.dispute.funds_reinstated`

Copy that endpoint's signing secret into `STRIPE_WEBHOOK_SECRET`. This service
does not accept Connect or organization-context events. Stripe's
[webhook guide](https://docs.stripe.com/webhooks) describes endpoint setup and
local forwarding with the Stripe CLI. Forward to port 8920's exact webhook path
and use the CLI listener's signing secret when testing locally.

Deploy the upgraded gateway, webhook, worker, and runtime before activating the
database switch. Verify the runtime's prepare response includes the logical
`billable_tokens` count and the worker uses the upgraded admission code. The
public usage response exposes that count as `billable_input_tokens`.
Then, on the isolated test project only:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -c "update public.zils_billing_settings set mode='test' where singleton;"
```

This singleton changes billing enforcement for the whole database. Do not run
it against a production project. Until both modes match, the gateway rejects
billing operations. The deployment remains a test environment throughout this
procedure.

The pinned [official Python SDK 16.0.0](https://github.com/stripe/stripe-python/releases/tag/v16.0.0)
verifies the original bytes and signature. Signatures outside a five-minute
window, wrong-mode events, malformed framing, duplicate signature headers, and
bodies over 256 KiB are rejected. API calls have bounded connection/read
timeouts and no automatic SDK retries; Stripe webhook retries and purchase
idempotency support recovery.

## Customer API

Billing management uses a non-anonymous Supabase user session in
`Authorization: Bearer <access_token>`. An API key cannot manage billing.

| Request | Response |
| --- | --- |
| `GET /v1/billing` | Mode, currency, balance/reserved/available nano-USD strings, free runs, top-up options, and the latest 100 transactions and payments |
| `POST /v1/billing/checkout` with `{"amount_cents":500,"idempotency_key":"<UUID>"}` | `{"url":"https://checkout.stripe.com/…","purchase_id":"<UUID>"}` |
| `POST /v1/billing/webhook` | Stripe-only callback, raw JSON plus `Stripe-Signature`, no bearer credential |

The server derives a purchase UUID from the verified account, mode, and client
idempotency UUID. Reusing a key with a different amount returns 409. A retry
reuses the same purchase and Stripe idempotency key. Persist the client key
until that purchase is resolved; `purchase_id` correlates it with the payment
list. Other accounts cannot reuse it to access the purchase.

An expired session returns 410 and its payment status becomes `expired`; start
a new purchase with a new UUID. A paid purchase or a completed session awaiting
its webhook returns 409; refresh billing to observe settlement. If the gateway
created a Stripe session but could not record its response, retries use the
same Stripe key for up to 23 hours. An older unattached purchase returns 409
and requires operator reconciliation, because Stripe can eventually discard
idempotency keys. Never resolve this by issuing the same purchase again.

## Payment reconciliation

Only a complete paid Checkout session with the stored purchase, verified owner,
fixed USD amount, matching successful PaymentIntent, and paid charge can credit
funds. Session and PaymentIntent metadata contain the server-selected purchase,
owner, and mode. The latest Stripe objects are retrieved before settlement;
the database transaction makes credit and any existing reversal atomic.

Refund events can arrive before the paid event. The handler finds the Checkout
session through its PaymentIntent and credits only the net balance in the same
transaction. Cumulative refunds only increase; replay and out-of-order delivery
cannot duplicate credit or reverse the same cents twice. Partial and full
refunds appear as `refunded` payments. If spent credit is later refunded, the
balance can become negative and new spending remains blocked until covered.

Open disputes freeze new usage. Winning or closing a warning clears that
dispute; a lost dispute reverses the full purchase and clears its freeze.
Reversals due to lost disputes appear as `refund` ledger entries. A later Stripe
refund cannot debit that purchase twice. Fully reversing the first purchase
removes its unspent training bonus; an already consumed bonus does not become
debt. Account reservations and every other unresolved dispute still apply.
If conflicting dispute updates have the same event timestamp, the ledger keeps
the account frozen conservatively. That rare case requires operator
reconciliation against Stripe's current dispute state before clearing the hold.

Run the automated checks from the repository root:

```bash
.venv-api/bin/python -m unittest tests.test_billing tests.test_decision_http
```

These use signed fixtures and real local HTTP transport with Stripe networking
mocked at the SDK boundary. Database concurrency and settlement checks are
separate. Before exposing test Checkout, complete an actual Stripe test payment,
replay its webhook, issue partial/full test refunds, and exercise dispute events
against the isolated test project. Fixture tests do not establish that a Stripe
account, webhook URL, or deployment has been configured correctly.

### Stripe sandbox validation

Hosted Stripe Checkout was exercised with official test cards, the Python SDK's
serialized requests, a CLI-authorized sandbox transport, signed webhook
forwarding, and the actual ledger migrations in local PostgreSQL. The dashboard
used a synthetic local sign-in. The observed results were:

| Scenario | Verified result |
| --- | --- |
| $5 payment | $5 credit and one included training run |
| Four concurrent duplicate webhook deliveries | One top-up entry; balance and allowance unchanged |
| $2 partial refund, then the remaining $3 | Balance fell to $3, then $0; full reversal removed the unused bonus |
| Disputed $5 payment | New spending rejected while the account still held $5 |
| Lost dispute | Original signed closing event reversed $5, removed the unused bonus, and cleared the hold |

This validation exposed two fixture blind spots that are covered by regression
tests: the current Checkout API requires `allowed_payment_method_types`, and
Stripe dispute IDs use the `du_` prefix. The closing event initially failed ID
validation and passed when replayed with its original signature after the fix.

This is sandbox evidence, not a production rollout. It does not validate a
hosted Supabase deployment, production API-key configuration, a publicly hosted
webhook endpoint, real customer sign-in, or GPU consumption. Repeat the checks
against the intended isolated hosted deployment before opening paid access.

Live payments require separate operator authorization and verification, a live
secret, live webhook, matching database mode, an HTTPS dashboard origin, and
`ZILS_BILLING_ALLOW_LIVE=true`. Merely supplying a live key while configured for
test is rejected. This release's intended initial rollout is test mode only.
