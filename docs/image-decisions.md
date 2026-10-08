# Image decisions

The optional Imajev vertical classifies one photo using a saved set of answers. Text
requests continue to use JevK5. It does not generate images, transcribe audio, accept
video, draw boxes, or compare reference pairs. Both image flags default to off.

## Client flow

1. Sign in and choose **Images**. Upload a still JPEG or PNG, describe the decision,
   and provide 2–16 answers. Photos stay private to the account.
2. For client training, review every answer and item group, confirm the split and
   success targets, and submit. Related views of one item belong in one split.
3. Review held-out accuracy, recall, false alarms, unknown count and denominators.
   **Training passed** and **API ready** are different states. A rejected result stays saved.
4. After **API ready**, **Try your model** fetches the owned model's saved question
   and immutable ID. The existing account/key controls access.

An unknown decision displays **Needs review**. Known-answer probabilities are
conditional on a known answer, while `unknown_probability` retains the native unknown
mass. `confidence` is the concentration of the full native distribution; it is not a
calibrated probability that the selected answer is correct. Never discard unknowns
when comparing models: the evaluator counts them as incorrect and includes their
probability in Brier loss and log loss.

## API contract

Create an image with `POST /v1/image-assets`, including `purpose: "prediction"`,
`filename`, `source_bytes`, and a SHA-256 of the original file. Use the returned PUT
URL and headers exactly, then `POST /v1/image-assets/{id}/complete` with `{}`. Creation
and finalization accept a signed-in session or the owner's existing Zils API key.
Only opaque asset IDs enter prediction requests; external URLs and encoded image
strings are rejected. Never send a source filename as decision context.

With an API key, send a standard decision request to `POST /v1/systemone`:

```json
{
  "model": "MODEL_ID_FROM_YOUR_OWNED_LISTING",
  "state": {},
  "questions": {
    "inspection": {
      "type": "choice",
      "instructions": "Is the product visibly damaged?",
      "criteria": {"normal": null, "damaged": null}
    }
  },
  "images": [{"asset_id": "FINALIZED_ASSET_UUID"}]
}
```

For a trained model, use its exact `task.question` and `task.outcome_order` from the
owned listing. JSON object order must follow the declared outcome order for ties.
The browser uses session routes `GET /v1/image-models` and
`POST /v1/image-decisions`. Those routes do not create API keys. Models owned by
another account are omitted from listings and return 404 on use. Runtime addresses,
tokens, signed image read URLs and raw held-out predictions are never returned.

## Billing

Image requests use the existing prepaid account, rate limits and API keys. The
`usage.billable_input_tokens` meter counts the model's processed visual tokens plus
one canonical copy of the supplied state and question. It excludes asset IDs,
filenames, model routing and internal prompt wrappers. The existing input-token
price applies; `usage.input_tokens` remains the full processed context used for
resource limits. Credit is reserved before prediction and settled once on success;
a failed prediction releases its reservation. Insufficient credit returns 402.

The browser's batch tool submits images sequentially, keeps completed results, and
pauses on a credit error. Top up in Billing and resume only the remaining images.

## Limits and retention

| Limit | Bound |
| --- | --- |
| Input | One still JPEG or PNG, 10 MiB source and canonical output |
| Decoded photo | 16 million pixels; 8,192 pixels per edge |
| Model input | RGB PNG; processor budget 65,536–400,000 pixels; at most 4,096 actual tokens |
| Question text | Flattened instructions and each answer description: at most 2,000 characters; put longer context in `state`, within the total token limit |
| Decision | One choice question; 2–16 named answers; realtime only |
| Prediction/draft assets | 24 hours; active submitted jobs retain required data |
| Training assets | 30 days after the job reaches a terminal state |
| Runtime read grants | At most 10 minutes, bounded by asset and lease expiry |
| Accepted artifacts | Retained until explicit retirement |

EXIF orientation is applied before canonical RGB encoding. Metadata is removed.
Truncated, animated, oversized and undecodable files fail before GPU work. Cleanup
waits for outstanding upload grants and finalization leases; disabling admissions
must not disable cleanup.

401 means sign in or refresh the key; 402 means add credit in Billing; 404 means the image/model is unavailable to
this account or expired. Re-upload expired photos. 413 means the actual model context
is too large. 422 means invalid image/question or a mismatch with the saved task.
429 means account/upload limits; 503 means unavailable runtime or capacity. No image
bulk endpoint is supported.

See [image training and rollout](image-training.md) for installation and evidence.
