# Image configuration examples

All addresses, paths, owner IDs and fingerprints are placeholders. Copy configurations
to private storage and replace them using authorized staging services. `isolated: true`
is an operator assertion, not permission to use production. Do not place tokens in JSON.

`service.env.example` keeps admissions off; `workflow.json` extends the existing text
workflow with a separately configured image runtime. `rehearsal.json` names two existing
authorized test accounts and their environment-variable references. Replace the all-zero
fingerprints with verified immutable runtime identities.

The dataset directory must contain train/calibration/test/fresh JSONL and the referenced
JPEG/PNG files. Example row (use distinct IDs, item groups and photos across splits):

```json
{"id":"train-001","group_id":"item-001","family":"inspection","state":{},"question":{"type":"choice","instructions":"Is the component visibly damaged?","criteria":{"normal":null,"damaged":null}},"label":"normal","image":"images/photo-001.jpg"}
```

Review licensing and consent for the chosen data. The research rehearsal does not
authorize using its public research dataset to train a commercial customer model.
