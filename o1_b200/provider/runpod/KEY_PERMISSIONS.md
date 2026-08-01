# RunPod API-Key Handling and Minimum Permissions

## Handling rules (enforced in code)

The key is read ONLY from `RUNPOD_API_KEY` or `RUNPOD_API_KEY_FILE`
(owner-only permissions enforced; group/other access refused). It is
registered with the redaction layer at load, which scrubs Authorization
headers, bearer tokens, RunPod `rpa_…` key shapes, registry credentials,
pre-signed URLs, Git credentials, and SSH material from stdout, stderr, JSON
logs, exception strings, reports, and archives. The key is never committed,
never placed in JSON configuration, never baked into the Docker image, never
printed, and never included in packaged artifacts — the secret-scan test
suite injects recognizable fake keys and proves none surface anywhere.

Do not paste the key into chat. Set it locally:

```sh
export RUNPOD_API_KEY="$(cat /path/to/keyfile)"   # or use RUNPOD_API_KEY_FILE
```

## A. Pre-rental read-only validation (this phase)

Create a key with the **least** scope RunPod's console offers. Required
capability: authenticated **GET** on:

- `/v2/catalog/gpus`, `/v2/catalog/datacenters` (catalog/pricing/availability)
- `/v2/pods` (list own pods — verifying no unexpected billable pod)
- `/v2/billing`, `/v2/billing/pods` (billing state)

If the console offers a read-only/restricted key type, use it. No pod
create/update, no template, no storage, no registry, and no serverless
permission is needed for preflight; requests with a valid but under-scoped
key return 403, which the preflight reports as-is.

## B. Later zero-touch rental (separate approval)

The rental key additionally needs:

- `POST /v2/pods` (create — exactly one pod, interlocked),
- `POST /v2/pods/{id}/action` (`stop`/`terminate`),
- `DELETE /v2/pods/{id}` (permanent termination),
- `GET /v2/pods/{id}/logs`.

Nothing else: no network-volume, template, registry-delegation, or
serverless permission. If scoping per-operation is not offered, use a
standard key but keep it in `RUNPOD_API_KEY_FILE` (0600), delete it from the
console after the session, and keep auto-pay disabled so the key cannot
draw new funds.
