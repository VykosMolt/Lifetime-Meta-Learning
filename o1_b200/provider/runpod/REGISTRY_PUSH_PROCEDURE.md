# Registry Push — the one remaining non-GPU staging action

The production image is built and identity-recorded
(`CONTAINER_IMAGE_RECORD.json`, local image id
`sha256:75582fe45ddaed41a6150adf01fb6fc62c1511828913e5e37e37e43b415e7f1b`,
~14 GB virtual / ~5.4 GB layer content). It was **not** pushed in the
pre-rental task because:

- a fresh GHCR container package defaults to **private**; a ~5 GB private
  package exceeds the 500 MB private free tier, creating a blocked-or-billable
  storage state (forbidden: no billable resource in this task);
- GHCR package visibility cannot be reliably switched to public via the API
  before the first push.

## Exact procedure (run once, before rental, ~10 minutes)

1. In the GitHub UI, pre-create the package as **public** by pushing a tiny
   placeholder OR simply push and immediately set
   `https://github.com/users/VykosMolt/packages/container/o1-b200-runner/settings`
   → *Change visibility* → **Public** (public GHCR storage is free).
2. Push:

```sh
gh auth token | docker login ghcr.io -u VykosMolt --password-stdin
docker tag o1-b200-runner:v0.2.0 ghcr.io/vykosmolt/o1-b200-runner:v0.2.0
docker push ghcr.io/vykosmolt/o1-b200-runner:v0.2.0
```

3. Record the **registry digest** printed by the push (`…@sha256:…`) into
   `RUNPOD_SESSION_CONFIG.json` (`image_digest_ref`) and regenerate the
   canonical pod-request render. The image contains no secrets and no model
   checkpoint; public visibility is consistent with the (already public)
   repository content. Mutable-tag references remain refused by the adapter —
   only the `@sha256:` digest form is accepted.
