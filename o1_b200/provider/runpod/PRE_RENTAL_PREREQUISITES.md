# External Pre-Rental Prerequisites — current state

Software readiness is PASS (155/155 checks; see
`reports/RUNPOD_PRE_RENTAL_READINESS.*`). Rental remains inappropriate until
ALL of the following flip to done. None of them is runner development; each
is credential-gated account/staging work.

## The gate (all must be green before credits/authorization)

```text
RUNPOD READ-ONLY PREFLIGHT:      PASS
REMOTE IMAGE DIGEST:             RESOLVED AND VERIFIED
PRIVATE HF ARTIFACT STAGING:     PASS
POD-SIDE DOWNLOAD HASH TEST:     PASS   (stage_artifacts_hf verify)
RESULT DESTINATION ROUND-TRIP:   PASS   (stage_artifacts_hf result-roundtrip)
RUNPOD_SESSION_CONFIG:           ZERO UNRESOLVED REQUIRED FIELDS
```

Current: software readiness PASS; every line above PENDING (all are
credential-gated operator actions).

## 1. Read-only preflight (GET-only; creates nothing)

```sh
# create a least-privilege key in the RunPod console first
# (see KEY_PERMISSIONS.md); never paste it into chat or the repo
export RUNPOD_API_KEY='<from your password manager>'
cd /home/moloch/ouro_worktrees/o1-v2-b200-runner
./o1_b200/scripts/runpod_pre_rental_readonly_check.sh
```

Must confirm live: key works; schema still matches pin `0bbdd828…`; a
single-B200 Secure offer exists at ≤ USD 5.89/h; no unexpected Pod runs.

## 2. Remote image publication

Follow `REGISTRY_PUSH_PROCEDURE.md` (3 commands + one visibility click).
The local image id is `sha256:75582fe4…`; the push prints the REMOTE
manifest digest — use the immutable reference
`ghcr.io/vykosmolt/o1-b200-runner@sha256:<remote-manifest-digest>` in
`RUNPOD_SESSION_CONFIG.json` (`image_digest_ref`). Do NOT rely on the
`v0.2.0` tag after pushing; mutable tags are refused by the adapter.
Cross-check that GHCR reports the same digest the push returned:

```sh
docker buildx imagetools inspect ghcr.io/vykosmolt/o1-b200-runner:v0.2.0 \
  | grep Digest        # must equal the digest printed by docker push
```

Note: the credential-piping step (`gh auth token | docker login …`) is
deliberately left to the operator.

## 3. Private large-artifact staging (before any billing clock)

```sh
hf auth login       # WRITE-capable token, private-repo scope
cd /home/moloch/ouro_worktrees/o1-v2-b200-runner
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo VykosMolt/o1-b200-staging upload
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo VykosMolt/o1-b200-staging verify   # = ARTIFACT DOWNLOAD TEST
```

Stages the 5.0 GB Ouro-RLTT checkpoint, tokenizer binding, axis package,
verified O1 package zip, calibration manifest, and seed matrix into a
PRIVATE Hugging Face repo (free tier; refuses to proceed if the repo is not
private), then re-downloads everything through the pod's fetch path and
verifies every SHA-256 against the transfer manifest. The pod later needs
only a READ-scoped `HF_TOKEN` env value at launch (never baked into the
image).

## 4. Result-destination round-trip (separate repo, separate scope)

```sh
PYTHONPATH=. python -m o1_b200.provider.runpod.stage_artifacts_hf \
    --repo VykosMolt/o1-b200-results result-roundtrip
```

Creates the PRIVATE results repo, uploads a probe archive through the pod's
upload path, re-downloads it through the driver's `hf://` download path,
and hash-compares (writes `RESULT_ROUNDTRIP_RECORD.json`). The results repo
is deliberately separate from the staging repo: the pod's WRITE token is
fine-grained to results only and can never touch the checkpoint.
`RUNPOD_SESSION_CONFIG.json` `result_source` is already resolved to
`hf://VykosMolt/o1-b200-results/O1_B200_CALIBRATION/results.tar.gz`; after
this test, `image_digest_ref` is the only unresolved field left.

## Then, and only then

Add the separately approved credits, create the one-use
`B200_RENTAL_AUTHORIZATION.json`, and launch
`./o1_b200/o1_runpod_b200_zero_touch.sh --authorization … --execute-authorized-rental`.
