# External Pre-Rental Prerequisites — current state

Software readiness is PASS (155/155 checks; see
`reports/RUNPOD_PRE_RENTAL_READINESS.*`). Rental remains inappropriate until
ALL of the following flip to done. None of them is runner development; each
is credential-gated account/staging work.

## Status

```text
RUNPOD SOFTWARE READINESS:      PASS
LIVE READ-ONLY PREFLIGHT:       PENDING  (needs RUNPOD_API_KEY)
REMOTE IMAGE PUBLICATION:       PENDING  (needs docker login; procedure below)
PRIVATE LARGE-ARTIFACT STAGING: PENDING  (needs `hf auth login`; 5.0 GB checkpoint)
ARTIFACT DOWNLOAD TEST:         PENDING  (runs after staging)
B200 RENTAL AUTHORIZATION:      NOT YET APPROPRIATE
```

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
The local image id is `sha256:75582fe4…`; the push prints the REMOTE digest
(`ghcr.io/vykosmolt/o1-b200-runner@sha256:…`) — put that exact string into
`RUNPOD_SESSION_CONFIG.json` (`image_digest_ref`). Mutable tags are refused
by the adapter. Note: the credential-piping step (`gh auth token | docker
login …`) is deliberately left to the operator.

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

## Then, and only then

Add the separately approved credits, create the one-use
`B200_RENTAL_AUTHORIZATION.json`, and launch
`./o1_b200/o1_runpod_b200_zero_touch.sh --authorization … --execute-authorized-rental`.
