# Real O1 v2 primary axis bank

These are real pre-O1 reconstruction measurements. They are not O1
calibration or confirmatory results.

| Axis | Canonical unit-RMS row SHA-256 | Status |
|---|---|---|
| A1_READABLE | `ce9b8feacd654df8d38c5fa72d48e0fe31714b680d5c4d135f061592cc9d7370` | real accepted readable direction |
| A2_WRITABLE | `0845b0926abecd83acb8239e4d7d15941cfb139ce1b816894dd19956ff92260e` | real exact-protocol reconstruction; historical tensor remains missing |
| A3_CAUSAL_MEAN | `284c4f43243295e4123e1e8885eda5ca9014e5cf374652b7032f1d2362d7fc74` | exact raw mean of causal induced updates |
| A4_SEQUENCE_MEAN | `2926c83868d5827a7e19495401a7b7241d897044f7ee1fca94d787c02f6c1f9c` | exact raw mean of sequence induced updates |

A3 raw mean L2 norm is `0.04112786989365476` (RMS
`0.0009088061155487916`). Its 10,000-draw task bootstrap cosine-to-full
q05/median are `0.6751363203716227` / `0.8070085885961052`; leave-one-task-out
minimum is `0.9414609452611218`.

A4 raw mean L2 norm is `0.05264098849247025` (RMS
`0.0011632124978558978`). Its bootstrap q05/median are
`0.7546887488744646` / `0.8644325268356808`; leave-one-task-out minimum is
`0.966199319294551`.

A3/A4 cosine is `0.5720586047705143`. Per-reference-example causal/sequence
cosines range from approximately `0.286143` to `0.535089`, with median
approximately `0.501418`. Capture parity is PASS. The checkpoint historical
tree hash is `7dbbe4ef...a802`; causal and sequence adapter hashes are
`f0831a9a...15f4` and `75140bf3...97c`.

The structured cosine matrix is:

```text
 1.000000  -0.037346  -0.018779   0.005942
-0.037346   1.000000  -0.043011  -0.025476
-0.018779  -0.043011   1.000000   0.572059
 0.005942  -0.025476   0.572059   1.000000
```

The axis verifier returns `SEALABLE`: exact A1/A2 embedded-source identity,
exact A3/A4 source-matrix recomputation, exact reference ordering, canonical
row bytes, deterministic random-bank regeneration, Gram matching, complete
hash coverage, and no bytecode all pass. Upstream fitting/capture claims remain
hash-bound or attested as stated in the reconstruction attestation.

