# O1 — Oracle Reachability Gate for the Ouro-RLTT Writable Action Space

**Preregistration v1.5 — executable draft for sealing (package maintenance revision 1.5.2)**
Jan Kirin · Project O, experiment 1
Status: **UNSEALED.** Every `FREEZE_SLOT` in the manifest must be replaced by
a real artifact hash or executable-calibration output, and the two pre-outcome
commitments must be created before their respective generations begin.

This revision incorporates the full adversarial audit through v1.4 and closes the
precommit, runtime-provenance, deterministic-randomization, executable-calibration,
seed-binding, and record-chronology gaps. Remaining `FREEZE_SLOT` values are left blank rather than invented.

**Status: CODE-VERIFIED, EXPERIMENT UNSEALED.** §4a and §4b and all policy choices
in the template are fixed. The actual checkpoint, source-derived axis tensors,
task cohorts, runtime artifact hashes, calibration records, calibrated values,
and timestamps are not present in this package. They must be produced from the
real frozen artifacts and bound through the executable chronology in §§14–16.

---

## 1. Scientific question and stopping logic

The prior work establishes that Ouro-RLTT's recurrent trajectories carry
externally readable process-quality information, that this information converts
at the decision level (abstention, terminal selection among completed
candidates), and that every tested route to *generative* control — directional
steering, tap-gated loop allocation, matched frozen forking, sign-conditioned
LoRA binding — is negative or bounded. The bounded four-task fork screen
removes evidence for a gain but cannot estimate a general deficit.

O1 asks the question that must be answered before any controller is trained:

> Does the predeclared internal intervention family contain verifier-correct
> outcomes that matched ordinary sampling does not already reach, at equal
> compute, under oracle selection?

This is deliberately not a question about readouts, critics, or interfaces.
An oracle with perfect knowledge of every branch's outcome is the ceiling on
what any learned controller over this action space could achieve. If the
ceiling is at zero, O2–O4 are unrunnable and no critic quality can rescue them.

**Stopping logic.**

| O1 result | Verdict | Next |
|---|---|---|
| No outcome diversity in the structured bank | Action parameterization inadequate | Report; revise action family or boundary; do not train a critic |
| Diversity, no selection headroom | Outcomes move but not in useful value | Report; revise |
| Selection headroom, no reachability headroom | Oracle can choose within the bank but does not beat matched sampling | Report as a compute-allocation opportunity, not a capability one; O2 optional, O4 not justified |
| Positive reachability headroom | Real generative opportunity | O2 offline action-value critics, then O3 binding, then O4 proposal learning |

**Four verdict states, and why three is not enough.** Powering a test to detect
+0.05 with 80% probability does not make failure to reject evidence that the
effect is below +0.05. A nonsignificant primary is therefore *not* by itself a
negative oracle, and v1.0 wrongly implied it was. The verdict is one of:

| State | Rule | Meaning |
|---|---|---|
| `POSITIVE_HEADROOM` | H > 0 and the directional exact two-sided McNemar test rejects at 0.05 | O2 justified; bootstrap interval reported descriptively |
| `HARMFUL` | H < 0 and the exact two-sided McNemar test rejects at 0.05 | The intervention reduces reachability; bootstrap interval reported descriptively |
| `PRACTICALLY_NULL_AT_0.05` | one-sided 95% upper bound < +0.05 | Headroom of at least the practical threshold is *excluded* — the action-space deficit claim |
| `INCONCLUSIVE` | neither of the above | Nothing established; do not report as a deficit |

Only `PRACTICALLY_NULL` sharpens the standing limitation into an action-space
deficit. `INCONCLUSIVE` means the run did not establish either positive headroom
or practical equivalence. The percentile bootstrap is descriptive and never
carries the equivalence verdict; an exact one-sided Clopper–Pearson bound on
`p10`, and therefore conservatively on `H_reach`, does.

A negative primary is a result, not a failure. It converts the standing
limitation — *the bounded screen cannot estimate a general deficit* — into a
powered estimate over a declared magnitude range, action basis, and difficulty
band. Both branches produce a publishable artifact, which is why the gate is
worth running before any external compute is requested.

---

## 2. Calibration cohort — construction and disjointness

Two cohorts are generated from the same frozen generator revision and are
disjoint by task ID **and task-content hash**. Their exact offsets and task
manifests are frozen artifacts; no placeholder value is supplied here.

Before any calibration continuation exists, `calibration_precommit.py` creates
`CALIBRATION_PRECOMMIT.json`. It binds the calibration task manifest, the exact
calibration-generation design, the actual checkpoint/tokenizer/prompt/parser/
verifier/axis hashes, the generation module, and the repository clean/diff
attestation. Every calibration row and the calibration metadata carry that
precommit hash.

Calibration runs the exact K=8 generation geometry, decoder, verifier, answer
format, checkpoint, writable boundary, axes, and token budget used by
confirmation. `calibration_analysis.py` deterministically recomputes:

1. baseline any-correct-of-8 by generator setting and the two difficulty strata;
2. per-alpha coherence verdicts and the complete pass/fail partition;
3. `alpha_star` and the optional secondary magnitude;
4. bank-level CRN discordance and its frozen one-sided upper confidence bound;
5. measured throughput, optional-arm drop plan, `G_power`, `G_max`, `G_confirm`,
   and the budget-constrained MDE.

The final manifest is rejected unless all copied calibration values equal this
recomputation byte-for-byte. A target effect larger than the calibrated
upper discordance bound produces the explicit halt
`TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND` rather than a raw exception.

---

## 3. Difficulty-band selection

Both headroom quantities are maximized at intermediate difficulty and vanish at
both ends. A pool whose baseline any-correct-of-8 sits near 1.0 caps `H_reach`
by construction regardless of bank size — the failure already observed once,
where a pool reached an any-correct-of-6 ceiling of 0.978 and could not
differentiate. Difficulty is therefore the primary design variable, not a
nuisance parameter.

**The band is calibrated on the calibration cohort and transferred unchanged.**
Selecting the confirmatory slice on its own realized baseline success would be
outcome-conditioned task selection and would invalidate the endpoint.

Two strata, frozen before any confirmatory intervention branch exists:

| Stratum | Predicted baseline any-correct-of-8 | Target share |
|---|---|---|
| hard-medium | 0.40 – 0.55 | 50% |
| medium | 0.55 – 0.70 | 50% |

Selection uses **generator-level** estimates `p_base(d)` from calibration, where
`d` covers the frozen structural knobs `proof_depth` and `rule_count`. Proof depth is
the instrument of choice because it is an exogenous difficulty control rather
than a statistic derived from intervention performance.

**Malformedness must not become the difficulty mechanism.** Calibration
additionally requires a well-formed commitment rate ≥ 0.85 at the chosen
settings, with a token budget large enough that difficulty comes from reasoning
rather than truncation. A stratum that meets its `p_base` band by truncating is
rejected.

---

## 4. Magnitude and coherence sweep

Perturbation magnitude is a live candidate explanation of the frozen-control
null. A confirmatory run at a single inherited α would produce an
uninterpretable negative: *no diversity at the one magnitude we tested* is not
*no diversity in this action family*.

Every perturbation is normalized to the residual RMS at the actual writable
boundary. On the calibration cohort, sweep both signs over

```
|α| ∈ {0.005, 0.01, 0.02, 0.04, 0.08}
```

spanning below the previously useful branch scale, the existing tested safe
envelope near 0.02, and beyond it.

**Coherence gate** — predeclared, deliberately weak enough not to reject
unusual but valid reasoning:

- valid final-answer syntax reachable
- no NaN or Inf
- non-empty continuation
- no catastrophic repetition loop
- well-formed rate noninferior to matched baseline at margin −0.05, judged on the lower one-sided 95% bound (§4); one margin, not two

Explicitly **not** gated on correctness or on model log-probability.

**Coherence criterion is frozen before the sweep, not derived from it.** The
sweep determines which magnitudes satisfy a fixed criterion; it does not define
the criterion. The criterion is: well-formed rate floor 0.85 applied
independently in each stratum, plus a noninferiority margin

```
p_WF,int − p_WF,base ≥ −0.05
```

judged on the **lower one-sided 95% confidence bound**, not the point estimate.
That makes coherence a practical-equivalence statement rather than an eyeball
call.

**Magnitude selection rule — fixed: `largest_passing`.** The −0.05
noninferiority requirement on the lower one-sided bound, plus the absolute 0.85
well-formed floor, already supplies the safety margin. `alpha_star` is the
largest member of the sealed passing set. `alpha_secondary`, when present, is
the largest passing grid value below it. If only one magnitude passes, it
becomes `alpha_star` and the secondary arm is omitted.

The calibration result must partition the full swept grid into disjoint passing
and failing sets; every failing magnitude has a reason. G15 rejects an incomplete
partition, a secondary magnitude that did not pass, or any copied value that
disagrees with the executable calibration result.

- `α★` → calibrated and frozen
- `α_secondary` → calibrated and frozen or null
- `alpha_selection_rule` → `largest_passing` (already fixed)

If every magnitude beyond 0.02 destroys coherence, that itself is the
result — it defines the usable extent of the action space. If several remain
coherent and none produce outcome diversity, the negative is correspondingly
stronger: *no outcome-distinct intervention was found over the predeclared
coherent magnitude range.*

---

## 4a. The writable boundary — **L3_24, frozen**

"Writable boundary" alone is underspecified: it does not say whether the
perturbation is applied to the final prompt position, all prompt positions or
the current decode token; once or on every generated token. All five fields
below are required.

```json
{
  "loop_index_zero_based": 2,
  "paper_loop_label": "L3",
  "layer_index_zero_based": 24,
  "boundary_semantics": "post-layer residual output",
  "first_affected_cache_slot": [2, 25],
  "position_scope": "final_nonpadding_prompt_position_only",
  "application_frequency": "one_shot_during_prefill",
  "delta_broadcast_rule":
    "Per branch, add delta only at the final non-padding prompt position at the
     L3_24 post-layer residual boundary. d has unit RMS over the hidden
     dimension; r is the pre-intervention per-position residual RMS;
     delta = sign * alpha * r * d. All other positions receive zero. Not
     repeated on generated tokens or at later loop/layer visits."
}
```

Validated by: V4 loop-targeted perturbation at physical layers 24/36/47; the
suffix-recompute splice measured at zero-based loop 2, layer 24; the first
affected cache slot empirically established as `(u, ℓ+1)`; the K = 8 splice
measured at 55% fewer layer passes than eight full perturbed prefills.

Chosen because it is already writable under validated machinery, because the
splice was measured there including the K = 8 geometry O1 uses, because it sits
after the large first-pass coordinate transition, and because roughly one and a
half recurrent passes remain downstream to be affected. Deeper boundaries save
more compute but leave less computation to influence.

**This is an instrument choice, not a claim that L3_24 is the model's optimal
control locus.** A negative result is scoped to this boundary and this action
family and must be reported that way.

## 4b. The four structured axes — **the four geometries the programme separated**

Not a divergence contest among historical directions: that would make the bank
uninterpretable. One canonical axis per geometry the work already established as
distinct.

| Axis | Represents | Construction (frozen pre-O1 material only) |
|---|---|---|
| `d₁` readable process-quality | what an external observer can read | leakage-clean locus-native L3_24 readout; corrected source-item-disjoint training material; fit at L3_24 only; standardized/PCA coefficients mapped back into raw 2048-d residual space; unit RMS |
| `d₂` empirical outcome | where successful and unsuccessful trajectories differ | `unitRMS(μ_correct − μ_incorrect)` at L3_24, task-balanced class means, first singular vector of the task-bootstrap mean-difference ensemble |
| `d₃` writable transport | what the validated branch mechanism naturally carries | leading principal direction of the centered exact-protocol S1/S3 injection-delta bundle at L3_24, from the existing bundle and never from O1 outcomes |
| `d₄` learned actuator | what light training actually learned to write | both independently trained adapters run on a sealed pre-O1 reference set; induced L3_24 residual update at the final prompt position; leading PC each; sign-aligned; cosine ≥ 0.90 required; averaged; unit RMS |

These are exactly the separation the geometric audit found:
`readout ≠ empirical outcome ≠ writable transport ≠ learned actuator`. O1 then
asks the clean question: *does either sign of any of these four already-existing
geometries expand verifier-correct reachability beyond matched sampling?* That
is far more interpretable than four directions selected for producing the most
calibration divergence.

**One caveat on `d₄`.** The published 0.951 is a cosine between adapter *proxy*
directions, not between leading PCs of induced L3_24 updates. A reconstruction
below 0.90 is evidence about the extraction method, not necessarily about the
adapters disagreeing. Pre-declare which reading applies before sealing, so a
reconstruction mismatch is not silently promoted to a scientific finding.

**Duplicate rule only: `|cos(dᵢ, dⱼ)| < 0.98`.** The blanket 0.8 cap of v1.1 is
removed. Readable and outcome directions are expected to overlap, and excluding
one because their angle is scientifically non-random would destroy the design's
interpretation. The Gram-matched random bank already controls for correlation.
If two axes exceed 0.98, replace the later with the next singular direction from
the same provenance family; if none exists, do not seal.

**No cross-locus transport.** Every axis must exist or be reconstructed at
exactly L3_24 in raw 2048-dimensional residual coordinates. No L4 vector reused
at L3, no concatenated 24/36/47 tap weights, no layer-36 projection, no pooled
four-loop direction used as a single-loop residual direction, no silent
substitution for a missing artifact. If any of the four cannot be produced from
frozen pre-O1 artifacts, **do not seal** — run a bounded axis-reconstruction
preflight first.

Freeze: four tensor hashes, a construction manifest per vector, the 4×4 Gram
matrix, norms before and after normalization, pairwise cosines, the reference
task manifest, and all reconstruction code hashes.

## 4c. Transport must be measured, or a null cannot be read

Every intervention row carries `downstream_delta_rms`: the measured downstream
residual change against its paired baseline. Required, not optional.

`d₂` is the direction where correct and incorrect trajectories differ, and the
two-null audit found it is *not* cleanly inside the writable span. If injecting
along `d₂` produces near-zero downstream change, the correct reading is "this
axis does not move the computation," not "there is no headroom." Without a
transport measurement, three distinct situations collapse into one reading of
"no outcome diversity":

1. the delta never propagated
2. it propagated but changed no outcome
3. it changed outcomes unhelpfully

`transport_summary()` reports median and quartile transport RMS and a dead rate
per signed action, so a per-direction null decomposes. It is the per-direction
analogue of the bank-level outcome-diversity check, and the zero-α arm must show
transport of exactly zero.

**Prior transport prediction, frozen before O1.** Appendix J.2 already measured
readable/outcome alignment with the writable span at the only loci where both
objects existed. Across L4_24, L4_36 and L4_47, readable↔writable angles were
87.73–89.38° and outcome↔writable angles were 87.69–89.61°, against writable-span
null p05 values of 87.06–87.53°. The overlaps were therefore at chance at every
reported rank. This is not a direct measurement at L3_24, but it is a quantitative
prior that d1 and d2 may produce little downstream transport. d3 should transport
by construction because it is the leading direction of the validated injection
deltas; d4 remains unknown.

The complete eight-action bank is retained. Removing d1/d2 would erase a direct
test of the readout–control boundary. Calibration estimates q_disc under the full
bank, so power remains correctly sized even if only d3/d4 are live; the expected
consequence is lower discordance and a larger required G. The confirmatory report
must state the observed number of transporting signed actions and axes. A null on
a mechanically dead d1 or d2 is interpreted as readout/outcome geometry lying
outside the usable writable span, not as evidence that the broader action space is
barren. A live d3 with null reachability is the stronger action-space result.

`axis_transport_profile()` aggregates the signed-action diagnostics without
changing the primary endpoint. It labels an axis `DEAD_BOTH_SIGNS` only when all
rows for both signs remain at or below the frozen dead threshold, and reports the
number of observed transporting signed actions and axes. This is diagnostic and
may not replace or rescue a null primary.

**Stream assignment: sealed cyclic Latin square.** Per-direction effects are
secondary endpoints, so binding `d₁` permanently to stream 0 would confound every
one of them with a single stream index. Instead

```
stream(a, g) = (a + r_g) mod 8,   a = 2·(direction_id − 1) + (0 if sign=+1 else 1)
```

with `r_g` the task's rank within its stratum in sealed manifest order, mod 8.
Every signed action visits every stream index once per block of eight tasks;
imbalance is at most one when a stratum is not divisible by eight. Same schedule
for the random bank. Enforced by gate G13.

---

## 5. CRN seed-matrix schema

Every intervention continuation is paired index-to-index with its own
zero-intervention baseline continuation sharing prompt, prefix state, sampling
seed and stream, temperature, top-p, token budget, stopping rule, verifier, and
continuation compute. The only difference is the intervention.

The master seed is fixed at `20260728`. The per-task streams are derived by the
single frozen scheme `sha256_domain_uint64_be_v1`:

```
seed(g,i) = uint64_be(
  SHA256(
    "O1_STREAM_SEED_V1\0" || uint64_be(master_seed) ||
    uint32_be(len(UTF8(task_id_g))) || UTF8(task_id_g) || uint8(i)
  )[0:8]
)
```

`build_seed_matrix.py` writes every task/stream seed explicitly. G16 regenerates
the matrix from the master seed and task manifest, so a globally shifted or
post-selected matrix fails even when all arms remain internally CRN-paired. The
same `seed(g,i)` is used at `stream_index=i` in every arm; G4 checks the rowwise
pairing and G16 checks agreement with the sealed derivation.

Canonical signed-action order before the per-task Latin-square rotation:

| action index `a` | structured / random action | paired baseline / zero-alpha action |
|---|---|---|
| 0 | +α·d₁ | no intervention |
| 1 | −α·d₁ | no intervention |
| 2 | +α·d₂ | no intervention |
| 3 | −α·d₂ | no intervention |
| 4 | +α·d₃ | no intervention |
| 5 | −α·d₃ | no intervention |
| 6 | +α·d₄ | no intervention |
| 7 | −α·d₄ | no intervention |

For task `g`, action index `a` is assigned to stream `(a+r_g) mod 8` under the
sealed Latin square in §4c. Baseline and zero-alpha use the same eight stream
indices and seeds, without an intervention.

Eight **distinct** streams per bank. Opposite signs of the same axis do **not**
share a stream in the primary bank — see §6.

A sign-symmetry diagnostic (`+α·d_k`, `0`, `−α·d_k` under one shared stream) may
be run on the *calibration* cohort. It must not define the confirmatory
max-over-8 endpoint.

---

## 6. Why opposite signs may not share a baseline stream

If `+α·d_k` and `−α·d_k` shared one stream and both were compared against the
same zero-intervention continuation, the baseline bank would hold **four**
unique continuations across eight rows. Its `max_i` would be a best-of-4 while
the intervention bank's `max_i` remains a best-of-8. The primary endpoint would
be biased **upward** by a pure sample-size mismatch invisible in the row count.

This is the same species as the two mechanisms found in the project-wide audit:
an effective-sample mismatch produced by how rows were constructed, not by how
they were analysed. It is recorded here so nobody re-derives the shared-stream
scheme later as an efficiency saving.

Gate **G3_STREAM_UNIQUENESS** requires eight distinct `stream_index` values and
eight distinct seeds within every `(task, arm)` bank, and reports the effective
bank size when it fails.

---

## 7. Coupling-survival logging

Common random numbers reduce variance when the paired outcomes are positively
coupled; negatively correlated pairs can make them worse, so the claim that CRN
"is never worse" — asserted in v1.0 — is false as stated. Whether pairing helps
here is an empirical question, and the coupling-survival curve is how it is
answered. At temperature 0.7 with
top-p 0.95, a small logit shift can reorder the nucleus at the first token,
after which the streams are aligned by position only, not by context.

For every paired branch, log

```
T_div = min{ t : token_int[t] ≠ token_base[t] }        (null if no divergence)
```

and report, by magnitude and direction:

- `P(no divergence before termination)`
- median and quartiles of `T_div`
- divergence within the first 1, 8, 32 tokens
- generated length after divergence
- bank-level outcome discordance
- the survival curve `S_α(t) = P(T_div > t)`, with the zero-α arm as reference

`T_div` is **diagnostic only**. No row is ever filtered on it, and no endpoint
conditions on it.

---

## 8. Full-bank zero-α null and halt tolerance

A pass/fail parity check answers *is the plumbing exact*. The full-bank null
answers something more useful: *what does the primary endpoint read under a
known-null intervention, through the entire pipeline including whatever bf16
nondeterminism survives in decode*. It is the direct analogue of the
shuffled-label controls used elsewhere in the programme.

The zero-α arm runs the complete pipeline: eight distinct seed streams, eight
ordinary baseline continuations, eight zero-perturbation splice-path
continuations, identical `max_i`, identical malformed rule, identical analysis
code.

```
H₀ = mean_g [ Y_g(zero_alpha, splice) − Y_g(baseline) ]
```

Report `H₀`, token-identity rate, first-divergence distribution, `n₁₀⁽⁰⁾`,
`n₀₁⁽⁰⁾`, exact paired test, well-formedness difference.

**Mechanical parity comes first.** A bank-level difference of zero can coexist
with massive but *symmetric* divergence, so the endpoint-level `τ` cannot be the
primary check on a path documented as bit-exact. Rowwise, against the
index-paired baseline:

- token identity rate ≥ `zero_alpha_null.min_token_identity_rate` (default 1.0)
- derived-`R` identity rate = 1.0
- equal generated lengths, equal termination status

`first_divergence` is **computed inside the analysis** from
`generated_token_ids`. Any upstream-supplied divergence field is ignored. If the
confirmatory implementation has documented unavoidable decode nondeterminism,
the identity threshold is calibrated and frozen separately — it is not replaced
by `τ`.

**Halt rule** — the confirmatory run stops for pipeline repair if any of:

- rowwise token identity below the sealed threshold
- derived-`R` identity below 1.0
- `|H₀| > τ`, with `τ = 0.01`
- exact McNemar p < 0.05 on the zero-α table (asymmetric discordance)
- the bootstrap interval on `H₀` excludes zero

**`H₀` is never subtracted from the primary.** A nonzero floor may be reported
as numerical noise only when it is small, symmetric, and inside `τ`. A +0.02
directional null is not an innocent calibration constant — it would threaten
the interpretation of a +0.03 or +0.05 intervention effect, which is exactly why
the halt fires instead.

Runs before the primary. Never dropped for budget.

---

## 9. Exact McNemar power

Let

```
p₁₀ = P(Y_int = 1, Y_base = 0)      p₀₁ = P(Y_int = 0, Y_base = 1)
H_reach = p₁₀ − p₀₁                 q_disc = p₁₀ + p₀₁
```

Power depends on `q_disc`, not on task count alone. `q_disc` must be estimated
**at bank level under the CRN geometry** — not per-branch, not under independent
sampling, not from a smaller provisional bank. No variance-reduction benefit is assumed: the sign and magnitude of the CRN
covariance are empirical. The number used for sizing must nevertheless come
from the actual paired-bank geometry, because an independent or smaller-bank
estimate is a different design and can give the wrong `G`.

`G` is set by an **exact finite sum**, not by simulation and not by a normal
approximation. v1.0 used Monte Carlo, whose answer moved with its own `n_sim`
default (319/811/2206/334 at 8,000 draws; 321/815/2238/336 at 20,000) — a
preregistered sample size cannot depend on a simulation setting. Since

```
N_disc ~ Bin(G, q),   N10 | n ~ Bin(n, π),   π = (q + δ) / (2q)
```

and `k_crit(n) < n//2` always, directional power is exactly

```
Power(G; q, δ) = Σ_n P(N_disc = n) · P(N10 ≥ n − k_crit(n) | n)
```

| q_disc | target H_reach | exact minimum G (80% directional, α = 0.05) |
|---|---|---|
| 0.25 | +0.08 | 324 |
| 0.25 | +0.05 | 817 |
| 0.25 | +0.03 | 2238 |
| 0.10 | +0.05 | 337 |

Deterministic and reproducible across calls. Illustrative only: the sealed `G`
is recomputed from the calibrated `q_disc` before confirmation opens.

**Size from a conservative upper bound on `q_disc`, not its point estimate.**
Calibration estimates `q_disc` with error, and greater symmetric discordance
demands more tasks for a fixed directional difference. The manifest seals a
predeclared one-sided upper confidence bound and `required_G` is evaluated
there.

**Practical-effect rule, declared in advance:**

- **+0.03** — scientifically interesting; worth replication and publication
- **+0.05** — sufficient to justify a serious binding/training experiment
- **+0.08** — strong headroom

Power for +0.05 unless the budget rule below binds. A statistically positive
+0.03 is not ignored, but does not by itself trigger a large external compute
request.

---

## 10. Deterministic budget fallback

Sealed **before** calibration opens, so the decision is not made while looking at
numbers one would rather not be looking at.

```
G_confirm = min(G_0.05, G_max)
```

`G_0.05` from §9 using calibrated `q_disc`. `G_max` from measured throughput
against the frozen wall-clock budget.

Frozen before calibration:

- maximum wall-clock budget: **7 days**
- calibration is **outside** the seven-day confirmatory budget
- rerun/failed-shard reserve: **15%**
- minimum completed-task fraction: **1.0** — resume missing sealed shards; failure to finish may correlate with length, malformedness or difficulty
- after interruption, resume only missing sealed shards with identical hashes and streams; never replace tasks and never analyse a partial cohort

If `G_0.05 ≤ G_max`: run powered for +0.05.

If `G_0.05 > G_max`, in this fixed order:

1. drop the random-antipodal bank from confirmation
2. drop the secondary magnitude from confirmation
3. retain only paired baseline vs structured @ `α★`
4. run `G_max`
5. compute and publish the budget-constrained MDE `δ_budget`
6. make **no claim excluding effects smaller than `δ_budget`**

The random bank and second magnitude remain available as calibration or later
secondary experiments. **The zero-α null cannot be dropped** — it validates the
primary comparison itself.

*Note on throughput:* Ouro at four loops costs roughly 4× a dense 2.6B forward
pass. Deep-boundary suffix sharing recovers part of it but does not make
thousands of long generations cheap. `G_max` must come from measured
candidates/hour on the calibration cohort, not from an estimate.

---

## 11. The single primary endpoint

```
H_reach = (1/G) Σ_g [ max_{i≤8} R(structured, α★)_gi − max_{j≤8} R(baseline)_gj ]
```

with:

- structured action bank, four antipodal axis pairs
- primary magnitude `α★`
- **both strata pooled**
- malformed = 0
- eight distinct, index-paired sampling streams
- **K-matched and maximum-token-matched**, which is what the design actually
  guarantees: arms are matched on bank size, max generation tokens, decoding
  settings and seed streams. Realized generated lengths and suffix-recompute
  costs can still differ, so realized layer-token work is recorded and reported
  separately rather than claimed as equal.
- the full confirmatory cohort

Inference:

- paired effect estimate
- exact McNemar test (two-sided, conditional binomial)
- 95% interval from paired task bootstrap, 10,000 draws, percentile — matching
  the clustered-resampling convention used throughout the programme
- discordant cells `n₁₀` and `n₀₁` reported explicitly
- **one-sided 95% bounds from an exact Clopper-Pearson bound on the discordant
  cell, not from the bootstrap.** When every paired difference is identical
  (`n_disc = 0`) the percentile bootstrap is degenerate and reports zero width
  at any `G`, which would manufacture a spurious equivalence verdict. Since
  `H_reach = p₁₀ − p₀₁ ≤ p₁₀`, the exact upper bound on `p₁₀` is a valid,
  conservative, never-degenerate bound. At `n₁₀ = 0` it gives +0.072 for G = 40
  and +0.0075 for G = 400 — which is the difference between `INCONCLUSIVE` and
  a defensible `PRACTICALLY_NULL`.

One primary endpoint. Two strata × two magnitudes × two malformed conventions ×
two contrasts would be sixteen cells, and no interpretation matrix survives
that.

---

## 12. Secondary endpoints and opening order

Opened only after the primary is computed and its hash recorded. Enforced in
code by `Seal.open_secondary`, which raises `SealViolation` otherwise.

1. secondary magnitude (`α_secondary`)
2. hard-medium stratum alone
3. medium stratum alone
4. random antipodal basis contrast
5. selection headroom `H_sel` within the structured bank
6. bank-level well-formedness difference
7. correctness conditional on well-formedness — **descriptive only**, since it
   conditions on a post-intervention outcome and is not an unbiased causal
   contrast
8. per-direction and per-sign effects

**No multiplicity-adjusted fishing for a replacement headline if the primary is
null.** A null primary is reported as a null primary.

The structured-vs-random contrast is secondary by design. The primary
comparison is structured intervention oracle vs matched **ordinary-sampling**
oracle; random directions distinguish a meaningful property of the chosen basis
from generic disruption that merely adds diversity.

**The random basis must match the structured Gram matrix, not be orthogonal.**
Four orthogonal random axes are not a fair control when the structured axes are
correlated — `max` over eight rows behaves differently under different pairwise
geometry. Given normalized structured axes `D` with `G_D = DᵀD = LLᵀ`, sample a
random orthonormal ambient basis `U` and set `R = ULᵀ`; then `RᵀR = LUᵀUL ᵀ =
LLᵀ = G_D`, so the random axes carry the same pairwise geometry while being
randomly oriented in the ambient space. One predeclared draw defines the matched
control; further draws form a secondary null distribution if compute permits.

---

## 13. Malformed-output convention

```
R_gi = 1  iff  (well_formed AND verifier_correct)      otherwise 0
Y_g   = max_{i≤8} R_gi
```

This is a **deliberate reversal** of the convention used for pre-answer
correctness AUROC, where malformed candidates are excluded from the label set
because folding them in lets a classifier solve "did this exhaust the budget"
instead of "is this correct."

Here the opposite is right: a continuation that never commits is a failure of
the intervention, and the question is whether the intervention produces a
*usable correct answer*. The reversal is stated explicitly because a reader who
knows the earlier convention will otherwise flag it.

Enforced by gate **G6_MALFORMED_CONVENTION**: `R` is always derived, never
trusted from upstream, and a stored `R` disagreeing with the derivation halts
the analysis.

The overall convention is primary. Well-formed-only analysis is secondary and
labelled descriptive.

---

## 14. Executable freeze chronology

The manifest cannot carry its own hash. v1.5 therefore uses two pre-outcome
commitments and one result seal.

### 14.1 Calibration precommit

Before any calibration continuation exists:

1. fill the pre-calibration design and real runtime-artifact hashes;
2. create the calibration task manifest;
3. run `calibration_precommit.py`;
4. record the resulting hash in every calibration row and in
   `CALIBRATION_METADATA.json`.

The command refuses to overwrite a prior precommit or to run after the declared
calibration-record path exists.

### 14.2 Executable calibration and final manifest

`calibration_analysis.py` derives the difficulty settings, alpha partition,
selected magnitudes, discordance upper bound, throughput, arm-drop plan, and
sample-size fields. `apply` copies those outputs into a new final manifest and
also records the calibration precommit, raw-record, metadata, and result hashes.
The confirmatory analysis recomputes the calibration and rejects any mismatch.

### 14.3 Runtime provenance and pre-generation seal

Before any confirmatory continuation exists:

1. reconstruct and verify the complete L3_24 axis package;
2. create the final confirmatory task manifest;
3. derive the deterministic seed matrix with `build_seed_matrix.py`;
4. hash the actual generation module, checkpoint, tokenizer, prompt, parser,
   verifier, structured tensor, and random tensor with
   `build_run_provenance.py`;
5. create `PREGENERATION_SEAL.json` with `pregeneration_seal.py`.

The pre-generation command first verifies the complete manifest, recursive
runtime configuration, axis package, calibration recomputation, task-content
disjointness, and deterministic seed matrix. It refuses to run when the
confirmatory-record path already exists. Every confirmatory row carries the
committed task-content hash, runtime-provenance hash, and pre-generation-seal
hash.

### 14.4 Result seal

`run_primary` consumes and verifies the independent pre-generation seal before
opening records. It then reruns provenance, axis, calibration, and record gates;
runs the exact zero-alpha parity gate; and writes an append-only `RESULT_SEAL`.
A second primary, an existing seal path, a halted zero-alpha run, or a secondary
requested before the primary is rejected.

The local chain is not a proof of chronology against a self-adversarial operator.
Because confirmatory records do not exist when `PREGENERATION_SEAL.json` is
created, they cannot be included in that seal; an operator controlling the local
filesystem could later rewrite old records to carry a newly created seal hash, or
run several internally valid sealed attempts and disclose only the best one.
Therefore, before the first confirmatory branch, the SHA-256 of
`PREGENERATION_SEAL.json` must be committed to an external, non-retroactively
editable record (a pushed repository commit or independent timestamp) and every
sealed attempt must be entered in the append-only experiment ledger. The result
artifact must cite that commitment. Without it, the executable package proves
internal consistency only, not historical precedence or completeness of attempts.

---

## 15. Runtime and axis artifacts

The resolved runtime configuration is one canonical nested object containing the
code, model, generator, decoding, difficulty, action-space, bank, cohort,
transport, zero-alpha, and artifact-hash subtrees. G14 compares it recursively;
a run at another layer, with another normalization, prompt, parser, answer
marker, generator knob, or dirty generation module cannot pass by agreeing on a
short selected-key list.

`RUN_PROVENANCE.json` records the actual artifact paths and hashes plus:

- generation-module SHA-256;
- repository-clean requirement;
- repository diff SHA-256;
- checkpoint file/tree hash;
- tokenizer, prompt, parser, verifier, structured-axis, and random-axis hashes.

The axis package requires:

```text
axes_l3_24.npy
random_axes_l3_24.npy
d4_components_l3_24.npy
AXIS_MANIFEST.json
GRAM_MATRIX.json
RECONSTRUCTION_ATTESTATION.json
axis_reconstruction.py
verify_axis_artifact.py
SHA256SUMS
```

The verifier checks shape, finiteness, unit RMS, exact order and locus,
near-duplicate axes, stored/recomputed Gram and cosine matrices, deterministic
fixed-domain random-bank regeneration, canonical row hashes, d4 reconstruction
from both supplied components, complete package hash coverage, verifier identity,
reconstruction-script identity, and the nine provenance fields per axis. It
reports structural verification separately from source-run claims that remain
attested unless the source artifacts or a hash-verified reconstruction log are
included.

---

## 16. Executable analysis and verification suites

The package is flat-layout safe and contains:

- `o1_analysis.py` — complete manifest/provenance/preseal verification, G0–G21
  record gates, exact McNemar inference, exact Clopper–Pearson practical-null
  bound, descriptive paired bootstrap, transport/coupling diagnostics, exact
  power and budget calculations, zero-alpha parity, and append-only result seal;
- `calibration_analysis.py` — deterministic calibration derivation and final
  manifest binding;
- `calibration_precommit.py` — pre-calibration commitment;
- `build_seed_matrix.py` — deterministic stream derivation;
- `build_run_provenance.py` — real runtime artifact hashing;
- `pregeneration_seal.py` — preflight and pre-generation commitment;
- `verify_axis_artifact.py` — A0–A17 axis-package verifier.

Verified synthetic/adversarial suites:

```text
53/53 confirmatory and record-integrity fixtures passed
17/17 calibration and calibration-precommit fixtures passed
13/13 cross-artifact, chronology, package-integrity, and bytecode-exclusion fixtures passed
7/7 end-to-end command-line workflow fixtures passed
15/15 axis-verifier attacks/checks passed
Exact planning minima: 324 / 817 / 2238 / 337
```

Run all checks with:

```bash
python run_all_tests.py
```

These tests verify the machinery, not the absent real checkpoint, source axes,
tasks, or calibration data. The experiment remains unsealed until those real
artifacts fill the remaining manifest slots and survive the same pipeline.

---

## 17. Prohibitions after confirmation opens

Once the first confirmatory outcome is read, none of the following may change:

- the task slice, its offsets, or the stratum assignment
- `α★`, the four structured axes, the four random axes, or the bank geometry
- `G`, or the stopping rule
- the malformed convention
- the primary endpoint definition or its inference procedure
- the secondary opening order

And: no re-selection of strata using confirmatory outcomes; no secondary
endpoint before the primary is opened and hashed; no subtraction of `H₀` from
the primary; no filtering of rows on `first_divergence`.

If a defect is discovered mid-run that requires any of these to change, the run
is void and restarts from a fresh sealed cohort. That is cheaper than a result
nobody can defend.

---

## Post-gate routing

A **positive** primary is a statement about what training could bind, and goes to
Jonathan Williams with the measured headroom, its uncertainty, the exact action
bank, and the staged binding experiment (O2 → O3 → O4).

A **`PRACTICALLY_NULL`** primary is a statement about the injectable action
family — that even oracle selection over these boundaries, directions, coherent
magnitudes and difficulties excludes reachability expansion of at least the
practical threshold. An `INCONCLUSIVE` primary is not that statement and must
not be sent as one. That is a
question about action coordinates and fork placement, and goes to Rui-Jie Zhu.
It is not a claim that Ouro has no useful intervention space.

The third outcome — selection headroom without reachability expansion — is
useful to both: local variation exists but not capability beyond best-of-K, so
the opportunity is compute allocation rather than new reachability.

The artifact is the same either way, which is why the gate runs locally and
first, and why no compute request precedes it.
