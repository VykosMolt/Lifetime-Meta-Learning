# O1 v2 axis-bank redesign experiment plan

Status: package/axis construction and preregistration only. No O1 calibration
or confirmatory continuation has been generated.

## Fixed predecessors

- failed reconstruction preservation:
  `ea40a48f7f41fdf60866eb2067033c62dd14be0f`;
- accepted `AXIS_DIAGNOSIS_V2`:
  `38c474ebea3c6173d48d659d3edc0317c36acfd4`;
- the inadequate original d2, rejected shared-PC d4, both centered adapter
  PC1s, and every failed capture tensor remain unchanged and excluded.

## Primary bank

The immutable order is:

1. `A1_READABLE`;
2. `A2_WRITABLE`;
3. `A3_CAUSAL_MEAN`;
4. `A4_SEQUENCE_MEAN`.

The bank has four axes and eight antipodal actions. A3 and A4 are separately
the unit-RMS raw arithmetic means of the exact causal and sequence
`[8,2048]` induced-update matrices. They are reference-conditioned actuator
directions, not universal adapter axes. They are not centered, subjected to
PCA, or averaged together.

## Boundary resolution

The accepted real capture code and parity report use paper physical layer 24
as `model.model.layers[23]`. Therefore v2 freezes:

- zero-based loop 2 / paper L3;
- physical layer 24 / module index 23;
- post-decoder-layer residual;
- final non-padding prompt token;
- one prefill visit only;
- first downstream cache layer index 24 in loop 2.

The carried v1.5.3 value `layer_index_zero_based=24` described a different
physical module and was not retained. Transport is measured before sampling
at the normalized physical L4_47 loop boundary
(`per_loop_hidden_states[3]`).

## Cohorts

The generator is a self-contained frozen extraction of the historical
Horizon Logic propositional generator. It creates symbolic tasks, verifies
them by exhaustive truth table, then renders natural language. Settings are
`rules_2`, `rules_3`, and `rules_4`.

- calibration: first 32 accepted tasks per setting, 96 total;
- confirmatory candidate population: first 800 accepted tasks per setting,
  2400 total, in a disjoint namespace.

The final confirmatory manifest cannot truthfully receive difficulty-band
strata before calibration. After calibration, settings are assigned using
the frozen baseline any-correct-of-8 bands and the required prefix of each
setting is selected in its already-sealed order. No realized confirmatory
outcome enters selection.

## Stop rule

Calibration is prohibited until `CALIBRATION_PRECOMMIT.json` and its containing
Git commit are present on an authenticated external remote. This redesign task
ends after that external commitment and does not itself start calibration.

