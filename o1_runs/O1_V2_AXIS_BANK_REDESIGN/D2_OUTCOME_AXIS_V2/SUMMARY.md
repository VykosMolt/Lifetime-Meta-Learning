# D2_OUTCOME_AXIS_V2 diagnosis

Verdict: `INSUFFICIENT_STABILITY`.

The exact `s3b2_features.pt` source hash is
`0bf994770ee6e14fa834680d5f621211a2758de92e55d0dc45e5db0926d3f34d`.
It contains 29 positive and 131 negative rows. Positive-task support is:
logic 2, math 2, reasoning 4, coding 0. No Horizon Logic or target-domain
positive exists.

The task-balanced L3_24 mean-difference candidate was diagnosed but is not in
the primary bank. Its task-clustered bootstrap signed cosine q05 is
approximately `0.655658`, below the frozen `0.70` stability threshold;
leave-one-positive-task-out and domain-held-out minima are approximately
`0.898886` and `0.756853`. The complete report enumerates every task, domain,
class row count, contribution, and held-out analysis.

The frozen v2 calibration manifest and confirmatory candidate pool have no
task-ID overlap with s3b2. Canonical content-hash equality is not directly
testable because s3b2 did not store Horizon canonical task JSON hashes. The
source task-ID exclusion list remains mandatory for any later secondary
experiment.

This diagnosis does not modify the v2 bank.

