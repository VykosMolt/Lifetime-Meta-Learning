"""Foundation Learner V0 — local, post-transfer analysis (contract section 14).

The accelerator produces only raw generations, checkpoints and exact verifier
records.  Everything in this package runs LOCALLY after termination, uses
numpy only (no scipy), and touches no GPU.

* ``stats``  — per-family means, macro averages, the clustered bootstrap
               (families with replacement, then episodes within family,
               10,000 replicates, seedable, identical resample indices across
               arms for paired differences), learning curves,
               retention/interference summaries, value-head rank metrics and
               poison/remap robustness.
* ``report`` — plain-markdown + JSON rendering of those numbers.
"""
from __future__ import annotations

VERSION = "0.1.0"

__all__ = ["VERSION"]
