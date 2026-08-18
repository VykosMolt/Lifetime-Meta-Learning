# Verifier confirmation pass (commit e27edd3) — what stands verified

Independent verifier, 2026-08-09/10, pristine shallow clone of the pushed
branch (immune to concurrent worktree edits). These items are VERIFIED with
the verifier's own primary evidence and do NOT need re-verification unless
the relevant files change again:

- Full suite on pristine clone: 1646 passed / 8 skipped / 0 failed; aggregate
  runner ALL PASS (unit+hostile, dress rehearsal, package checksums), exit 0.
- SHA256SUMS exact coverage 255/255 on the committed tree; zip
  439fe91545b5de26429149eaf433b368274df3bf4a4b1eeca3a51bea6265bbd9
  (506,831,556 B) = sidecar = filled manifest; zip content policy verified by
  introspection (git-tracked ∪ pregen; no reports/, no manifest, no .pt, no
  secrets; uniform 1980 dates / 0o644); dirty-tree refusal works (3
  adversarial builds).
- Fresh-clone reproducibility: verifier independently regenerated all pregen
  from code in a shallow clone and rebuilt the zip to the exact same hash;
  regenerated pregen tree byte-identical; SHA256SUMS byte-identical.
- Split chain: assignment unchanged; split manifest digest
  6cf330b47af9406edf8022564ea330f3f164fbaef22618710a11fa3b2f40d9b5
  independently reproduced from the contract text; 12/12 generator source
  hashes match disk.
- Pregen under the new K_seal: 60/60 + 58/58 checksums; sealed shards opaque
  (own decipher + scan: 0 plaintext hits over 5044 probes); cross-split
  instance/rule/family overlap 0/0/0.
- Two-phase sealed gate: 6-step + budget adversarial probe all correct
  (exactly 2 provisional attempts, commit/abort semantics, 0444 results).
  Known-and-declared: deleting the ledger resets the budget (procedural
  protection, Amendment 1).
- CORE_MATCHING production stage: ran in rehearsal, wrote
  flb200.core_matching_report.v1; 7/7 adversarial mismatch drives refused.
- Hint-leak fixture non-vacuity independently proven (reverted
  constraint_rules 1.0.0 semantics in scratch → main assertion fires).
- Label balance on shipped shards: constraint_rules majority 0.5142,
  grammar_classification 0.5071; residual skews (boolean_rule 0.6154,
  graph_edge_semantics 0.6404) now reported via constant-answer floor
  metrics.
- Sealed eval restores the promoted arm (FL3_final, manifest hash recorded),
  commit phase COMMITTED_AFTER_EVALUATION, all 3 sealed families covered.
- Smoke test (GPU): PASS 10/10, bit-identical loss + generation to the
  committed report. Checkpoint tree hash a701f7a7…6889 re-verified.
- O1 worktree add95c8 clean before/after; FL commits touch zero O1 paths.
- make_manifest --check: exactly the 7 declared unresolved fields.

Overall verdict for e27edd3: FAILED — solely because of Defect A (= reviewer
N1, train-mode decode; 1600 cache-incompatibility warnings in one
pristine-commit rehearsal), Defect B (detector MRO blind spot → false record
in every arm result), and minor Defect C (fresh-clone test hard-fail instead
of skip). See REVIEWER_CONFIRMATION_N1.md for the consolidated fix map.

Process note: the audited worktree was mutated mid-audit by the next repair
worker (10 tracked files + 1 new test from 23:43 UTC). The verifier
re-established all evidence on a pristine clone; the in-flight edits
independently corroborate Defects A/B. Any post-fix release requires: new
commit, full re-run, re-release (new zip hash), fresh-clone re-verification.
