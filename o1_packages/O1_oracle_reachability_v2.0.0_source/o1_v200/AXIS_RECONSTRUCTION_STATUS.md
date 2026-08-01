# O1 v2.0 L3_24 axis reconstruction status

The v2 package expects:

1. the accepted real A1 readable source row;
2. the accepted real A2 writable source row;
3. the exact causal `[8,2048]` induced-update matrix;
4. the exact sequence `[8,2048]` induced-update matrix;
5. the unchanged ordered eight-task reference JSONL;
6. complete per-axis provenance.

The assembler embeds these sources and emits the v2 real bank. The verifier
recomputes both mean actuator axes and rejects centering, PC substitution,
reference reordering, source-matrix mutation, copied random banks, wrong locus,
duplicate axes, incomplete hashes, and packaged bytecode.

This source package contains construction code and synthetic adversarial
fixtures. A real axis package is a separate run artifact. The source package
must never imply that code verification is O1 calibration or confirmation.
