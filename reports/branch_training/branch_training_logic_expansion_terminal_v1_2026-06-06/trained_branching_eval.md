# Trained Branching Eval (Part O) — Ouro-RLTT vs +SFT adapter

TRAINED_BRANCHING_EVAL_VERDICT = MODEL_INTERNAL_BRANCHING_PARTIAL

Bounded heldout (12/domain, K=3). Macro positive_oracle@K: base 0.708 vs sft 0.688 (lift -0.02). Branch diversity: base 2.17 vs sft 2.62 (lift +0.45).

| domain | base_oracle | sft_oracle | base_div | sft_div |
| --- | --- | --- | --- | --- |
| logic | 0.8330 | 0.7500 | 2.4200 | 2.5800 |
| math | 0.7500 | 0.9170 | 1.9200 | 2.6700 |
| reasoning | 0.9170 | 0.6670 | 2.1700 | 2.6700 |
| coding | 0.3330 | 0.4170 | 2.1700 | 2.5800 |

Proof-of-capability: a 300-step bounded SFT adapter, not a converged model.
