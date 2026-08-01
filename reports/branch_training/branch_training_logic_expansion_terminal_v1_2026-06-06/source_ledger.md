# Branch-Training Source Ledger (logic-priority)

BRANCH_TRAINING_DATA_PULL_VERDICT = LOGIC_EXPANDED_READY

Real logic families reachable: 10 (fol_entailment, logical_deduction_bbh, logical_entailment, lsat_analytical_reasoning, lsat_logical_reasoning, lsat_reading, mcq_logical_reading, prontoqa_reasoning, proofwriter_deduction, ruletaker_deduction). Core domains (coding/math/reasoning/alignment) reuse the corecontent_v2 HF cache. Aggressive logic scale (>=20k groups) is guaranteed by synthetic verifier-backed generation in Part C.

## Logic datasets probed

| dataset_name | family | license | row_count | success |
| --- | --- | --- | --- | --- |
| lucasmccabe/logiqa | mcq_logical_reading | cc-by-nc-4.0 | 7376 | True |
| datatune/LogiQA2.0 | mcq_logical_reading | cc-by-nc-sa-4.0 | 63718 | True |
| tasksource/reclor | mcq_logical_reading | non-commercial-research | 4638 | True |
| metaeval/reclor | mcq_logical_reading | non-commercial-research | 4638 | True |
| hails/agieval-lsat-ar | lsat_analytical_reasoning | mit | 230 | True |
| hails/agieval-lsat-lr | lsat_logical_reasoning | mit | 510 | True |
| hails/agieval-lsat-rc | lsat_reading | mit | 269 | True |
| yale-nlp/FOLIO | fol_entailment | cc-by-sa-4.0 | 0 | False |
| tasksource/folio | fol_entailment | cc-by-sa-4.0 | 1001 | True |
| minimario/FOLIO | fol_entailment | cc-by-sa-4.0 | 1004 | True |
| tasksource/proofwriter | proofwriter_deduction | cc-by-4.0 | 585552 | True |
| renma/ProofWriter | proofwriter_deduction | cc-by-4.0 | 600 | True |
| tasksource/ruletaker | ruletaker_deduction | cc-by-4.0 | 480152 | True |
| renma/PrOntoQA | prontoqa_reasoning | apache-2.0 | 500 | True |
| longface/prontoqa | prontoqa_reasoning | apache-2.0 | 0 | False |
| tasksource/logicbench | logicbench_rule | mit | 0 | False |
| Sumit/LogicBench | logicbench_rule | mit | 0 | False |
| tasksource/bigbench | logical_deduction_bbh | apache-2.0 | 1200 | True |
| tasksource/logical-entailment | logical_entailment | mit | 99876 | True |

## Core-domain reuse (from corecontent_v2)

| dataset_name | family | license | diagnostic_only |
| --- | --- | --- | --- |
| openai/openai_humaneval | coding | MIT | False |
| google-research-datasets/mbpp | coding | cc-by-4.0 | False |
| codeparrot/apps | coding | MIT | False |
| open-r1/verifiable-coding-problems-python | coding | apache-2.0 | False |
| openai/gsm8k | math | MIT | False |
| EleutherAI/hendrycks_math | math | MIT | False |
| ChilleD/SVAMP | math | mit | False |
| lucasmccabe/logiqa | logic | cc-by-nc-4.0 | False |
| allenai/ai2_arc | reasoning | cc-by-sa-4.0 | False |
| allenai/ai2_arc::easy | reasoning | cc-by-sa-4.0 | False |
| openbookqa | reasoning | apache-2.0 | False |
| commonsense_qa | reasoning | mit | False |
| ChilleD/StrategyQA | reasoning | apache-2.0 | False |
| Anthropic/hh-rlhf | alignment | mit | False |
| HuggingFaceH4/ultrafeedback_binarized | alignment | mit | False |
| stanfordnlp/SHP | alignment | mit | False |
| PKU-Alignment/PKU-SafeRLHF | alignment | cc-by-nc-4.0 | False |
| openlifescienceai/mmlu_anatomy | anatomy | mit | True |
| allenai/sciq::science | science | cc-by-nc-3.0 | True |
