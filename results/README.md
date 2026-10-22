# Results

Every number below is read from a file in this folder. Each file is written by a script from saved model replies, so rerunning a script gives the same file without calling OpenAI (see the main README, steps 11 and 18–22). Intervals are 95% bootstrap intervals from `intervals.csv`: 10,000 resamples with seed 7, percentile method (the settings are constants in `evaluate.py`), the 24 questions resampled for answer measures (GraphRAG and the baseline on the same draws), the 34 test reports resampled for extraction.

The two systems answer the same 24 multi-hop benchmark questions about 20 test reports (`benchmark/eval_queries.json`). **GraphRAG** answers from the report's text plus facts from the knowledge graph; the **baseline** is the same pipeline without the graph (report text only).

## Files

| File | What it holds | Written by |
|---|---|---|
| `calibration_dev.csv` | the dev-split calibration of the linking threshold and cut-off | `calibrate.py` |
| `benchmark_answers.json` | the 48 answers, with their context, citations, tokens, cost and seconds | `evaluate.py answers` |
| `extraction_test.csv` | extraction and linking on the 34 test reports beside the exact-name baseline, by level and kind | `evaluate.py extraction` |
| `extraction_test_reports.csv` | the same counts per report (used for the intervals) | `evaluate.py extraction` |
| `operations.csv` | per answer: tokens, cost, citations, seconds per stage | `evaluate.py operations` |
| `answer_accuracy.csv` | per answer: the ATT&CK IDs named against the question's answer IDs | `evaluate.py accuracy` |
| `rating_sheet.html`, `rating_key.json` | the blind rating page and its A/B key | `evaluate.py sheet` |
| `ratings.csv` | the rater's 144 scores as saved from the page | the rating page |
| `human_ratings.csv` | the scores joined to the key | `evaluate.py ratings` |
| `ragas_scores.json`, `ragas_scores.csv` | RAGAS scores per answer, with the judge's calls, tokens and cost | `evaluate.py ragas` |
| `hallucination.csv` | per answer: statements and statements the judge found unsupported | `evaluate.py hallucination` |
| `intervals.csv` | the 95% bootstrap intervals | `evaluate.py intervals` |

## Extraction and linking (34 test reports, 469 expert facts)

| Level | System | Precision | Recall | F1 |
|---|---|---|---|---|
| parent | extraction and linking (gpt-6-sol) | 0.756 [0.700, 0.803] | 0.667 [0.605, 0.731] | **0.709** [0.661, 0.752] |
| parent | exact-name baseline | 0.403 [0.308, 0.488] | 0.488 [0.379, 0.584] | **0.442** [0.342, 0.528] |
| sub-technique | extraction and linking (gpt-6-sol) | 0.548 [0.496, 0.598] | 0.535 [0.484, 0.587] | 0.542 [0.498, 0.582] |
| sub-technique | exact-name baseline | 0.313 [0.229, 0.382] | 0.429 [0.310, 0.532] | 0.362 [0.265, 0.443] |

F1 difference (extraction and linking minus the baseline): parent level +0.267 [0.172, 0.363], sub-technique level +0.180 [0.099, 0.262]. A fact is a (report, ATT&CK entry) pair; at parent level a sub-technique counts as its parent technique, as the experts label parent techniques only. The exact-name baseline links every ATT&CK name or alias found in a report's text as a whole word, ignoring case; its most frequent wrong links come from names that are everyday words ("at", "malware", "server", "software", "credentials"; listed by `evaluate.py extraction` on screen). For context only (a different setting): when its systems choose the sentences themselves, the AnnoCTR paper's best document-level F1 is 56.6 for techniques and 47.7 for tactics, from two different setups; given the sentences the experts annotated, its best is 65.7 and 84.2 (Lange et al., LREC-COLING 2024, Table 9).

## Answers: operations

| | GraphRAG | baseline |
|---|---|---|
| citation-ID validity (citations whose ID was in the answer's context; a repeated citation counts each time) | 229 of 229 (222 counting each ID once per answer) | 66 of 66 (46) |
| seconds per question, end to end: median (max) | 4.10 (8.15) | 2.19 (3.38) |
| tokens in per question: median (max) | 13,553 (22,595) | 2,642 (4,744) |
| cost per question at list price: mean (total for the 24) | $0.0304 ($0.73) | $0.0069 ($0.17) |

Both systems meet the design target of a median under 10 seconds end to end. The citation-ID validity check proves each cited ID was in the context; it does not prove the cited text supports the claim.

## Answers: accuracy against the answer IDs

| | GraphRAG | baseline | difference |
|---|---|---|---|
| F1 (mean of 24) | 0.492 [0.344, 0.638] | 0.125 [0.021, 0.250] | +0.367 [0.208, 0.530] |
| recall (mean of 24) | 0.571 [0.402, 0.729] | 0.129 [0.025, 0.258] | +0.442 [0.271, 0.617] |
| precision (mean over the answers naming an ID of the asked kind) | 0.563 (21 answers) | 0.781 (4 answers) | |

An answer names an entry when the entry's ATT&CK ID appears outside the answer's citations (for technique questions a sub-technique ID counts as its parent). An entry named without its ID does not count; the column `named_by_name_only` lists such cases (4 GraphRAG answers, 3 baseline answers). The rule counts the IDs an answer names, not whether it commits to them: the baseline's Q14 answer says it cannot determine which techniques the group uses, then lists eight techniques from the report, three of them reference items, so it earns recall 0.75 and precision 0.375 (the rater scored it Completeness 1), and the baseline's figures above include that credit.

## Answers: blind human ratings (1 to 5)

| Criterion | GraphRAG | baseline | difference | GraphRAG wins / ties / losses |
|---|---|---|---|---|
| Factuality | 3.17 [2.71, 3.67] | 4.63 [4.25, 4.92] | −1.46 [−2.08, −0.79] | 2 / 4 / 18 |
| Completeness | 3.50 [2.92, 4.08] | 1.38 [1.00, 1.83] | +2.13 [1.50, 2.75] | 18 / 6 / 0 |
| Coherence | 4.88 [4.67, 5.00] | 5.00 [5.00, 5.00] | −0.13 [−0.33, 0.00] | 0 / 22 / 2 |
| sum of the three (3 to 15) | 11.54 [10.67, 12.42] | 11.00 [10.54, 11.46] | +0.54 [−0.38, 1.46] | 13 / 1 / 10 |

The exact baseline means are 4.625 (Factuality) and 1.375 (Completeness); the coherence difference is −0.125. One rater (the team's one member, who built the system) scored all 48 answers without knowing which system wrote which; the page showed each question's two answers as A and B in an order drawn with a fixed seed, with citations replaced by numbers. Completeness is the share of the reference answer's items an answer gives. An answer saying that its context lacks the information was scored on what it states, so a true statement of that kind can score Factuality 5 while scoring Completeness 1: 21 of the baseline's 24 answers were rated Completeness 1, and 19 of those 21 Factuality 5, which is why its Factuality is high. Limits: one rater gives no agreement statistic, and the kind of evidence an answer cites (report text or graph facts) can hint at its system.

## Answers: RAGAS (judge gpt-6-astra, reasoning effort low)

| Measure | GraphRAG | baseline | difference |
|---|---|---|---|
| Faithfulness | 0.976 [0.947, 1.000] | 0.987 [0.966, 1.000] | −0.011 [−0.045, 0.021] |
| Answer relevancy | 0.737 [0.655, 0.800] | 0.169 [0.040, 0.310] | +0.568 [0.442, 0.685] |
| Context recall | 0.367 [0.257, 0.479] | 0.051 [0.000, 0.128] | +0.316 [0.210, 0.428] |
| Hallucination rate (unsupported statements, pooled) | 0.018 [0.000, 0.041] (3 of 167) | 0.017 [0.000, 0.041] (2 of 121) | +0.001 [−0.030, 0.032] |

Faithfulness and the hallucination rate measure whether the answer's statements are supported by the answer's own context, not whether they are correct: an answer can be faithful to wrong facts in its context. Answer relevancy scores an answer that declines to answer as 0; 19 of the baseline's 24 answers score 0. RAGAS's verdict step judges each statement without seeing the question (none of the 48 verdict prompts contains it). GraphRAG's unsupported statement in Q15 says the facts answer "the question", and the judge's only reason is that no question was supplied. In Q11 the judge gives that reason plus a second one: that the context identifies the campaign's malware as PeppyRAT rather than CrimsonRAT (the reference answer lists both, S0115 Crimson and S0643 Peppy). The judge and the answering model are both OpenAI GPT-6 models, and judge models can favour text like their own, so RAGAS is reported as the judge's estimate beside the human ratings and the ID-based accuracy. The judge cost $14.17 at list price for the 48 answers.

## Reading the results together

Measured: GraphRAG answers far more of each question than the baseline (Completeness, answer accuracy recall, answer relevancy and context recall are all higher, and none of their difference intervals includes zero), both systems stay close to their own context (faithfulness about 0.98), and the overall human score, the sum of the three criteria, does not separate the two systems (+0.54 [−0.38, 1.46]).

Interpretation: because GraphRAG's statements are supported by its context while its Factuality is lower, its wrong items most likely come from the context itself, that is, from the knowledge graph the extraction step built (parent-level precision 0.756), whose links the summary lines count. Q01 is an example: GraphRAG names M1018 and M1026 as the top mitigations, the judge finds both statements supported by the context (faithfulness 1.00), and the reference answer is M1047. The baseline rarely answers, so it rarely states anything wrong.
