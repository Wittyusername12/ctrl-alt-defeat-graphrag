# Benchmark: gold triples and evaluation questions

Both files in this folder are written by `gold.py` from the data in `data/` and are never edited by hand. `tests/test_gold.py` fails when either file differs from a fresh build, and `gold.py --check` makes the same comparison from the command line.

| File | What it holds |
|---|---|
| `gold_triples.json` | 682 gold triples: every ATT&CK entry the AnnoCTR experts linked in the 16 dev and 34 test reports |
| `eval_queries.json` | 24 multi-hop questions about the test reports, each with its answer, a reference answer in words, the facts the answer rests on, and a Cypher query that computes the same answer in Neo4j |

## Gold triples

A triple (report, `MENTIONS`, entry) holds when the experts linked at least one mention in the report to the ATT&CK entry. In the knowledge graph the same fact is the path Report `-HAS_CHUNK->` Chunk `-MENTIONS->` entry, which is what `score.py` reads from Neo4j.

| Split | Reports | Triples | Technique | Tactic | Group | Software |
|---|---|---|---|---|---|---|
| dev | 16 | 213 | 128 | 38 | 10 | 37 |
| test | 34 | 469 | 327 | 85 | 13 | 44 |

- **Complete, not a sample.** Every expert link to ATT&CK in these 50 reports is in the file (2,300 linked mentions), so precision can be measured: a system fact outside the file is outside the experts' annotation.
- **Mapped to ATT&CK v20.0.** The experts linked an older ATT&CK. Ten report-level links to T1562 Impair Defenses, which v20.0 revokes, become links to T1685 Disable or Modify Tools (listed under `revoked`). Six links name IDs missing from the v20.0 Enterprise file and are left out (listed under `dropped`, with their mentions): T1629 (dev), T1631.001 (two test reports), T1623.001, T1437 and T1509. All five IDs are ATT&CK Mobile techniques (checked in `mobile-attack/mobile-attack.json` at tag `ATT&CK-v20.0` of <https://github.com/mitre/cti>), not Enterprise ones.
- **Parent techniques only.** No dev or test triple names a sub-technique: the experts labelled parent techniques. `score.py` therefore reports scores at parent level as well as at sub-technique level.
- Each mention keeps AnnoCTR's text, whether the mention names the concept (explicit, AnnoCTR class CE) or only describes it (implicit, class CI), and AnnoCTR's context around the mention.

## Evaluation questions

Each question needs at least two connected facts: a gold fact about a test report plus at least one ATT&CK relationship, or the gold facts of two reports joined through the entries they share. Answers are computed by `gold.py` from the gold triples and the ATT&CK v20.0 file, never from the system's own graph.

**The parent rule.** ATT&CK records many relationships for sub-techniques, while the experts name parent techniques, so a relationship that ATT&CK records for a sub-technique counts for its parent technique. Example: ATT&CK records REvil (S0496) using T1566.001 Spearphishing Attachment, so REvil uses T1566 Phishing. Reference answers name the sub-technique each time the rule is used.

| Pattern | Questions | Path through the graph |
|---|---|---|
| `mitigations_top` | Q01–Q05 | Report → Technique ← (sub-technique) ← Mitigation, counted per mitigation |
| `tactics` | Q06–Q09 | Report → Technique → Tactic |
| `group_software` | Q10–Q11 | Report → Group → Software ← Report |
| `groups_using_report_software` | Q12 | Report → Group → Software ← Report |
| `group_techniques` | Q13–Q14 | Report → Group → Technique (or its sub-technique) ← Report |
| `groups_most_techniques` | Q15 | the same, counted per group |
| `software_techniques` | Q16–Q18 | Report → Software → Technique (or its sub-technique) ← Report |
| `software_techniques_missing` | Q19 | the report's techniques with no such path |
| `shared_techniques` | Q20–Q22 | Report A → Technique ← Report B |
| `groups_all_techniques` | Q23–Q24 | Report → Technique ← (sub-technique) ← Group, for every technique of the report |

- **How the questions were chosen.** By hand, from the gold facts of the test reports (never from system output): ten patterns, all five vendors, 20 of the 34 test reports, and answers of 1 to 8 items, short enough to rate on the 1–5 scale. For the "largest number" questions, the leader is the same whether or not relationships recorded for sub-techniques are counted.
- **Two computations.** `cypher` holds a query that computes each answer inside Neo4j from the ATT&CK part of the graph, with the gold facts passed as `parameters`. `gold.py --neo4j` runs all 24 queries and compares each result with the answer computed in Python.
- **Checked on attack.mitre.org.** On 19 October 2026, while the website showed ATT&CK v20.0 content, the pages behind three answers were fetched and their lists compared with the answers item by item: Q11 (the Transparent Tribe page's software table), Q13 (the techniques on Chimera's page) and Q17 (the techniques on PlugX's page). Each question's `checked_on_attack_website` records the page and what the page lists; `gold.py` stops if a computed answer stops matching.
- `reference_answer` states each answer in words, because RAGAS context recall needs one. `supporting_facts` lists the gold facts and the ATT&CK relationship IDs the answer rests on, written the way answers cite them (`[edge:attack:USES:S0496:T1566.001]`).

## Licence

These files are adapted from AnnoCTR (Lukas Lange, Marc Müller, Ghazaleh Haratinezhad Torbati, Dragan Milchevski, Patrick Grau, Subhash Pujari, Annemarie Friedrich, LREC-COLING 2024, <https://arxiv.org/abs/2404.07765>; Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries; source <https://github.com/boschresearch/anno-ctr-lrec-coling-2024>, commit `d510b6949e1938d47c93a43eedd562dc538439dc`), which is licensed under CC BY-SA 4.0 (full text in `../data/annoctr/LICENSE.txt`, summary at <https://creativecommons.org/licenses/by-sa/4.0/>). The files are shared under the same licence.

Changes from AnnoCTR: the experts' links are grouped into one triple per report and ATT&CK entry; ATT&CK IDs are mapped to v20.0 as described above; ATT&CK names and relationships are added; the questions and their answers are new.

ATT&CK names and relationships come from MITRE ATT&CK®. © 2026 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation (licence: `../data/attack/LICENSE.txt`).

Neither the AnnoCTR authors, the report donors nor MITRE endorse this project. The material is provided as is, without warranties, as Section 5 of the CC BY-SA 4.0 licence states.
