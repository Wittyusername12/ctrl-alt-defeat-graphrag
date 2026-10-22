# Fresh-clone check (22 Oct 2026)

This check asks whether the repository works from the README alone. On 22 Oct 2026 the repository was cloned from GitHub into an empty folder on the project laptop, its virtual environment was made again from `requirements.txt`, the tests were run, and the whole knowledge graph was built in a new, empty Neo4j database from the saved model replies, following README steps 1–13. Every count and score equals the original runs of 19 Oct 2026, both answers to the tracer question were found among the saved replies, and the rebuild sent no request to OpenAI. Dates are US Central time.

## Setup

| Item | In this check |
|---|---|
| Machine | the project laptop, Windows 11, Command Prompt |
| Folder | `C:\Users\shane\dev\fresh\ctrl-alt-defeat-graphrag`, empty before the clone |
| Commit | `a8f412a` on branch `build` |
| Python | 3.11.9; the virtual environment at the README's path was deleted and made again (README steps 2–3) |
| Packages | the 120 pinned in `requirements.txt`, each installed from pip's local cache |
| `.env` | copied from the working folder, which holds the values README step 4 asks for |
| Neo4j | 2026.09.0, enterprise edition, with APOC 2026.09.0 and Graph Data Science 2026.09.0, in the existing local instance |
| Database | `fresh`, new and empty, named in the fresh folder's `.env` as `NEO4J_DATABASE=fresh`; the working database `neo4j` was not touched |

README step 6 creates a Neo4j instance. Instead, this check created the empty database `fresh` in the existing instance (the enterprise edition holds several databases), from the fresh folder with the project's Python:

```
%USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe -c "import db; driver, _ = db.connect(); driver.execute_query('CREATE DATABASE fresh IF NOT EXISTS WAIT', database_='system'); driver.close()"
```

## Results

| README step | Command | Result line, with the main numbers |
|---|---|---|
| 1 | `git clone` | 1,307 objects received (24.65 MiB) |
| 3 | `pip install -r requirements.txt` | 120 packages installed |
| 5 | `scripts\try_openai.py` | `RESULT: OK`: gpt-6-sol answered in 3.1 s (102 tokens in, 23 out) with a structured reply that matched; the embedding had 1,536 numbers |
| 7 | `scripts\check_setup.py` | `RESULT: OK`: 120 of 120 packages at their pinned versions; Neo4j 2026.09.0 enterprise edition, database `fresh`; APOC and GDS 2026.09.0, 6 of 6 procedures; 124 data files match their SHA-256 |
| Tests | `python -m pytest` | `176 passed in 47.86s` |
| 8 | `apply_schema.py` | `RESULT: OK - every constraint and every vector index are in place` (8 constraints; 4 vector indexes online) |
| 9 | `load_attack.py` | `RESULT: OK - 1785 ATT&CK nodes and 20426 relationships in Neo4j, matching the RDF graph and the counts measured on 18 Oct` (194,611 RDF triples) |
| 9 | `enrich.py` | `RESULT: OK - every ATT&CK node carries pagerank, community and fastrp; results in analytics/` (the saved GDS results loaded: 6 Louvain communities, modularity 0.2953) |
| 11 | `extract.py --split dev` | `RESULT: OK - every chunk has a complete reply; replies saved under replies/extract`: 50 of 50 chunks; 980 mentions and 121 uses relations kept, 10 items dropped; 0 new replies ($0.00) |
| 11 | `link.py --split dev` | `RESULT: OK - every report linked, loaded and read back; embeddings saved under replies/embeddings`: 16 of 16 reports; 980 mentions: 637 exact, 30 embedding, 313 unlinked |
| 11 | `score.py --split dev` | `RESULT: OK - 16 report(s) scored`: parent level precision 0.684, recall 0.690, F1 0.687 |
| 11 | `extract.py --split test` | `RESULT: 1 problem(s)`: 98 of 99 chunks with a complete reply; 1,818 mentions and 178 uses relations kept, 12 items dropped; 0 new replies ($0.00). The problem is the chunk `quoteintelligence_2021-06-30_daily-operations-of-a-nigerian#4`, whose saved reply was cut off at 4,000 tokens and again at 8,000, as in the original run |
| 11 | `link.py --split test` | `RESULT: OK - every report linked, loaded and read back; embeddings saved under replies/embeddings`: 34 of 34 reports; 1,818 mentions: 1,169 exact, 1 exact with the tie broken by category, 43 embedding, 605 unlinked |
| 17 | `score.py --split test` | `RESULT: OK - 34 report(s) scored`: parent level precision 0.756, recall 0.667, F1 0.709; sub-technique level 0.548, 0.535, 0.542 |
| 12 | `scripts\check_summaries.py` | `RESULT: OK - the summary-line Cypher gives the hand-worked values and equals the Python count on every report` (40,864 rows over the 49 reports with ATT&CK entries) |
| 13 | `rag.py` | `RESULT: OK - every citation names a chunk or fact that was in the context; answers saved under replies/answers`: both answers reused from the saved replies; GraphRAG's 19 citations all name IDs in its context |
| — | `git status`, after the rebuild | `nothing to commit, working tree clean` |

## Compared with the original runs

| Measure | Original runs (19 Oct 2026) | This check |
|---|---|---|
| ATT&CK in Neo4j | 1,785 nodes, 20,426 relationships | equal |
| Dev links | 980 mentions: 637 exact, 30 embedding, 313 unlinked | equal |
| Dev, parent level | 215 facts, TP 147, FP 68, FN 66 (`results/calibration_dev.csv`, the row of the chosen τ and c) | equal |
| Test links | 1,818 mentions: 1,169 exact, 1 exact with the tie broken by category, 43 embedding, 605 unlinked | equal |
| Test, parent level | precision 0.756, recall 0.667, F1 0.709 (`results/README.md`) | equal |
| Test, sub-technique level | precision 0.548, recall 0.535, F1 0.542 (`results/README.md`) | equal |
| Summary-line rows | 40,864 over 49 reports | equal |
| Tracer question | 2 chunks (similarity 0.8338 and 0.8033 on Neo4j's 0–1 scale); personalized PageRank on 2,258 nodes and 41,450 relationships; 44 summary lines; 120 fact lines | equal |

Two outputs show that the build is deterministic, not only close:

- **Personalized PageRank.** `rag.py` compares the 200 scores it computes with the scores saved for the same question in the original run (`replies/pagerank/bfb2c38e912b07a983d73a934ae88b51761309ee05302d4a50958d561864bed2.json`) and printed `scores same as the saved scores`: all 200 scores were exactly equal, although the database and its internal node IDs were new (GDS runs with one thread).
- **The saved answers.** `llm.py` names each saved reply by the SHA-256 of the whole request: the model, its settings and every message. Both of the tracer question's answers were found under their saved names, so the prompts built from the new graph (its chunks, summary lines and fact lines) were identical, character for character, to the prompts of the original run.

No request went to OpenAI during the rebuild. After `rag.py`, `git status` in the fresh folder printed `nothing to commit, working tree clean`: no committed file had changed and no new file had appeared. `llm.py` saves every new OpenAI reply and embedding as a new file under `replies/`, so a request would have left a new file behind. The only requests in the whole check were the two that `scripts\try_openai.py` sends: one short chat request and one embedding.

## What the check does not cover

- It ran on the same laptop, Neo4j instance and OpenAI account as the original runs. It shows that a new folder, a new virtual environment and an empty database give the same results, not that a different computer does.
- README step 10 (one report, then the tracer question) was not run on its own: its report is part of the dev split in step 11, and its question is step 13. `calibrate.py` (step 11), the chat page and the local model (steps 14–15), and the benchmark and evaluation steps (16 and 18–22) were not part of this check.
