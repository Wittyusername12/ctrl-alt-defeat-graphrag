# CTRL ALT DEFEAT — GraphRAG over cyber threat reports

Term project for CS 6338 Knowledge Engineering, Angelo State University. Team CTRL ALT DEFEAT: Shane Waldrop.

The system answers questions about cyber threat reports. A knowledge graph in Neo4j joins MITRE ATT&CK v20.0 (Enterprise) with the AnnoCTR threat reports. A language model answers from retrieved report text plus the graph, and every answer cites the report chunks and graph edges it used. The same pipeline without the graph is the text-only baseline.

**Status:** full build (Phase 5). Every stage runs: ATT&CK loading, extraction, linking and loading, scoring, and question answering with and without the graph. Extraction, linking and scoring take one report or a whole split (`--split dev`, `--split test`). The benchmark (gold triples, evaluation questions) and the scorer are in place.

## Repository layout

| Path | What it is |
|---|---|
| `schemas.py` | every node label, relationship type and property name, defined once and imported everywhere else |
| `schema.cypher` | one uniqueness constraint per node label, and the vector index for chunk embeddings |
| `db.py` | the Neo4j connection, configured from `.env`; every data query names its Cypher version |
| `datafiles.py` | where the data files live, and the SHA-256 check |
| `apply_schema.py` | applies `schema.cypher` and lists the constraints and indexes in place |
| `load_attack.py` | MITRE ATT&CK v20.0: STIX JSON → RDF with rdflib (saved as `out/attack-v20.ttl`) → Neo4j, with count checks |
| `annoctr.py` | the AnnoCTR reports: text, chunks (4,000 characters, 400 shared) and the experts' ATT&CK labels |
| `llm.py` | every OpenAI call (gpt-6-sol, text-embedding-3-small), saved under `replies/` and reused on reruns |
| `extract.py` | the model lists techniques, tactics, groups, software and "uses" relations in each chunk, each with its evidence sentence; code checks every item and counts each reason for dropping one; a reply cut off at 4,000 tokens is asked again once at 8,000; a run stops after 3 chunks in a row without a complete reply and ends with its tokens and their cost at list price |
| `link.py` | links each mention to ATT&CK (exact name, tie-break rules, then embedding similarity above a threshold, for mentions whose model confidence reaches a cut-off) and loads the report, its chunks and the links into Neo4j |
| `calibrate.py` | sets link.py's threshold and cut-off on the 16 dev reports by the rule written before the dev run, from saved replies only; writes `results/calibration_dev.csv` |
| `rag.py` | answers a question from the most similar chunks plus the graph around the entities they mention (ranked by personalized PageRank), citing `[chunk:…]` and `[edge:…]`; the same question without the graph is the baseline |
| `models.py` | Pydantic data models for the graph (ATT&CK entries, relationships, chunks) and for the question-answering payloads (`QueryRequest`, `QueryResponse`); the extraction reply models are in `extract.py` |
| `gold.py` | writes the benchmark: the experts' facts as gold triples, and 24 multi-hop questions with answers computed from those facts and ATT&CK; `--neo4j` recomputes every answer with Cypher in Neo4j |
| `score.py` | precision, recall and F1 of the system's facts in Neo4j against the gold triples, at sub-technique and parent-technique level |
| `benchmark/` | `gold_triples.json` and `eval_queries.json`, written by `gold.py` only (see `benchmark/README.md`) |
| `replies/` | saved model replies and embeddings; committing them makes every rerun identical |
| `results/` | tables computed from saved replies (the dev calibration table) |
| `data/` | the data files, their licences and their SHA-256 list (see `data/README.md`) |
| `scripts/` | `try_openai.py` and `check_setup.py`, the checks for a new machine |
| `tests/` | `pytest` tests |
| `docs/milestones/` | the proposal and design document (PDF and LaTeX sources) |

## What you need

| Item | Version used | Notes |
|---|---|---|
| Windows | 11 | The commands below are for Command Prompt (`cmd.exe`). |
| Python | 3.11 | From python.org. The project installs its packages into its own virtual environment. |
| Git | 2.51 | |
| Neo4j Desktop | 2.2 | One local instance running Neo4j 2026.09.0 with the APOC and Graph Data Science plugins. |
| OpenAI API key | | Models: gpt-6-sol (extraction and answers), text-embedding-3-small (vectors), gpt-6-astra (evaluation judge). |
| Ollama | 0.40 | Only for the local-model demonstration (`qwen3:4b`). |

Keep the repository and the virtual environment outside any folder that OneDrive syncs.

## Setup (Command Prompt)

1. Clone the repository and enter its folder:
   ```
   git clone https://github.com/Wittyusername12/ctrl-alt-defeat-graphrag.git
   cd ctrl-alt-defeat-graphrag
   ```
2. Create the virtual environment from Python 3.11. The path below is the python.org installer's default location; change it if Python 3.11 is installed elsewhere.
   ```
   %LOCALAPPDATA%\Programs\Python\Python311\python.exe -m venv %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag
   ```
3. Install the pinned packages into that environment (`requirements.txt` lists all 120 with exact versions):
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Create `.env` from the template and paste the OpenAI key after `OPENAI_API_KEY=` (the Neo4j values come in step 6):
   ```
   copy .env.example .env
   notepad .env
   ```
5. Check the OpenAI connection (one short structured request and one embedding):
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\try_openai.py
   ```
6. In Neo4j Desktop, create a local instance running Neo4j 2026.09.0, install the APOC and Graph Data Science plugins on that instance, start it, and put its connection URI and password into `.env`.
7. Check Python, the packages, `.env`, Neo4j and its plugins, and the data files:
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\check_setup.py
   ```
   The last line reads `RESULT: OK` when everything is in place.

## Build the knowledge graph (Command Prompt, in the repository folder, with the Neo4j instance started)

8. Apply the schema (the constraints and the vector index; running it again changes nothing):
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe apply_schema.py
   ```
9. Load MITRE ATT&CK v20.0: 1,785 nodes and 20,426 relationships. The script checks its counts at every stage and ends with `RESULT: OK` when they all match:
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe load_attack.py
   ```
10. Extract, link and load a report, then ask a question. With no report named, the scripts use the Phase 2 report `lab52_2020-06-02_mustang-panda-recent-activity-dll` (dev split); with no question, `rag.py` asks which ATT&CK mitigations address the techniques in that report. Saved replies are reused, so a rerun calls OpenAI only for what is new:
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe link.py
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe rag.py
    ```
11. Extract, link, load and score a whole split. Changes are tried on the 16 dev reports; the 34 test reports are run once, after the calibration on dev. `extract.py` stops after 3 chunks in a row without a complete reply, and ends with the tokens of its new replies and their cost at list price; `link.py` gives each report a short summary:
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py --split dev
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe link.py --split dev
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe score.py --split dev
    ```
    `calibrate.py` recomputes the dev calibration from the saved replies alone (no OpenAI call, no Neo4j) and checks that `link.py` holds the chosen threshold and cut-off:
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe calibrate.py
    ```

## Benchmark and scoring (Command Prompt, in the repository folder, with the Neo4j instance started)

12. Check the benchmark: `--check` rebuilds both benchmark files and compares them with the files in `benchmark/`; `--neo4j` also runs each question's Cypher query in Neo4j and compares its answer with the file's. Running `gold.py` with neither option rewrites both files from the data:
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe gold.py --neo4j
    ```
13. Score the system's facts in Neo4j against the gold triples, for named reports or for a whole split (`--split dev` or `--split test`):
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe score.py lab52_2020-06-02_mustang-panda-recent-activity-dll
    ```

## Tests

```
%USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe -m pytest
```
