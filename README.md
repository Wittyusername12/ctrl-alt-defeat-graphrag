# CTRL ALT DEFEAT — GraphRAG over cyber threat reports

Term project for CS 6338 Knowledge Engineering, Angelo State University. Team CTRL ALT DEFEAT: Shane Waldrop.

The system answers questions about cyber threat reports. A knowledge graph in Neo4j joins MITRE ATT&CK v20.0 (Enterprise) with the AnnoCTR threat reports. A language model answers from retrieved report text plus the graph, and every answer cites the report chunks and graph edges it used. The same pipeline without the graph is the text-only baseline.

**Status:** skeleton stage. One report runs through every stage: ATT&CK loading, extraction, linking, loading, and question answering with and without the graph.

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
| `extract.py` | the model lists techniques, tactics, groups, software and "uses" relations in each chunk, each with its evidence sentence; code checks every item and counts each reason for dropping one |
| `link.py` | links each mention to ATT&CK (exact name, tie-break rules, then embedding similarity) and loads the report, its chunks and the links into Neo4j |
| `rag.py` | answers a question from the most similar chunks plus the graph around the entities they mention (ranked by personalized PageRank), citing `[chunk:…]` and `[edge:…]`; the same question without the graph is the baseline |
| `replies/` | saved model replies and embeddings; committing them makes every rerun identical |
| `data/` | the data files, their licences and their SHA-256 list (see `data/README.md`) |
| `scripts/` | `try_openai.py` and `check_setup.py`, the checks for a new machine |
| `tests/` | `pytest` tests |

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
10. Extract, link and load a report, then ask a question. With no report named, the scripts use the test report `lab52_2020-06-02_mustang-panda-recent-activity-dll` (dev split); with no question, `rag.py` asks which ATT&CK mitigations address the techniques in that report. Saved replies are reused, so a rerun calls OpenAI only for what is new:
    ```
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe extract.py
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe link.py
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe rag.py
    ```

## Tests

```
%USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe -m pytest
```
