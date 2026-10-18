# CTRL ALT DEFEAT — GraphRAG over cyber threat reports

Term project for CS 6338 Knowledge Engineering, Angelo State University. Team CTRL ALT DEFEAT: Shane Waldrop.

The system answers questions about cyber threat reports. A knowledge graph in Neo4j joins MITRE ATT&CK v20.0 (Enterprise) with the AnnoCTR threat reports. A language model answers from retrieved report text plus the graph, and every answer cites the report chunks and graph edges it used. The same pipeline without the graph is the text-only baseline.

**Status:** setup stage. The two scripts below check a new machine; the pipeline follows in later commits.

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
7. Check Python, the packages, `.env`, Neo4j and its plugins:
   ```
   %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\check_setup.py
   ```
   The last line reads `RESULT: OK` when everything is in place.
