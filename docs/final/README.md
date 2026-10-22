# Final report: the scripts behind its numbers

The final technical report takes every number from files committed in this repository: the ATT&CK file, the AnnoCTR reports, the saved model replies under `replies/`, the GDS results under `analytics/` and the results under `results/`. `evaluate.py` prints most of the results; the scripts below print or write the rest. None of them calls OpenAI or needs Neo4j, and only `report_data.py` writes files.

| Script | Report sections | What it prints or writes |
|---|---|---|
| `graph_counts.py` | II, III | the text side of the graph replayed from the saved replies (chunks, Entity nodes, MENTIONS by target and link method, text USES), the test split's link decisions, dropped items, the STIX identifiers and descriptions on ATT&CK relationships, and the RDF triple counts |
| `graph_metrics.py` | IV | the topology of the ATT&CK projection (Table II), the degree shares in Fig. 3, PageRank, the Spearman correlation, the Louvain communities (Table III) and the FastRP neighbours of Mustang Panda |
| `evaluation_checks.py` | V, VI | the exact-name baseline's false links by name, summary lines about nodes outside the 200 kept, answer sentences with their own citation, stage times, and the baseline's ratings and answer relevancy per question |
| `provenance_example.py` | Listing 1, Fig. 4 | the fact "APT-C-36 uses njRAT" in Turtle, and the MENTIONS edges and answers behind question Q10 |
| `report_data.py` | Figs. 3, 5, 6 | writes the chart data files `data/*.dat` from `results/intervals.csv` and the ATT&CK file |

Run each one from the repository folder in Command Prompt with the project's Python, for example:

```
%USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe docs\final\graph_metrics.py
```

The last line of each run starts with `RESULT`. The report's LaTeX source and PDF are added to this folder when the report is final.
