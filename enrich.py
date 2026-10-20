r"""Graph analytics on ATT&CK: GDS PageRank, Louvain communities and FastRP vectors, written onto the ATT&CK nodes.

Run from the repository folder with the project's Python, after apply_schema.py and load_attack.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe enrich.py

The GDS queries are in analytics.cypher. The first run (on the laptop) projects ATT&CK alone, runs the three
algorithms with one thread, and saves the results under analytics/:
  analytics/attack_analytics.csv   attack_id, label, PageRank score and Louvain community of every ATT&CK entry
  analytics/fastrp.npz             the 128-number FastRP vector of every ATT&CK entry
  analytics/analytics_meta.json    when and how the results were computed (GDS version, settings, counts)
Later runs, on any machine, load those files instead of running GDS again (FastRP's random start depends on
internal node IDs, so recomputing elsewhere could change the vectors); --recompute runs GDS again and
overwrites the files. Either way the results are written onto the ATT&CK nodes as the properties pagerank,
community and fastrp, read back, and the FastRP vector indexes are probed.

The last line starts with RESULT. Exit code 0 when every ATT&CK node carries all three results, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from neo4j.exceptions import DriverError, Neo4jError

import datafiles
import db
import load_attack
from schemas import (ATTACK_ID, ATTACK_LABELS, COMMUNITY, FASTRP, FASTRP_DIMENSIONS, FASTRP_VECTOR_INDEX, GROUP, KEY,
                     NAME, PAGERANK)

ANALYTICS_FILE = Path(__file__).resolve().parent / "analytics.cypher"
ANALYTICS_DIR = datafiles.REPO / "analytics"
TABLE_FILE = ANALYTICS_DIR / "attack_analytics.csv"
VECTORS_FILE = ANALYTICS_DIR / "fastrp.npz"
META_FILE = ANALYTICS_DIR / "analytics_meta.json"
GRAPH_NAME = "attack_analytics"
EXPECTED_NODES = sum(load_attack.EXPECTED_NODES.values())   # 1,785 live ATT&CK entries
PROBE_GROUP = "G0129"                                       # Mustang Panda, the tracer report's group
PROBE_WAIT_SECONDS = 60


def queries(text: str | None = None) -> dict[str, str]:
    """The named queries of analytics.cypher: each starts at a '// name:' line; other comment lines are dropped."""
    text = ANALYTICS_FILE.read_text(encoding="utf-8") if text is None else text
    named, current = {}, None
    for line in text.splitlines():
        match = re.match(r"\s*//\s*name:\s*(\w+)", line)
        if match:
            current = match.group(1)
            named[current] = []
        elif current and not line.strip().startswith("//"):
            named[current].append(line)
    return {name: "\n".join(lines).strip().rstrip(";").strip() for name, lines in named.items()}


# ---------------------------------------------------------------- computing (GDS) and the files

def compute(driver, database: str, expected_nodes: int = EXPECTED_NODES) -> dict:
    """Run the GDS queries once; returns the per-entry results and the run's metadata."""
    named = queries()
    db.run(driver, database, named["drop"], graph=GRAPH_NAME)       # a projection left by an interrupted run
    try:
        projection = db.run(driver, database, named["project"], graph=GRAPH_NAME)[0]
        if projection["nodes"] != expected_nodes:
            raise ValueError(f"the projection holds {projection['nodes']} nodes, expected {expected_nodes}")
        pagerank = {row["attack_id"]: row["score"] for row in db.run(driver, database, named["pagerank"], graph=GRAPH_NAME)}
        louvain = {row["attack_id"]: row["community"] for row in db.run(driver, database, named["louvain"], graph=GRAPH_NAME)}
        stats = db.run(driver, database, named["louvain_stats"], graph=GRAPH_NAME)[0]
        fastrp = {row["attack_id"]: row["embedding"] for row in db.run(driver, database, named["fastrp"], graph=GRAPH_NAME)}
    finally:
        db.run(driver, database, named["drop"], graph=GRAPH_NAME)
    for name, values in (("PageRank", pagerank), ("Louvain", louvain), ("FastRP", fastrp)):
        if len(values) != expected_nodes:
            raise ValueError(f"{name} returned {len(values)} nodes, expected {expected_nodes}")
    version = db.run(driver, database, "RETURN gds.version() AS version")[0]["version"]
    meta = {"computed": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gds_version": version,
            "projection": {"nodes": projection["nodes"], "relationships": projection["relationships"]},
            "louvain": {"communities": stats["communityCount"], "modularity": stats["modularity"],
                        "levels": stats["ranLevels"]},
            "settings": "analytics.cypher (PageRank damping 0.85; Louvain defaults; FastRP 128 numbers, iteration "
                        "weights 0, 1, 1, seed 42; every algorithm with concurrency 1)"}
    return {"pagerank": pagerank, "community": renumber(louvain), "fastrp": fastrp, "meta": meta}


def renumber(communities: dict[str, int]) -> dict[str, int]:
    """Community numbers 0, 1, 2, ... from the largest community down (ties: the smallest attack_id first)."""
    members: dict[int, list[str]] = {}
    for attack_id, community in communities.items():
        members.setdefault(community, []).append(attack_id)
    order = sorted(members.values(), key=lambda ids: (-len(ids), min(ids)))
    return {attack_id: number for number, ids in enumerate(order) for attack_id in ids}


def save(results: dict, labels: dict[str, str]) -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(results["pagerank"])
    with TABLE_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ATTACK_ID, "label", PAGERANK, COMMUNITY])
        for attack_id in ids:
            writer.writerow([attack_id, labels[attack_id], repr(float(results["pagerank"][attack_id])),
                             results["community"][attack_id]])
    vectors = np.asarray([results["fastrp"][attack_id] for attack_id in ids], dtype=np.float32)
    np.savez_compressed(VECTORS_FILE, attack_ids=np.asarray(ids), vectors=vectors)
    META_FILE.write_text(json.dumps(results["meta"], indent=1) + "\n", encoding="utf-8", newline="\n")


def load() -> dict:
    """The saved results, as compute() returns them."""
    pagerank, community, labels = {}, {}, {}
    with TABLE_FILE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pagerank[row[ATTACK_ID]] = float(row[PAGERANK])
            community[row[ATTACK_ID]] = int(row[COMMUNITY])
            labels[row[ATTACK_ID]] = row["label"]
    with np.load(VECTORS_FILE, allow_pickle=False) as saved:
        fastrp = {str(attack_id): vector for attack_id, vector in zip(saved["attack_ids"], saved["vectors"])}
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    return {"pagerank": pagerank, "community": community, "fastrp": fastrp, "meta": meta, "labels": labels}


def files_exist() -> bool:
    return all(path.is_file() for path in (TABLE_FILE, VECTORS_FILE, META_FILE))


# ---------------------------------------------------------------- Neo4j

def labels_query() -> str:
    return ("MATCH (n) WHERE " + " OR ".join(f"n:`{label}`" for label in ATTACK_LABELS)
            + f"\nRETURN n.`{ATTACK_ID}` AS attack_id, labels(n)[0] AS label, n.`{NAME}` AS name")


def write_query(label: str) -> str:
    return (f"UNWIND $rows AS row\n"
            f"MATCH (n:`{label}` {{`{KEY[label]}`: row.attack_id}})\n"
            f"SET n.`{PAGERANK}` = row.pagerank, n.`{COMMUNITY}` = row.community, n.`{FASTRP}` = row.fastrp\n"
            f"RETURN count(n) AS written")


def read_back_query(label: str) -> str:
    return (f"MATCH (n:`{label}`)\n"
            f"WHERE n.`{PAGERANK}` IS NOT NULL AND n.`{COMMUNITY}` IS NOT NULL AND size(n.`{FASTRP}`) = {FASTRP_DIMENSIONS}\n"
            f"RETURN count(n) AS n")


def nearest_query(label: str) -> str:
    return (f"MATCH (g:`{label}` {{`{ATTACK_ID}`: $attack_id}})\n"
            f"CALL db.index.vector.queryNodes('{FASTRP_VECTOR_INDEX[label]}', $k, g.`{FASTRP}`) YIELD node, score\n"
            f"RETURN node.`{ATTACK_ID}` AS attack_id, node.`{NAME}` AS name, score")


def rows_by_label(results: dict, labels: dict[str, str]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {label: [] for label in ATTACK_LABELS}
    for attack_id in sorted(results["pagerank"]):
        rows[labels[attack_id]].append({"attack_id": attack_id, "pagerank": float(results["pagerank"][attack_id]),
                                        "community": int(results["community"][attack_id]),
                                        "fastrp": [float(x) for x in results["fastrp"][attack_id]]})
    return rows


def write(driver, database: str, results: dict, labels: dict[str, str]) -> list[str]:
    problems = []
    for label, items in rows_by_label(results, labels).items():
        written = sum(db.run(driver, database, write_query(label), rows=batch)[0]["written"] for batch in db.batches(items))
        found = db.run(driver, database, read_back_query(label))[0]["n"]
        if written == len(items) == found:
            print(f"  ok       {label}: {found} nodes carry {PAGERANK}, {COMMUNITY} and a {FASTRP_DIMENSIONS}-number {FASTRP}")
        else:
            problems.append(f"{label}: {written} of {len(items)} written, {found} read back")
            print(f"  PROBLEM  {problems[-1]}")
    return problems


def probe(driver, database: str, attack_id: str = PROBE_GROUP, k: int = 6) -> tuple[list[dict], list[str]]:
    """Search the Group FastRP index with one group's own vector; the index works when it finds that group first."""
    rows: list[dict] = []
    deadline = time.monotonic() + PROBE_WAIT_SECONDS
    while time.monotonic() < deadline:   # the index takes in new vectors in the background
        rows = db.run(driver, database, nearest_query(GROUP), attack_id=attack_id, k=k)
        if rows and rows[0]["attack_id"] == attack_id:
            return rows, []
        time.sleep(2)
    return rows, [f"the {FASTRP_VECTOR_INDEX[GROUP]} index did not return {attack_id} first "
                  f"(returned {rows[0]['attack_id'] if rows else 'nothing'})"]


# ---------------------------------------------------------------- main

def summary(results: dict, names: dict[str, str]) -> None:
    meta = results["meta"]
    print(f"  note     computed {meta['computed']} with GDS {meta['gds_version']}; projection "
          f"{meta['projection']['nodes']} nodes, {meta['projection']['relationships']} relationships")
    top = sorted(results["pagerank"].items(), key=lambda item: (-item[1], item[0]))[:10]
    print("  PageRank, the 10 highest: " + "; ".join(f"{aid} {names.get(aid, '')[:28]} {score:.3f}" for aid, score in top))
    sizes = Counter(results["community"].values())
    print(f"  Louvain: {meta['louvain']['communities']} communities, modularity {meta['louvain']['modularity']:.4f}, "
          f"{meta['louvain']['levels']} levels; the 5 largest hold "
          + ", ".join(str(size) for _, size in sizes.most_common(5)) + " entries")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GDS analytics on ATT&CK, saved under analytics/ and written to Neo4j.")
    parser.add_argument("--recompute", action="store_true", help="run GDS again and overwrite the saved files")
    args = parser.parse_args(argv)
    print("Graph analytics on ATT&CK (PageRank, Louvain, FastRP)")
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    problems = []
    with driver:
        try:
            entries = db.run(driver, database, labels_query())
            labels = {row["attack_id"]: row["label"] for row in entries}
            names = {row["attack_id"]: row["name"] for row in entries}
            if len(labels) != EXPECTED_NODES:
                text = f"{len(labels)} ATT&CK nodes in Neo4j, expected {EXPECTED_NODES} (run load_attack.py first)"
                print(f"  PROBLEM  {text}\n\nRESULT: FAILED - {text}")
                return 1
            if args.recompute or not files_exist():
                print("  ok       running GDS (no saved results, or --recompute)")
                save(compute(driver, database), labels)
                results = load()   # every machine writes the same numbers: the ones in the files
                print(f"  ok       saved {TABLE_FILE.relative_to(datafiles.REPO)}, {VECTORS_FILE.relative_to(datafiles.REPO)} "
                      f"and {META_FILE.relative_to(datafiles.REPO)}")
            else:
                results = load()
                print(f"  ok       loaded the saved results from {ANALYTICS_DIR.relative_to(datafiles.REPO)} (GDS not run)")
                missing = sorted(set(labels) - set(results["pagerank"]))
                if missing:
                    text = f"the saved results lack {len(missing)} ATT&CK entries, for example {missing[0]}"
                    print(f"  PROBLEM  {text}\n\nRESULT: FAILED - {text}")
                    return 1
            summary(results, names)
            problems += write(driver, database, results, labels)
            nearest, probe_problems = probe(driver, database)
            problems += probe_problems
            if not probe_problems:
                print(f"  ok       {FASTRP_VECTOR_INDEX[GROUP]} finds {PROBE_GROUP} from its own vector; the most similar "
                      f"groups by FastRP: " + "; ".join(f"{row['attack_id']} {row['name']} {row['score']:.3f}"
                                                         for row in nearest[1:]))
        except (Neo4jError, DriverError) as exc:
            problems.append(db.explain(exc))
            print(f"  PROBLEM  {db.explain(exc)}")
        except ValueError as exc:
            problems.append(str(exc))
            print(f"  PROBLEM  {exc}")
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print(f"\nRESULT: OK - every ATT&CK node carries {PAGERANK}, {COMMUNITY} and {FASTRP}; results in analytics/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
