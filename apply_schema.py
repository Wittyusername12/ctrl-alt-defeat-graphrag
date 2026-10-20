r"""Apply schema.cypher to the project's Neo4j database, then count and list the constraints and indexes.

Run from the repository folder with the project's Python, before any data is loaded:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe apply_schema.py

Every statement in schema.cypher uses IF NOT EXISTS, so running this script again changes nothing.
The last line starts with RESULT. Exit code 0 when every constraint and index is in place, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from neo4j.exceptions import DriverError, Neo4jError

import db
from schemas import KEY, NODE_LABELS, VECTOR_INDEXES

SCHEMA_FILE = Path(__file__).resolve().parent / "schema.cypher"
INDEX_WAIT_SECONDS = 120


def statements(text: str) -> list[str]:
    """The statements in a Cypher file: comment lines dropped, split at each semicolon that ends a line."""
    kept = [line for line in text.splitlines() if not line.strip().startswith("//")]
    return [part.strip() for part in re.split(r";[ \t]*(?:\n|$)", "\n".join(kept)) if part.strip()]


def wait_for_index(driver, database: str, name: str) -> dict | None:
    """The SHOW INDEXES row of the named index once it is no longer populating (or after the time limit)."""
    deadline = time.monotonic() + INDEX_WAIT_SECONDS
    while True:
        rows = [row for row in db.run_plain(driver, database, "SHOW INDEXES YIELD name, type, state, options")
                if row["name"] == name]
        if not rows or rows[0]["state"] != "POPULATING" or time.monotonic() > deadline:
            return rows[0] if rows else None
        time.sleep(2)


def apply(driver, database: str) -> list[str]:
    problems: list[str] = []
    failed_indexes = set()
    for statement in statements(SCHEMA_FILE.read_text(encoding="utf-8")):
        first_line = statement.splitlines()[0]
        try:
            db.run_plain(driver, database, statement)
            print(f"  ok       {first_line}")
        except Neo4jError as exc:
            problems.append(f"{first_line}: {db.explain(exc)}")
            print(f"  PROBLEM  {first_line}\n           {db.explain(exc)}")
            failed_indexes |= {name for name in VECTOR_INDEXES if f"INDEX {name} " in first_line}

    print("\nConstraints")
    rows = db.run_plain(driver, database, db.CONSTRAINTS_QUERY)
    for label in NODE_LABELS:
        match = db.unique_constraint(rows, label, KEY[label])
        if match:
            print(f"  ok       {match['name']}: {label}.{KEY[label]} is unique ({match['type']})")
        else:
            problems.append(f"no uniqueness constraint on {label}.{KEY[label]}")
            print(f"  PROBLEM  no uniqueness constraint on {label}.{KEY[label]}")
    print(f"  note     {len(rows)} constraint(s) in the database; the schema defines {len(NODE_LABELS)}")

    print("\nVector indexes")
    for name in VECTOR_INDEXES:
        index = wait_for_index(driver, database, name)
        if index is None:
            if name in failed_indexes:
                print(f"  note     {name} was not created (the PROBLEM line under Statements says why)")
            else:
                problems.append(f"the vector index {name} does not exist")
                print(f"  PROBLEM  the vector index {name} does not exist")
        elif index["state"] != "ONLINE":
            problems.append(f"the vector index {name} is {index['state']}")
            print(f"  PROBLEM  the vector index {name} is {index['state']}")
        else:
            config = (index.get("options") or {}).get("indexConfig") or {}
            settings = ", ".join(f"{key.removeprefix('vector.')} {config[key]}" for key in
                                 ("vector.dimensions", "vector.similarity_function", "vector.quantization.type")
                                 if key in config)
            print(f"  ok       {name}: {index['type']}, {index['state']}; {settings}")
    return problems


def main() -> int:
    print("Apply schema.cypher to Neo4j")
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - no connection to Neo4j")
        return 1
    with driver:
        print(f"  ok       connected, database {database!r}\n\nStatements")
        try:
            problems = apply(driver, database)
        except (Neo4jError, DriverError) as exc:
            print(f"  PROBLEM  {db.explain(exc)}\n\nRESULT: FAILED - {db.explain(exc)}")
            return 1
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK - every constraint and every vector index are in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
