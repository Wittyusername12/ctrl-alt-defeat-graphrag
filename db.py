"""Connection to the project's Neo4j database.

The connection settings come only from the .env file in the repository folder, read by explicit path.
Every data query sent through run() starts with an explicit Cypher version line, so the result does not
depend on the database's default Cypher version. Schema statements (CREATE CONSTRAINT, CREATE VECTOR
INDEX) and SHOW commands are written the same way in Cypher 5 and Cypher 25; they go through
run_plain() exactly as written.
"""
from __future__ import annotations

import logging
from pathlib import Path

import neo4j
from dotenv import dotenv_values
from neo4j.exceptions import AuthError, ConfigurationError, DriverError, Neo4jError, ServiceUnavailable

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"
CYPHER_VERSION = "CYPHER 25"
BATCH_SIZE = 1000

# The scripts report every failure themselves; this keeps the driver's own log lines out of the output.
logging.getLogger("neo4j").addHandler(logging.NullHandler())


class SetupError(Exception):
    """A problem with .env or the connection, explained in plain words."""


def settings() -> dict[str, str]:
    """NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD and NEO4J_DATABASE from the repository's .env."""
    if not ENV_FILE.is_file():
        raise SetupError(f"no .env at {ENV_FILE}; copy .env.example to .env and fill it in")
    values = {name: (value or "").strip() for name, value in dotenv_values(ENV_FILE).items()}
    missing = [name for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not values.get(name)]
    if missing:
        raise SetupError(f"{', '.join(missing)} empty or missing in {ENV_FILE}")
    values.setdefault("NEO4J_DATABASE", "neo4j")
    values["NEO4J_DATABASE"] = values["NEO4J_DATABASE"] or "neo4j"
    return values


def connect() -> tuple[neo4j.Driver, str]:
    """An open driver and the database name. The caller closes the driver."""
    env = settings()
    uri, user = env["NEO4J_URI"], env["NEO4J_USER"]
    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, env["NEO4J_PASSWORD"]))
    except (ConfigurationError, ValueError) as exc:
        raise SetupError(f"NEO4J_URI {uri!r} is not a usable address: {exc}") from exc
    try:
        driver.verify_connectivity()
    except AuthError as exc:
        driver.close()
        raise SetupError(f"Neo4j at {uri} refused user {user!r} with the password in .env") from exc
    except ServiceUnavailable as exc:
        driver.close()
        raise SetupError(f"no Neo4j server answered at {uri}; is the instance started in Neo4j Desktop?") from exc
    return driver, env["NEO4J_DATABASE"]


def versioned(query: str) -> str:
    """The query with an explicit Cypher version line in front."""
    return f"{CYPHER_VERSION}\n{query}"


def run(driver: neo4j.Driver, database: str, query: str, **parameters) -> list[dict]:
    """Run one data query (with the Cypher version line) and return its rows as dictionaries."""
    result = driver.execute_query(versioned(query), parameters_=parameters, database_=database)
    return [record.data() for record in result.records]


def run_plain(driver: neo4j.Driver, database: str, statement: str, **parameters) -> list[dict]:
    """Run one schema statement or SHOW command exactly as written and return its rows."""
    result = driver.execute_query(statement, parameters_=parameters, database_=database)
    return [record.data() for record in result.records]


CONSTRAINTS_QUERY = "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties"


def unique_constraint(rows: list[dict], label: str, key: str) -> dict | None:
    """The SHOW CONSTRAINTS row that makes `key` unique on nodes labelled `label`, if there is one."""
    for row in rows:
        if (row.get("entityType") == "NODE" and row.get("labelsOrTypes") == [label]
                and row.get("properties") == [key] and "UNIQUE" in str(row.get("type", "")).upper()):
            return row
    return None


def batches(rows: list, size: int = BATCH_SIZE):
    """The rows in consecutive slices of at most `size`."""
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def explain(exc: Exception) -> str:
    """A one-line description of a Neo4j or setup failure."""
    if isinstance(exc, SetupError):
        return str(exc)
    if isinstance(exc, Neo4jError):
        return f"Neo4j error {exc.code}: {exc.message}"
    if isinstance(exc, DriverError):
        return f"Neo4j driver error: {exc}"
    return f"{type(exc).__name__}: {exc}"
