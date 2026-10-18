r"""Setup check for the CTRL ALT DEFEAT GraphRAG project.

Run from the repository folder with the project's own Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe scripts\check_setup.py

Checks, in order:
  1. Python: version 3.11, running inside a virtual environment, nothing inside OneDrive.
  2. Packages: every package in requirements.txt is installed at the pinned version.
  3. .env in the repository folder: every name listed in .env.example has a value.
     Secret values are never printed, only their length.
  4. Neo4j: the server at NEO4J_URI accepts NEO4J_USER and NEO4J_PASSWORD; prints its version.
  5. Plugins: the APOC and Graph Data Science versions, and the procedures the project calls.

The last line starts with RESULT. Exit code 0 when nothing is wrong, 1 otherwise.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# The neo4j driver logs its own warnings; this script reports every failure itself.
logging.getLogger("neo4j").addHandler(logging.NullHandler())

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env"
ENV_TEMPLATE = REPO / ".env.example"
REQUIREMENTS = REPO / "requirements.txt"
SECRETS = {"OPENAI_API_KEY", "NEO4J_PASSWORD"}
PROCEDURES = [
    "gds.graph.project",
    "gds.graph.drop",
    "gds.pageRank.write",
    "gds.louvain.write",
    "gds.fastRP.write",
    "db.index.vector.queryNodes",
]

problems: list[str] = []


def ok(text: str) -> None:
    print(f"  ok       {text}")


def note(text: str) -> None:
    print(f"  note     {text}")


def problem(text: str) -> None:
    problems.append(text)
    print(f"  PROBLEM  {text}")


def check_python() -> None:
    print("\n1. Python")
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        ok(f"Python {version}")
    else:
        problem(f"Python {version}; the project uses Python 3.11")
    if sys.prefix != sys.base_prefix:
        ok(f"virtual environment: {sys.prefix}")
    else:
        problem(f"not a virtual environment: {sys.executable}; run the project's own Python")
    for label, path in (("repository", REPO), ("Python environment", Path(sys.prefix))):
        if "onedrive" in str(path).casefold():
            problem(f"{label} is inside OneDrive: {path}")
        else:
            ok(f"{label} outside OneDrive: {path}")
    try:
        branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=20)
        commit = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        note("git did not run, so the commit is not shown")
        return
    if branch.returncode == 0 and commit.returncode == 0:
        note(f"repository on branch {branch.stdout.strip()}, commit {commit.stdout.strip()}")


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def same_version(installed: str, pinned: str) -> bool:
    try:
        from packaging.version import Version
        return Version(installed) == Version(pinned)
    except Exception:
        return installed == pinned


def check_packages() -> None:
    print("\n2. Packages")
    try:
        lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        problem(f"requirements.txt not readable: {exc}")
        return
    wanted = []
    for number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line)
        if match is None:
            problem(f"requirements.txt line {number} not understood: {raw}")
            continue
        wanted.append((match.group(1), match.group(2)))
    installed = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            installed[normalized(name)] = dist.version
    missing = [(name, pin) for name, pin in wanted if normalized(name) not in installed]
    different = [(name, pin, installed[normalized(name)]) for name, pin in wanted
                 if normalized(name) in installed and not same_version(installed[normalized(name)], pin)]
    right = len(wanted) - len(missing) - len(different)
    if not missing and not different:
        ok(f"all {len(wanted)} packages in requirements.txt are installed at the pinned version")
        return
    if missing:
        examples = ", ".join(f"{name}=={pin}" for name, pin in missing[:6])
        problem(f"{len(missing)} of {len(wanted)} packages are not installed, for example {examples}")
    for name, pin, have in different[:12]:
        problem(f"{name} is {have}; requirements.txt pins {pin}")
    if len(different) > 12:
        problem(f"{len(different) - 12} more packages are at other versions")
    note(f"{right} of {len(wanted)} packages are at the pinned version")


def read_env() -> dict[str, str]:
    print("\n3. .env")
    try:
        template = ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        problem(f".env.example not readable: {exc}")
        return {}
    names = [m.group(1) for m in (re.match(r"([A-Z][A-Z0-9_]*)=", line) for line in template) if m]
    also_set = sorted(name for name in os.environ if name.startswith(("OPENAI_", "NEO4J_")))
    if also_set:
        note("Windows environment variables also set: " + ", ".join(also_set) + ". The project reads .env; "
             "the openai library still uses OPENAI_BASE_URL, OPENAI_ORG_ID and OPENAI_PROJECT_ID if set.")
    if not ENV_FILE.is_file():
        problem(f"no .env at {ENV_FILE}; copy .env.example to .env and fill it in")
        return {}
    try:
        from dotenv import dotenv_values
    except ImportError:
        problem("python-dotenv is not installed, so .env could not be read (see 2. Packages)")
        return {}
    env = {name: (value or "").strip() for name, value in dotenv_values(ENV_FILE).items()}
    ok(f"read {ENV_FILE}")
    for name in names:
        value = env.get(name, "")
        if not value:
            problem(f"{name} is empty or missing in .env")
        elif name in SECRETS:
            ok(f"{name} is set ({len(value)} characters)")
        else:
            ok(f"{name} = {value}")
    return env


def check_neo4j(env: dict[str, str]) -> None:
    print("\n4. Neo4j")
    try:
        import neo4j
        from neo4j.exceptions import (AuthError, ClientError, ConfigurationError, DriverError,
                                      Neo4jError, ServiceUnavailable)
    except ImportError:
        problem("the neo4j Python driver is not installed (see 2. Packages); the database was not tested")
        return
    uri, user, password = env.get("NEO4J_URI"), env.get("NEO4J_USER"), env.get("NEO4J_PASSWORD")
    database = env.get("NEO4J_DATABASE") or "neo4j"
    if not (uri and user and password):
        problem("NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD must all be set in .env; the database was not tested")
        return
    try:
        with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            ok(f"connected to {uri} as {user}")
            with driver.session(database=database) as session:
                for record in session.run("CALL dbms.components() YIELD name, versions, edition "
                                          "RETURN name, versions, edition"):
                    ok(f"{record['name']} {', '.join(record['versions'])}, {record['edition']} edition, "
                       f"database {database!r}")
                print("\n5. Plugins")
                for label, query in (("APOC", "RETURN apoc.version() AS version"),
                                     ("Graph Data Science", "RETURN gds.version() AS version")):
                    try:
                        ok(f"{label} {session.run(query).single(strict=True)['version']}")
                    except ClientError as exc:
                        problem(f"{label} is not installed on this instance ({exc.code})")
                available = {record["name"] for record in session.run("SHOW PROCEDURES YIELD name RETURN name")}
                found = [name for name in PROCEDURES if name in available]
                missing = [name for name in PROCEDURES if name not in available]
                if found:
                    ok(f"procedures found ({len(found)} of {len(PROCEDURES)}): " + ", ".join(found))
                if missing:
                    problem("procedures not found: " + ", ".join(missing))
    except AuthError:
        problem(f"Neo4j at {uri} refused user {user!r} with the password in .env")
    except ServiceUnavailable as exc:
        problem(f"no Neo4j server answered at {uri}; is the instance started in Neo4j Desktop? ({exc})")
    except (ConfigurationError, ValueError) as exc:
        problem(f"NEO4J_URI {uri!r} is not a usable address: {exc}")
    except Neo4jError as exc:
        problem(f"Neo4j error {exc.code}: {exc.message}")
    except DriverError as exc:
        problem(f"Neo4j driver error: {exc}")


def main() -> int:
    print("Setup check for the CTRL ALT DEFEAT GraphRAG project")
    check_python()
    check_packages()
    env = read_env()
    check_neo4j(env)
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for text in problems:
            print(f"  - {text}")
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
