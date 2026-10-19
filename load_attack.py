r"""Load MITRE ATT&CK v20.0 (Enterprise) into the knowledge graph: STIX JSON -> RDF (rdflib) -> Turtle -> Neo4j.

Run from the repository folder with the project's Python, after apply_schema.py:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe load_attack.py

Steps:
  1. Check data/attack/enterprise-attack.json against its SHA-256 in data/SHA256SUMS.
  2. Read the live objects (neither revoked nor deprecated) of five kinds, and the relationships between them.
  3. Build an RDF graph with rdflib: a small OWL vocabulary, one resource per ATT&CK object (named by its
     attack.mitre.org address), one plain triple per relationship, and one rdf:Statement resource per
     relationship carrying the relationship's own ID and description (an RDF 1.1 triple cannot carry
     properties; a Neo4j relationship can). Saved as out/attack-v20.ttl.
  4. Count the RDF classes with a SPARQL query and compare every count with the STIX file and with the
     counts measured from the same file on 18 Oct 2026.
  5. Write the nodes and relationships to Neo4j, taken from the RDF graph, in batches of 1,000. MERGE is
     used, so running the script again changes nothing.
  6. Read the counts back from Neo4j and compare them with the RDF graph.

Option --no-neo4j stops after step 4.
The last line starts with RESULT. Exit code 0 when every count matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from neo4j.exceptions import DriverError, Neo4jError
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

import datafiles
import db
from schemas import (ALIASES, ATTACK_ID, ATTACK_LABELS, DESCRIPTION, EDGE_ID, EDGE_IRI, GROUP, IS_SUBTECHNIQUE, KEY,
                     MITIGATES, MITIGATION, NAME, ONTOLOGY_IRI, PART_OF_TACTIC, RDF_PROPERTY, RDF_RELATIONSHIP,
                     SHORTNAME, SOFTWARE, SOFTWARE_TYPE, SOURCE, SOURCE_ATTACK, STIX_ID, SUBTECHNIQUE_OF, TACTIC,
                     TECHNIQUE, URL, USES)

CAD = Namespace(ONTOLOGY_IRI)
EDGE = Namespace(EDGE_IRI)

STIX_LABEL = {
    "attack-pattern": TECHNIQUE,
    "x-mitre-tactic": TACTIC,
    "intrusion-set": GROUP,
    "malware": SOFTWARE,
    "tool": SOFTWARE,
    "course-of-action": MITIGATION,
}
STIX_RELATIONSHIP = {"uses": USES, "mitigates": MITIGATES, "subtechnique-of": SUBTECHNIQUE_OF}

# The (source label, relationship type, target label) combinations the graph allows.
EDGE_KINDS = (
    (SOFTWARE, USES, TECHNIQUE),
    (GROUP, USES, TECHNIQUE),
    (GROUP, USES, SOFTWARE),
    (MITIGATION, MITIGATES, TECHNIQUE),
    (TECHNIQUE, SUBTECHNIQUE_OF, TECHNIQUE),
    (TECHNIQUE, PART_OF_TACTIC, TACTIC),
)

# Counted from the same file (sha256 66f3fe3f…) on 18 Oct 2026 with a separate script; campaigns left out.
EXPECTED_NODES = {TECHNIQUE: 706, TACTIC: 15, GROUP: 178, SOFTWARE: 842, MITIGATION: 44}
EXPECTED_SUBTECHNIQUES = 479
EXPECTED_EDGES = {
    (SOFTWARE, USES, TECHNIQUE): 11608,
    (GROUP, USES, TECHNIQUE): 4815,
    (GROUP, USES, SOFTWARE): 1166,
    (MITIGATION, MITIGATES, TECHNIQUE): 1477,
    (TECHNIQUE, SUBTECHNIQUE_OF, TECHNIQUE): 479,
    (TECHNIQUE, PART_OF_TACTIC, TACTIC): 881,
}


class LoadError(Exception):
    """The input file does not look the way this loader expects."""


@dataclass
class Node:
    label: str
    attack_id: str
    stix_id: str
    name: str
    url: str
    description: str | None = None
    aliases: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


@dataclass
class Edge:
    kind: tuple[str, str, str]  # (source label, relationship type, target label)
    source: str                 # attack_id of the source node
    target: str                 # attack_id of the target node
    stix_id: str | None = None
    description: str | None = None

    @property
    def edge_id(self) -> str:
        return f"{SOURCE_ATTACK}:{self.kind[1]}:{self.source}:{self.target}"


def kind_name(kind: tuple[str, str, str]) -> str:
    return f"{kind[0]} -{kind[1]}-> {kind[2]}"


# ---------------------------------------------------------------- 2. read the STIX file

def is_live(obj: dict) -> bool:
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


def attack_reference(obj: dict) -> tuple[str, str]:
    """MITRE's ID and web address for an object, from its mitre-attack external reference."""
    for reference in obj.get("external_references", []):
        if reference.get("source_name") == "mitre-attack" and reference.get("external_id"):
            attack_id = reference["external_id"]
            return attack_id, reference.get("url") or f"https://attack.mitre.org/{attack_id}"
    raise LoadError(f"{obj.get('id')} has no mitre-attack external ID")


def read_stix(path: Path) -> tuple[list[Node], list[Edge]]:
    objects = json.loads(path.read_text(encoding="utf-8"))["objects"]
    by_stix: dict[str, Node] = {}
    phases: dict[str, list[str]] = {}  # technique attack_id -> tactic short names
    for obj in objects:
        label = STIX_LABEL.get(obj["type"])
        if label is None or not is_live(obj):
            continue
        attack_id, url = attack_reference(obj)
        names = obj.get("aliases", []) if label == GROUP else obj.get("x_mitre_aliases", [])
        aliases = tuple(dict.fromkeys(name for name in names if name != obj["name"]))
        extra: dict = {}
        if label == TECHNIQUE:
            extra[IS_SUBTECHNIQUE] = bool(obj.get("x_mitre_is_subtechnique", False))
            phases[attack_id] = [phase["phase_name"] for phase in obj.get("kill_chain_phases", [])
                                 if phase.get("kill_chain_name") == "mitre-attack"]
        elif label == SOFTWARE:
            extra[SOFTWARE_TYPE] = obj["type"]
        elif label == TACTIC:
            extra[SHORTNAME] = obj["x_mitre_shortname"]
        by_stix[obj["id"]] = Node(label, attack_id, obj["id"], obj["name"], url,
                                  obj.get("description"), aliases, extra)
    nodes = list(by_stix.values())
    duplicate_ids = [aid for aid, count in Counter(node.attack_id for node in nodes).items() if count > 1]
    if duplicate_ids:
        raise LoadError(f"ATT&CK IDs used by more than one live object: {duplicate_ids[:5]}")

    edges: list[Edge] = []
    for obj in objects:
        if obj["type"] != "relationship" or not is_live(obj):
            continue
        relationship = STIX_RELATIONSHIP.get(obj["relationship_type"])
        source, target = by_stix.get(obj["source_ref"]), by_stix.get(obj["target_ref"])
        if relationship is None or source is None or target is None:
            continue  # other relationship types, campaigns, detection strategies, revoked or deprecated ends
        kind = (source.label, relationship, target.label)
        if kind not in EDGE_KINDS:
            raise LoadError(f"unexpected relationship {obj['id']}: {kind_name(kind)}")
        edges.append(Edge(kind, source.attack_id, target.attack_id, obj["id"], obj.get("description")))

    tactics = {node.extra[SHORTNAME]: node for node in nodes if node.label == TACTIC}
    for technique, names in phases.items():
        for name in names:
            if name not in tactics:
                raise LoadError(f"{technique} names an unknown tactic {name!r}")
            edges.append(Edge((TECHNIQUE, PART_OF_TACTIC, TACTIC), technique, tactics[name].attack_id))

    duplicate_edges = [eid for eid, count in Counter(edge.edge_id for edge in edges).items() if count > 1]
    if duplicate_edges:
        raise LoadError(f"relationships listed more than once: {duplicate_edges[:5]}")
    return nodes, edges


# ---------------------------------------------------------------- 3. build the RDF graph

def term(name: str) -> URIRef:
    """The RDF property for a node property or relationship type defined in schemas.py."""
    return CAD[RDF_PROPERTY[name]] if name in RDF_PROPERTY else CAD[RDF_RELATIONSHIP[name]]


def build_graph(nodes: list[Node], edges: list[Edge]) -> Graph:
    graph = Graph()
    graph.bind("cad", CAD)
    graph.bind("edge", EDGE)
    graph.bind("owl", OWL)

    # Vocabulary: one OWL class per ATT&CK label, one object property per relationship type.
    for label in ATTACK_LABELS:
        graph.add((CAD[label], RDF.type, OWL.Class))
        graph.add((CAD[label], RDFS.label, Literal(label)))
    domain_range = {MITIGATES: (MITIGATION, TECHNIQUE), SUBTECHNIQUE_OF: (TECHNIQUE, TECHNIQUE),
                    PART_OF_TACTIC: (TECHNIQUE, TACTIC)}
    for relationship in RDF_RELATIONSHIP:
        prop = term(relationship)
        graph.add((prop, RDF.type, OWL.ObjectProperty))
        graph.add((prop, RDFS.label, Literal(relationship)))
        # USES links a Group or Software to a Technique or Software, so it gets no single domain or range.
        if relationship in domain_range:
            graph.add((prop, RDFS.domain, CAD[domain_range[relationship][0]]))
            graph.add((prop, RDFS.range, CAD[domain_range[relationship][1]]))
    for name in RDF_PROPERTY:
        graph.add((term(name), RDF.type, OWL.DatatypeProperty))

    # One resource per ATT&CK object, named by its attack.mitre.org address.
    address: dict[str, URIRef] = {}
    for node in nodes:
        subject = URIRef(node.url)
        address[node.attack_id] = subject
        graph.add((subject, RDF.type, CAD[node.label]))
        graph.add((subject, term(ATTACK_ID), Literal(node.attack_id)))
        graph.add((subject, RDFS.label, Literal(node.name)))
        graph.add((subject, term(STIX_ID), Literal(node.stix_id)))
        graph.add((subject, term(URL), Literal(node.url, datatype=XSD.anyURI)))
        if node.description:
            graph.add((subject, term(DESCRIPTION), Literal(node.description)))
        for alias in node.aliases:
            graph.add((subject, term(ALIASES), Literal(alias)))
        for name, value in node.extra.items():
            graph.add((subject, term(name), Literal(value)))
    if len(address) != len(nodes):
        raise LoadError("two ATT&CK objects share one attack.mitre.org address")

    # One plain triple per relationship, plus an rdf:Statement resource holding the relationship's own values.
    for edge in edges:
        subject, predicate, obj = address[edge.source], term(edge.kind[1]), address[edge.target]
        graph.add((subject, predicate, obj))
        statement = EDGE[edge.edge_id]
        graph.add((statement, RDF.type, RDF.Statement))
        graph.add((statement, RDF.subject, subject))
        graph.add((statement, RDF.predicate, predicate))
        graph.add((statement, RDF.object, obj))
        graph.add((statement, term(EDGE_ID), Literal(edge.edge_id)))
        graph.add((statement, term(SOURCE), Literal(SOURCE_ATTACK)))
        if edge.stix_id:
            graph.add((statement, term(STIX_ID), Literal(edge.stix_id)))
        if edge.description:
            graph.add((statement, term(DESCRIPTION), Literal(edge.description)))
    return graph


def sparql_class_counts(graph: Graph) -> dict[str, int]:
    """Number of resources in each ATT&CK class, counted with a SPARQL query."""
    query = """
        SELECT ?class (COUNT(?resource) AS ?n)
        WHERE { ?resource a ?class . ?class a owl:Class . }
        GROUP BY ?class
    """
    counts = {str(row[0]): int(row[1]) for row in graph.query(query, initNs={"owl": OWL})}
    return {label: counts.get(str(CAD[label]), 0) for label in ATTACK_LABELS}


# ---------------------------------------------------------------- rows for Neo4j, read from the RDF graph

def literal(graph: Graph, subject, name: str) -> str | None:
    value = graph.value(subject, term(name))
    return None if value is None else str(value)


def rows_from_graph(graph: Graph) -> tuple[dict[str, list[dict]], dict[tuple, list[dict]]]:
    """Node rows per label and relationship rows per kind, taken from the RDF graph."""
    label_of: dict = {}
    attack_id_of: dict = {}
    node_rows: dict[str, list[dict]] = {label: [] for label in ATTACK_LABELS}
    for label in ATTACK_LABELS:
        for subject in graph.subjects(RDF.type, CAD[label]):
            attack_id = literal(graph, subject, ATTACK_ID)
            label_of[subject], attack_id_of[subject] = label, attack_id
            props = {NAME: str(graph.value(subject, RDFS.label)), STIX_ID: literal(graph, subject, STIX_ID),
                     URL: literal(graph, subject, URL)}
            description = literal(graph, subject, DESCRIPTION)
            if description:
                props[DESCRIPTION] = description
            aliases = sorted(str(alias) for alias in graph.objects(subject, term(ALIASES)))
            if aliases:
                props[ALIASES] = aliases
            for name in (IS_SUBTECHNIQUE, SOFTWARE_TYPE, SHORTNAME):
                value = graph.value(subject, term(name))
                if value is not None:
                    props[name] = value.toPython()
            node_rows[label].append({"key": attack_id, "props": props})

    relationship_of = {term(relationship): relationship for relationship in RDF_RELATIONSHIP}
    edge_rows: dict[tuple, list[dict]] = {kind: [] for kind in EDGE_KINDS}
    for statement in graph.subjects(RDF.type, RDF.Statement):
        subject, obj = graph.value(statement, RDF.subject), graph.value(statement, RDF.object)
        kind = (label_of[subject], relationship_of[graph.value(statement, RDF.predicate)], label_of[obj])
        props = {SOURCE: literal(graph, statement, SOURCE)}
        for name in (STIX_ID, DESCRIPTION):
            value = literal(graph, statement, name)
            if value:
                props[name] = value
        edge_rows[kind].append({"edge_id": literal(graph, statement, EDGE_ID), "src": attack_id_of[subject],
                                "dst": attack_id_of[obj], "props": props})
    for rows in node_rows.values():
        rows.sort(key=lambda row: row["key"])
    for rows in edge_rows.values():
        rows.sort(key=lambda row: row["edge_id"])
    return node_rows, edge_rows


# ---------------------------------------------------------------- 5-6. Neo4j

def node_query(label: str) -> str:
    return (f"UNWIND $rows AS row\n"
            f"MERGE (n:`{label}` {{`{KEY[label]}`: row.key}})\n"
            f"SET n += row.props\n"
            f"RETURN count(n) AS written")


def edge_query(kind: tuple[str, str, str]) -> str:
    source, relationship, target = kind
    return (f"UNWIND $rows AS row\n"
            f"MATCH (a:`{source}` {{`{KEY[source]}`: row.src}})\n"
            f"MATCH (b:`{target}` {{`{KEY[target]}`: row.dst}})\n"
            f"MERGE (a)-[r:`{relationship}` {{`{EDGE_ID}`: row.edge_id}}]->(b)\n"
            f"SET r += row.props\n"
            f"RETURN count(r) AS written")


def node_count_query(label: str) -> str:
    return f"MATCH (n:`{label}`) RETURN count(n) AS n"


def subtechnique_count_query() -> str:
    return f"MATCH (n:`{TECHNIQUE}`) WHERE n.`{IS_SUBTECHNIQUE}` = true RETURN count(n) AS n"


def edge_count_query(kind: tuple[str, str, str]) -> str:
    source, relationship, target = kind
    return (f"MATCH (:`{source}`)-[r:`{relationship}`]->(:`{target}`) "
            f"WHERE r.`{SOURCE}` = '{SOURCE_ATTACK}' RETURN count(r) AS n")


def missing_constraints(driver, database: str) -> list[str]:
    """ATT&CK labels without their uniqueness constraint (apply_schema.py creates them)."""
    rows = db.run_plain(driver, database, db.CONSTRAINTS_QUERY)
    return [label for label in ATTACK_LABELS if db.unique_constraint(rows, label, KEY[label]) is None]


def write(driver, database: str, node_rows: dict, edge_rows: dict) -> list[str]:
    """Write every row and return a problem sentence for any kind whose rows did not all reach the database."""
    problems = []
    for label, rows in node_rows.items():
        written = sum(db.run(driver, database, node_query(label), rows=batch)[0]["written"]
                      for batch in db.batches(rows))
        if written == len(rows):
            print(f"  ok       {label}: {written} nodes written")
        else:
            problems.append(f"{label}: {written} of {len(rows)} nodes written")
            print(f"  PROBLEM  {problems[-1]}")
    for kind, rows in edge_rows.items():
        written = sum(db.run(driver, database, edge_query(kind), rows=batch)[0]["written"]
                      for batch in db.batches(rows))
        if written == len(rows):
            print(f"  ok       {kind_name(kind)}: {written} relationships written")
        else:
            problems.append(f"{kind_name(kind)}: {written} of {len(rows)} relationships written "
                            f"(rows whose end node was not found are skipped)")
            print(f"  PROBLEM  {problems[-1]}")
    return problems


def read_back(driver, database: str) -> tuple[dict[str, int], int, dict[tuple, int]]:
    nodes = {label: db.run(driver, database, node_count_query(label))[0]["n"] for label in ATTACK_LABELS}
    subtechniques = db.run(driver, database, subtechnique_count_query())[0]["n"]
    edges = {kind: db.run(driver, database, edge_count_query(kind))[0]["n"] for kind in EDGE_KINDS}
    return nodes, subtechniques, edges


# ---------------------------------------------------------------- comparisons and main

def compare(name: str, found: int, *expected: tuple[str, int]) -> str | None:
    """Print one comparison line; return a problem sentence when any expected number differs."""
    wrong = [f"{number} {source}" for source, number in expected if number != found]
    if wrong:
        message = f"{name}: {found}, but {' and '.join(wrong)}"
        print(f"  PROBLEM  {message}")
        return message
    print(f"  ok       {name}: {found}")
    return None


def finish(problems: list[str], success: str) -> int:
    if problems:
        print(f"\nRESULT: {len(problems)} problem(s)")
        for message in problems:
            print(f"  - {message}")
        return 1
    print(f"\nRESULT: OK - {success}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load MITRE ATT&CK v20.0 into Neo4j through rdflib.")
    parser.add_argument("--no-neo4j", action="store_true", help="build and check the RDF graph only")
    args = parser.parse_args(argv)
    print("Load MITRE ATT&CK v20.0 (Enterprise)")

    print("\n1. Data file")
    issue = datafiles.check(datafiles.ATTACK_FILE)
    if issue:
        print(f"  PROBLEM  {issue}")
        return finish([issue], "")
    print(f"  ok       {datafiles.ATTACK_FILE.relative_to(datafiles.REPO).as_posix()} matches its SHA-256")

    print("\n2. STIX objects")
    started = time.perf_counter()
    try:
        nodes, edges = read_stix(datafiles.ATTACK_FILE)
    except (LoadError, KeyError, ValueError) as exc:
        print(f"  PROBLEM  the STIX file could not be read: {exc}")
        return finish([f"the STIX file could not be read: {exc}"], "")
    stix_nodes = Counter(node.label for node in nodes)
    stix_edges = Counter(edge.kind for edge in edges)
    print(f"  ok       {len(nodes)} objects and {len(edges)} relationships read in {time.perf_counter() - started:.1f} s")

    print("\n3. RDF graph (rdflib)")
    started = time.perf_counter()
    graph = build_graph(nodes, edges)
    datafiles.OUT.mkdir(exist_ok=True)
    graph.serialize(destination=str(datafiles.ATTACK_TURTLE), format="turtle", encoding="utf-8")
    megabytes = datafiles.ATTACK_TURTLE.stat().st_size / 1e6
    print(f"  ok       {len(graph)} triples, saved as {datafiles.ATTACK_TURTLE.relative_to(datafiles.REPO).as_posix()} "
          f"({megabytes:.1f} MB), in {time.perf_counter() - started:.1f} s")

    print("\n4. Counts in the RDF graph (classes counted with SPARQL)")
    problems: list[str | None] = []
    classes = sparql_class_counts(graph)
    node_rows, edge_rows = rows_from_graph(graph)
    for label in ATTACK_LABELS:
        problems.append(compare(label, classes[label], ("in the STIX file", stix_nodes[label]),
                                ("measured on 18 Oct", EXPECTED_NODES[label])))
    subtechniques = sum(1 for row in node_rows[TECHNIQUE] if row["props"].get(IS_SUBTECHNIQUE))
    problems.append(compare("sub-techniques", subtechniques, ("measured on 18 Oct", EXPECTED_SUBTECHNIQUES)))
    for kind in EDGE_KINDS:
        problems.append(compare(kind_name(kind), len(edge_rows[kind]), ("in the STIX file", stix_edges[kind]),
                                ("measured on 18 Oct", EXPECTED_EDGES[kind])))
    found_problems = [message for message in problems if message]
    if args.no_neo4j or found_problems:
        return finish(found_problems, "RDF graph built and counted; Neo4j not touched (--no-neo4j)")

    print("\n5. Write to Neo4j")
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        print(f"  PROBLEM  {exc}")
        return finish([str(exc)], "")
    with driver:
        try:
            missing = missing_constraints(driver, database)
            if missing:
                message = f"no uniqueness constraint for {', '.join(missing)}; run apply_schema.py first"
                print(f"  PROBLEM  {message}")
                return finish([message], "")
            print(f"  ok       connected, database {database!r}; the uniqueness constraints are in place")
            started = time.perf_counter()
            found_problems += write(driver, database, node_rows, edge_rows)
            print(f"  note     written in {time.perf_counter() - started:.1f} s")
            print("\n6. Read back from Neo4j")
            found_nodes, found_subtechniques, found_edges = read_back(driver, database)
        except (Neo4jError, DriverError) as exc:
            print(f"  PROBLEM  {db.explain(exc)}")
            return finish(found_problems + [db.explain(exc)], "")
    checks = [compare(label, found_nodes[label], ("in the RDF graph", classes[label])) for label in ATTACK_LABELS]
    checks.append(compare("sub-techniques", found_subtechniques, ("in the RDF graph", subtechniques)))
    checks += [compare(kind_name(kind), found_edges[kind], ("in the RDF graph", len(edge_rows[kind])))
               for kind in EDGE_KINDS]
    found_problems += [message for message in checks if message]
    return finish(found_problems, f"{sum(found_nodes.values())} ATT&CK nodes and {sum(found_edges.values())} "
                                  f"relationships in Neo4j, matching the RDF graph and the counts measured on 18 Oct")


if __name__ == "__main__":
    sys.exit(main())
