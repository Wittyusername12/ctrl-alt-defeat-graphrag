"""Every name in the knowledge graph, defined once.

Node labels, relationship types, property names, the RDF vocabulary and the vector index all come from
this file. Every other file imports these names instead of typing them, and tests/test_names.py checks
that schema.cypher uses only names defined here.
"""
from __future__ import annotations

# Node labels
REPORT = "Report"
CHUNK = "Chunk"
TECHNIQUE = "Technique"
TACTIC = "Tactic"
GROUP = "Group"
SOFTWARE = "Software"
MITIGATION = "Mitigation"
ENTITY = "Entity"  # a name from a report that links to nothing in ATT&CK

ATTACK_LABELS = (TECHNIQUE, TACTIC, GROUP, SOFTWARE, MITIGATION)
NODE_LABELS = (REPORT, CHUNK) + ATTACK_LABELS + (ENTITY,)

# Relationship types
HAS_CHUNK = "HAS_CHUNK"              # Report -> Chunk
MENTIONS = "MENTIONS"                # Chunk -> ATT&CK node or Entity, with provenance
USES = "USES"                        # Group or Software -> Technique or Software
SUBTECHNIQUE_OF = "SUBTECHNIQUE_OF"  # sub-technique -> parent Technique
PART_OF_TACTIC = "PART_OF_TACTIC"    # Technique -> Tactic
MITIGATES = "MITIGATES"              # Mitigation -> Technique

ATTACK_RELATIONSHIPS = (USES, SUBTECHNIQUE_OF, PART_OF_TACTIC, MITIGATES)
RELATIONSHIP_TYPES = (HAS_CHUNK, MENTIONS) + ATTACK_RELATIONSHIPS

# The unique key property of each node label (one uniqueness constraint per label in schema.cypher)
REPORT_ID = "report_id"
CHUNK_ID = "chunk_id"
ATTACK_ID = "attack_id"   # MITRE's ID, for example T1036, TA0005, G0129, S0013, M1040
ENTITY_ID = "entity_id"
KEY = {
    REPORT: REPORT_ID,
    CHUNK: CHUNK_ID,
    TECHNIQUE: ATTACK_ID,
    TACTIC: ATTACK_ID,
    GROUP: ATTACK_ID,
    SOFTWARE: ATTACK_ID,
    MITIGATION: ATTACK_ID,
    ENTITY: ENTITY_ID,
}

# Other property names
NAME = "name"
ALIASES = "aliases"
DESCRIPTION = "description"
URL = "url"
STIX_ID = "stix_id"                  # the object's or relationship's ID in MITRE's STIX file
IS_SUBTECHNIQUE = "is_subtechnique"  # Technique
SOFTWARE_TYPE = "software_type"      # Software: "malware" or "tool"
SHORTNAME = "shortname"              # Tactic, for example "defense-evasion"
EDGE_ID = "edge_id"                  # every relationship; answers cite it as [edge:<edge_id>]
SOURCE = "source"                    # every ATT&CK relationship and every USES: where the fact came from
SOURCE_ATTACK = "attack"             # from MITRE ATT&CK
SOURCE_TEXT = "text"                 # extracted from a report
EMBEDDING = "embedding"              # list of floats

# Vector index on chunk embeddings (text-embedding-3-small returns 1,536 numbers)
CHUNK_VECTOR_INDEX = "chunk_embedding"
EMBEDDING_DIMENSIONS = 1536

# Graph analytics on ATT&CK nodes (enrich.py; GDS on the ATT&CK-only projection, saved under analytics/)
PAGERANK = "pagerank"      # global PageRank score
COMMUNITY = "community"    # Louvain community number
FASTRP = "fastrp"          # FastRP vector (list of floats)
FASTRP_DIMENSIONS = 128
FASTRP_VECTOR_INDEX = {TECHNIQUE: "fastrp_technique", GROUP: "fastrp_group", SOFTWARE: "fastrp_software"}
VECTOR_INDEXES = (CHUNK_VECTOR_INDEX, *FASTRP_VECTOR_INDEX.values())

# RDF vocabulary used by load_attack.py. Classes reuse the node labels; each relationship type and each
# property gets an RDF property name in the usual lowerCamelCase form.
ONTOLOGY_IRI = "https://github.com/Wittyusername12/ctrl-alt-defeat-graphrag/ontology#"
EDGE_IRI = "https://github.com/Wittyusername12/ctrl-alt-defeat-graphrag/edge/"
RDF_RELATIONSHIP = {
    USES: "uses",
    SUBTECHNIQUE_OF: "subtechniqueOf",
    PART_OF_TACTIC: "partOfTactic",
    MITIGATES: "mitigates",
}
RDF_PROPERTY = {
    ATTACK_ID: "attackId",
    ALIASES: "alias",
    DESCRIPTION: "description",
    URL: "url",
    STIX_ID: "stixId",
    IS_SUBTECHNIQUE: "isSubtechnique",
    SOFTWARE_TYPE: "softwareType",
    SHORTNAME: "shortname",
    EDGE_ID: "edgeId",
    SOURCE: "source",
}
