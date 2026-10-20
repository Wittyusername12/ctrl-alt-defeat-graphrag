// Knowledge-graph schema for the CTRL ALT DEFEAT GraphRAG project.
// One uniqueness constraint per node label, so a node can never be created twice, one vector index for
// chunk embeddings, and three for FastRP vectors on ATT&CK nodes. Every label, property and index name here
// is defined in schemas.py
// (tests/test_names.py checks that). apply_schema.py runs these statements before any data is loaded.
// Each statement uses IF NOT EXISTS, so running the file again changes nothing.

CREATE CONSTRAINT report_key IF NOT EXISTS FOR (n:Report) REQUIRE n.report_id IS UNIQUE;
CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE;
CREATE CONSTRAINT technique_key IF NOT EXISTS FOR (n:Technique) REQUIRE n.attack_id IS UNIQUE;
CREATE CONSTRAINT tactic_key IF NOT EXISTS FOR (n:Tactic) REQUIRE n.attack_id IS UNIQUE;
CREATE CONSTRAINT group_key IF NOT EXISTS FOR (n:Group) REQUIRE n.attack_id IS UNIQUE;
CREATE CONSTRAINT software_key IF NOT EXISTS FOR (n:Software) REQUIRE n.attack_id IS UNIQUE;
CREATE CONSTRAINT mitigation_key IF NOT EXISTS FOR (n:Mitigation) REQUIRE n.attack_id IS UNIQUE;
CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE;

// Chunk embeddings from text-embedding-3-small (1,536 numbers each), compared by cosine similarity.
// Quantization is switched off ('NONE') so stored vectors keep full precision; Neo4j 2026.06 added
// vector.quantization.type (NONE, SCALAR or BINARY) and the default is a quantized type.
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (n:Chunk) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'NONE'
}};

// FastRP vectors on ATT&CK nodes (128 numbers each, written by enrich.py from analytics/fastrp.npz), compared by
// cosine similarity: one index for each label the chat page compares. Quantization off, as for chunks.
CREATE VECTOR INDEX fastrp_technique IF NOT EXISTS
FOR (n:Technique) ON (n.fastrp)
OPTIONS {indexConfig: {
  `vector.dimensions`: 128,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'NONE'
}};

CREATE VECTOR INDEX fastrp_group IF NOT EXISTS
FOR (n:Group) ON (n.fastrp)
OPTIONS {indexConfig: {
  `vector.dimensions`: 128,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'NONE'
}};

CREATE VECTOR INDEX fastrp_software IF NOT EXISTS
FOR (n:Software) ON (n.fastrp)
OPTIONS {indexConfig: {
  `vector.dimensions`: 128,
  `vector.similarity_function`: 'cosine',
  `vector.quantization.type`: 'NONE'
}};
