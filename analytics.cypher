// Graph analytics for the CTRL ALT DEFEAT GraphRAG project: GDS (Neo4j Graph Data Science) on ATT&CK alone.
// enrich.py runs these queries in order; each query starts with a "// name:" line and ends with a semicolon.
// $graph is the in-memory projection's name. Every GDS algorithm runs with one thread (concurrency 1), and
// FastRP with a fixed seed. The results are saved under analytics/ once, on the laptop, and every machine
// loads them from those files, because FastRP's random start depends on internal node IDs.
// Every label, relationship type and property name here is defined in schemas.py (tests/test_names.py).

// name: project
// One undirected projection of the five ATT&CK labels and the four ATT&CK relationship types as MITRE records
// them (source 'attack'), so loading reports cannot change the results. Every ATT&CK entry has at least one
// such relationship, so every entry appears in the projection (enrich.py checks the node count).
MATCH (s)-[r]->(t)
WHERE r.source = 'attack' AND type(r) IN ['USES', 'SUBTECHNIQUE_OF', 'PART_OF_TACTIC', 'MITIGATES']
WITH gds.graph.project($graph, s, t,
       {sourceNodeLabels: labels(s), targetNodeLabels: labels(t), relationshipType: type(r)},
       {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS relationships;

// name: pagerank
// Global PageRank, damping 0.85 (the GDS defaults otherwise: at most 20 iterations, tolerance 1e-7).
CALL gds.pageRank.stream($graph, {dampingFactor: 0.85, concurrency: 1})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).attack_id AS attack_id, score;

// name: louvain
// Louvain communities (GDS defaults: at most 10 levels and 10 inner iterations, tolerance 1e-4).
CALL gds.louvain.stream($graph, {concurrency: 1})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).attack_id AS attack_id, communityId AS community;

// name: louvain_stats
// The modularity of Louvain's final communities, and how many levels Louvain ran.
CALL gds.louvain.stats($graph, {concurrency: 1})
YIELD communityCount, modularity, ranLevels
RETURN communityCount, modularity, ranLevels;

// name: fastrp
// FastRP: 128 numbers per node; iteration weights (0, 1, 1) mix 2- and 3-step neighbourhoods; no self-influence,
// no degree normalisation, no node features; seed 42.
CALL gds.fastRP.stream($graph, {embeddingDimension: 128, iterationWeights: [0.0, 1.0, 1.0],
                                nodeSelfInfluence: 0.0, normalizationStrength: 0.0, randomSeed: 42, concurrency: 1})
YIELD nodeId, embedding
RETURN gds.util.asNode(nodeId).attack_id AS attack_id, embedding;

// name: drop
CALL gds.graph.drop($graph, false)
YIELD graphName
RETURN graphName;
