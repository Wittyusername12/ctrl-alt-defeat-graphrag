r"""Section IV of the final report: topology of the ATT&CK projection, PageRank, Louvain communities and FastRP.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe docs\final\graph_metrics.py

Steps:
  1. Rebuild the GDS projection from the ATT&CK v20.0 file (load_attack.read_stix): the five ATT&CK labels and the
     four relationship types, undirected. Print the numbers of Table II: relationships, density, degree, connected
     components, shortest paths (exact breadth-first search from every node), clustering coefficient.
  2. Print the degree shares quoted in Fig. 3's caption.
  3. Read the committed GDS results (analytics/attack_analytics.csv): PageRank's top entries, the Spearman
     correlation of PageRank and degree (tied ranks averaged), the modularity recomputed from the communities, and
     each community's make-up for Table III.
  4. Read the committed FastRP vectors (analytics/fastrp.npz): the groups nearest to G0129 Mustang Panda, and the
     USES targets LuminousMoth shares with Mustang Panda.
No OpenAI call, no Neo4j; nothing is written. The last line starts with RESULT.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import datafiles  # noqa: E402
import load_attack  # noqa: E402

LABELS = ("Technique", "Tactic", "Group", "Software", "Mitigation")


def bfs(start: str, neighbours: dict[str, set[str]]) -> dict[str, int]:
    distance = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in neighbours[node]:
            if other not in distance:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def main() -> int:
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    label = {node.attack_id: node.label for node in nodes}
    name = {node.attack_id: node.name for node in nodes}
    ids = sorted(label)
    n = len(ids)

    print("1. Topology (Table II)")
    adjacent: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacent[edge.source].append(edge.target)
        adjacent[edge.target].append(edge.source)
    degree = {a: len(adjacent[a]) for a in ids}
    values = np.array([degree[a] for a in ids], dtype=float)
    pairs = {frozenset((edge.source, edge.target)) for edge in edges}
    print(f"  nodes {n}, relationships {len(edges)} ({int(values.sum())} counted in both directions), "
          f"distinct node pairs {len(pairs)}, self-loops {sum(1 for e in edges if e.source == e.target)}")
    print(f"  density {len(pairs) / (n * (n - 1) / 2):.5f}")
    top = max(ids, key=lambda a: degree[a])
    print(f"  degree mean {values.mean():.2f}, median {np.median(values):.0f}, maximum {int(values.max())} ({top})")
    for kind in LABELS:
        own = np.array([degree[a] for a in ids if label[a] == kind], dtype=float)
        print(f"  {kind:10s} n={len(own):4d} mean degree {own.mean():6.2f} median {np.median(own):5.1f} "
              f"max {int(own.max())}")
    neighbours = {a: set(adjacent[a]) - {a} for a in ids}
    components, seen = [], set()
    for a in ids:
        if a not in seen:
            component = bfs(a, neighbours)
            seen |= set(component)
            components.append(component)
    print(f"  connected components {len(components)}")
    lengths, eccentricity = Counter(), {}
    for a in ids:
        distance = bfs(a, neighbours)
        eccentricity[a] = max(distance.values())
        lengths.update(d for d in distance.values() if d)
    ordered = sum(lengths.values())
    print(f"  average shortest path {sum(d * c for d, c in lengths.items()) / ordered:.3f}, "
          f"diameter {max(eccentricity.values())}, radius {min(eccentricity.values())}")
    within3 = sum(c for d, c in lengths.items() if d <= 3)
    print(f"  ordered pairs within 3 hops {within3:,} of {ordered:,} ({within3 / ordered:.1%})")
    two_hop = [sum(1 for d in bfs(a, neighbours).values() if 1 <= d <= 2) for a in ids]
    print(f"  entries within 2 hops of one entry: mean {np.mean(two_hop):.0f} ({np.mean(two_hop) / n:.1%})")
    clustering = []
    for a in ids:
        k = len(neighbours[a])
        links = sum(1 for u in neighbours[a] for v in neighbours[u] if v in neighbours[a]) / 2
        clustering.append(links / (k * (k - 1) / 2) if k >= 2 else 0.0)
    print(f"  average local clustering coefficient {np.mean(clustering):.3f}")

    print("\n2. Degree shares (Fig. 3)")
    for kind in LABELS:
        own = [degree[a] for a in ids if label[a] == kind]
        print(f"  {kind:10s} degree <= 10: {sum(1 for d in own if d <= 10) / len(own):.1%}; "
              f">= 50: {sum(1 for d in own if d >= 50) / len(own):.1%}; >= 100: {sum(1 for d in own if d >= 100)}; "
              f"maximum {max(own)}")

    print("\n3. PageRank and Louvain (committed GDS results)")
    with (ROOT / "analytics" / "attack_analytics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pagerank = {row["attack_id"]: float(row["pagerank"]) for row in rows}
    community = {row["attack_id"]: int(row["community"]) for row in rows}
    if set(pagerank) != set(ids):
        print("  PROBLEM  the analytics file does not cover the ATT&CK nodes")
        return 1
    print("  top 10: " + ", ".join(f"{a} {pagerank[a]:.3f}" for a in sorted(ids, key=lambda a: -pagerank[a])[:10]))
    for kind in ("Group", "Software", "Mitigation"):
        best = sorted((a for a in ids if label[a] == kind), key=lambda a: -pagerank[a])[:3]
        print(f"  top {kind}: " + ", ".join(f"{a} {name[a]} {pagerank[a]:.3f}" for a in best))
    by_degree = [degree[a] for a in ids]
    by_score = [pagerank[a] for a in ids]
    print(f"  Spearman (tied ranks averaged) {stats.spearmanr(by_score, by_degree).statistic:.4f}, "
          f"Pearson {stats.pearsonr(by_score, by_degree).statistic:.4f}")
    m = len(edges)
    inside, degree_sum = Counter(), Counter()
    for edge in edges:
        if community[edge.source] == community[edge.target]:
            inside[community[edge.source]] += 1
    for a in ids:
        degree_sum[community[a]] += degree[a]
    modularity = sum(inside[c] / m - (degree_sum[c] / (2 * m)) ** 2 for c in degree_sum)
    print(f"  communities {len(degree_sum)}, modularity recomputed {modularity:.5f}")
    tactics_of = defaultdict(set)
    for edge in edges:
        if edge.kind[1] == "PART_OF_TACTIC":
            tactics_of[edge.source].add(edge.target)
    for number, c in enumerate(sorted(degree_sum, key=lambda c: -sum(1 for a in ids if community[a] == c)), 1):
        members = [a for a in ids if community[a] == c]
        counts = Counter(label[a] for a in members)
        make_up = "/".join(str(counts[kind]) for kind in LABELS)
        tactics = Counter(t for a in members if label[a] == "Technique" for t in tactics_of[a]).most_common(3)
        leaders = []
        for kind in ("Technique", "Group", "Software"):
            best = max((a for a in members if label[a] == kind), key=lambda a: pagerank[a], default=None)
            leaders.append(f"{best} {name[best]}" if best else "-")
        print(f"  {number}. size {len(members)} Te/Ta/G/S/M {make_up}; techniques' tactics "
              + ", ".join(f"{name[t]} {k}" for t, k in tactics) + "; highest PageRank " + "; ".join(leaders))

    print("\n4. FastRP (committed vectors)")
    saved = np.load(ROOT / "analytics" / "fastrp.npz", allow_pickle=True)
    order = [str(a) for a in saved["attack_ids"]]
    vectors = saved["vectors"].astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = vectors @ vectors[order.index("G0129")]
    groups = sorted((j for j, a in enumerate(order) if label[a] == "Group" and a != "G0129"), key=lambda j: -cosine[j])
    print("  nearest groups to G0129: " + ", ".join(
        f"{order[j]} {name[order[j]]} {(1 + cosine[j]) / 2:.3f} (cosine {cosine[j]:.3f})" for j in groups[:5]))
    uses = defaultdict(set)
    for edge in edges:
        if edge.kind[1] == "USES":
            uses[edge.source].add(edge.target)
    shared = uses["G1014"] & uses["G0129"]
    aliases = next(node.aliases for node in nodes if node.attack_id == "G0129")
    print(f"  LuminousMoth USES targets {len(uses['G1014'])}, shared with Mustang Panda {len(shared)}; "
          f"'LUMINOUS MOTH' among Mustang Panda's aliases: {'LUMINOUS MOTH' in aliases}")
    print("\nRESULT: OK - Section IV numbers printed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
