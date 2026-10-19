r"""The benchmark: the experts' facts as gold triples, and multi-hop questions answered from them.

Run from the repository folder with the project's Python:
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe gold.py
        writes benchmark\gold_triples.json and benchmark\eval_queries.json
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe gold.py --check
        builds both files in memory and compares them with the files in the repository
    %USERPROFILE%\.venvs\ctrl-alt-defeat-graphrag\Scripts\python.exe gold.py --neo4j
        runs every question's Cypher query in Neo4j (ATT&CK part only, with the gold facts as parameters) and
        compares each answer with the answer computed here; needs load_attack.py to have run first

Gold triples. One triple (report, MENTIONS, ATT&CK entry) for every ATT&CK entry that the AnnoCTR experts
linked at least one mention to, in every dev and test report: the complete expert annotation of those
reports, not a sample. The experts' IDs are mapped to ATT&CK v20.0: a revoked ID becomes the ID that
revoked it (T1562 -> T1685); an ID missing from the v20.0 Enterprise file is listed under "dropped" with its
mentions. Each triple keeps the experts' mentions: the text, explicit or implicit, and AnnoCTR's context.

Questions. Each question needs at least two connected facts: a gold fact about a test report plus at least
one ATT&CK relationship, or the gold facts of two reports joined through the entries they share. Answers are
computed here from the gold triples and the ATT&CK v20.0 file, never from the system's own graph. The experts
link parent techniques only (no dev or test label names a sub-technique), while ATT&CK records many
relationships for sub-techniques, so a relationship that ATT&CK records for a sub-technique counts for its
parent technique: REvil uses T1566.001 Spearphishing Attachment, so REvil uses T1566 Phishing. Each question
also carries a Cypher query that computes the same answer inside Neo4j, and a reference answer in words.

Both files are written by this script only: tests/test_gold.py fails when either differs from a fresh build.
The last line starts with RESULT. Exit code 0 on success, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import annoctr
import datafiles
import load_attack
from schemas import (ATTACK_ID, GROUP, MENTIONS, MITIGATES, MITIGATION, PART_OF_TACTIC, SOFTWARE, SOURCE,
                     SOURCE_ATTACK, SUBTECHNIQUE_OF, TACTIC, TECHNIQUE, USES)

BENCHMARK = datafiles.REPO / "benchmark"
GOLD_FILE = BENCHMARK / "gold_triples.json"
QUERIES_FILE = BENCHMARK / "eval_queries.json"

GOLD_SPLITS = ("dev", "test")
GOLD_LABELS = (TECHNIQUE, TACTIC, GROUP, SOFTWARE)  # what the experts labelled; nobody labelled mitigations
# Counted from the same files on 18 Oct 2026 with a separate script (see the tracker, section 5).
EXPECTED = {
    "dev": {"reports": 16, "facts": 213, TECHNIQUE: 128, TACTIC: 38, GROUP: 10, SOFTWARE: 37},
    "test": {"reports": 34, "facts": 469, TECHNIQUE: 327, TACTIC: 85, GROUP: 13, SOFTWARE: 44},
}
ANNOCTR_SOURCE = ("https://github.com/boschresearch/anno-ctr-lrec-coling-2024, folder AnnoCTR, commit "
                  "d510b6949e1938d47c93a43eedd562dc538439dc, files linking_mitre_only/dev.jsonl and test.jsonl")
LICENCE = ("Adapted from AnnoCTR (Lange et al., LREC-COLING 2024; Copyright (c) 2023 Robert Bosch GmbH and its "
           "subsidiaries), licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); this file is "
           "shared under the same licence. ATT&CK names and relationships: (c) 2026 The MITRE Corporation, reproduced "
           "and distributed with the permission of The MITRE Corporation. Changes: see benchmark/README.md.")

VENDORS = {"intel471": "Intel 471", "lab52": "Lab52", "proofpoint": "Proofpoint",
           "quoteintelligence": "QuoIntelligence", "zscaler": "Zscaler"}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
MAX_ANSWER_ITEMS = 8  # a longer answer is too long to rate by hand on the 1-5 scale


class GoldError(Exception):
    """The data does not give what the benchmark expects."""


# ---------------------------------------------------------------- ATT&CK

class Attack:
    """ATT&CK v20.0 as the benchmark needs it: names, labels, parents, and every relationship."""

    def __init__(self, nodes: list, edges: list):
        self.node = {node.attack_id: node for node in nodes}
        self.parent = {edge.source: edge.target for edge in edges if edge.kind[1] == SUBTECHNIQUE_OF}
        self.targets: dict[tuple[str, str], set[str]] = defaultdict(set)  # (source, type) -> targets
        for edge in edges:
            self.targets[(edge.source, edge.kind[1])].add(edge.target)

    def ids(self, label: str) -> list[str]:
        return sorted(attack_id for attack_id, node in self.node.items() if node.label == label)

    def label(self, attack_id: str) -> str:
        return self.node[attack_id].label

    def name(self, attack_id: str) -> str:
        return self.node[attack_id].name

    def up(self, attack_id: str) -> str:
        """A sub-technique's parent technique; any other ID unchanged."""
        return self.parent.get(attack_id, attack_id)

    def reach(self, source: str, relationship: str) -> dict[str, list[str]]:
        """The targets of the source's relationships of one type, with technique targets counted at parent
        level: each key is a target (a parent technique for technique targets), each value the targets ATT&CK
        records for that key (the technique itself, one or more of its sub-techniques, or both)."""
        reached = defaultdict(list)
        for target in sorted(self.targets[(source, relationship)]):
            key = self.up(target) if self.label(target) == TECHNIQUE else target
            reached[key].append(target)
        return dict(reached)


def load() -> tuple[Attack, set[str], dict[str, str]]:
    nodes, edges = load_attack.read_stix(datafiles.ATTACK_FILE)
    return Attack(nodes, edges), {node.attack_id for node in nodes}, load_attack.read_revocations(datafiles.ATTACK_FILE)


def edge_id(source: str, relationship: str, target: str) -> str:
    return f"{SOURCE_ATTACK}:{relationship}:{source}:{target}"


# ---------------------------------------------------------------- gold triples

def build_gold(attack: Attack, live: set[str], replaced: dict[str, str]) -> dict:
    """The gold triples of every dev and test report, with the revoked and dropped expert IDs listed."""
    triples, revoked, dropped, counts = [], [], [], {}
    for split in GOLD_SPLITS:
        reports = annoctr.report_ids(split)
        tally = Counter()
        for report_id in reports:
            facts: dict[str, dict] = {}
            missing: dict[str, dict] = {}
            for label in annoctr.expert_labels(report_id):
                mention = {"text": label["mention"], "explicit": label["explicit"], "context": label["context"]}
                attack_id = annoctr.v20_id(label["attack_id"], live, replaced)
                if attack_id is None:
                    missing.setdefault(label["attack_id"], {
                        "split": split, "subject": report_id, "annotated_id": label["attack_id"],
                        "label_title": label["label_title"], "reason": "not in the ATT&CK v20.0 Enterprise file",
                        "mentions": []})["mentions"].append(mention)
                    continue
                if attack.label(attack_id) not in GOLD_LABELS:
                    raise GoldError(f"{report_id}: expert label {attack_id} is a {attack.label(attack_id)}")
                fact = facts.setdefault(attack_id, {
                    "split": split, "subject": report_id, "predicate": MENTIONS, "object": attack_id,
                    "object_label": attack.label(attack_id), "object_name": attack.name(attack_id),
                    "annotated_ids": [], "mentions": []})
                if label["attack_id"] not in fact["annotated_ids"]:
                    fact["annotated_ids"].append(label["attack_id"])
                fact["mentions"].append(mention)
            for attack_id, fact in sorted(facts.items(), key=lambda item: (GOLD_LABELS.index(item[1]["object_label"]),
                                                                           item[0])):
                triples.append(fact)
                tally[fact["object_label"]] += 1
                for annotated in fact["annotated_ids"]:
                    if annotated != attack_id:
                        revoked.append({"split": split, "subject": report_id, "annotated_id": annotated,
                                        "object": attack_id})
            dropped += [missing[key] for key in sorted(missing)]
        counts[split] = {"reports": len(reports), "facts": sum(tally.values()),
                         **{label: tally[label] for label in GOLD_LABELS}}
        if counts[split] != EXPECTED[split]:
            raise GoldError(f"{split}: built {counts[split]}, expected {EXPECTED[split]}")
    return {
        "about": "Gold triples of the CTRL ALT DEFEAT GraphRAG project, written by gold.py; never edited by hand.",
        "licence": LICENCE,
        "sources": {"annoctr": ANNOCTR_SOURCE,
                    "attack": f"MITRE ATT&CK v20.0 Enterprise, data/attack/enterprise-attack.json, SHA-256 "
                              f"{datafiles.sha256(datafiles.ATTACK_FILE)}"},
        "rules": [
            "A triple (report, MENTIONS, entry) holds when the experts linked at least one mention in the report to "
            "the ATT&CK entry. In the knowledge graph the same fact is a path Report -HAS_CHUNK-> Chunk -MENTIONS-> entry.",
            "Every dev and test report is covered completely: every expert link to ATT&CK in those reports is here, "
            "so a fact outside the file is not in the experts' annotation.",
            "Expert IDs are mapped to ATT&CK v20.0: a revoked ID becomes the ID that revoked it (listed under "
            "'revoked'); an ID missing from the v20.0 Enterprise file is left out (listed under 'dropped').",
            "Each mention keeps AnnoCTR's text, whether the mention names the concept (explicit, AnnoCTR class CE) or "
            "only describes it (implicit, class CI), and AnnoCTR's context around the mention.",
        ],
        "counts": counts,
        "revoked": revoked,
        "dropped": dropped,
        "triples": triples,
    }


def gold_by_report(gold: dict, split: str | None = None) -> dict[str, set[str]]:
    """report_id -> the gold ATT&CK IDs of that report, for one split or both."""
    facts: dict[str, set[str]] = defaultdict(set)
    for triple in gold["triples"]:
        if split is None or triple["split"] == split:
            facts[triple["subject"]].add(triple["object"])
    return dict(facts)


# ---------------------------------------------------------------- the questions

# (question id, pattern, report IDs, the ATT&CK entry the question names, if any). Chosen by hand from the gold
# facts of the test reports (never from system output) to cover ten question patterns, all five vendors and
# 20 of the 34 test reports, with answers of 1 to 8 items. For the "largest number" questions, the leader is the
# same whether or not relationships recorded for sub-techniques are counted.
QUESTIONS = (
    ("Q01", "mitigations_top", ("intel471_2021-12-09_emotet-returns-december-2021",), None),
    ("Q02", "mitigations_top", ("lab52_2022-01-10_tokyox-dll-side-loading-an",), None),
    ("Q03", "mitigations_top", ("proofpoint_2021-10-28_ta575-uses-squid-game-lures",), None),
    ("Q04", "mitigations_top", ("quoteintelligence_2021-02-16_dns-over-https-doh",), None),
    ("Q05", "mitigations_top", ("zscaler_2021-10-15_atomsilo-ransomware-enters-league-double",), None),
    ("Q06", "tactics", ("intel471_2021-07-14_how-cybercriminals-create-turbulence-for",), None),
    ("Q07", "tactics", ("lab52_2022-01-12_tokyox-dll-side-loading-an",), None),
    ("Q08", "tactics", ("proofpoint_2021-12-07_university-targeted-credential-phishing-campaigns",), None),
    ("Q09", "tactics", ("quoteintelligence_2021-05-20_how-the-demise-of-third",), None),
    ("Q10", "group_software", ("lab52_2021-05-17_literature-lover-targeting-colombia-with",), None),
    ("Q11", "group_software", ("lab52_2022-01-24_new-transparentribe-operation-targeting-india",), None),
    ("Q12", "groups_using_report_software", ("quoteintelligence_2021-04-28_us-sanctions-against-russias-cyber",), None),
    ("Q13", "group_techniques", ("lab52_2021-04-29_chimera-apt-updates-on-its",), "G0114"),
    ("Q14", "group_techniques", ("lab52_2022-01-24_new-transparentribe-operation-targeting-india",), "G0134"),
    ("Q15", "groups_most_techniques", ("quoteintelligence_2021-06-30_daily-operations-of-a-nigerian",), None),
    ("Q16", "software_techniques", ("intel471_2021-09-30_public-sector-cybersecurity-threats-2021",), "S0496"),
    ("Q17", "software_techniques", ("lab52_2022-01-10_tokyox-dll-side-loading-an",), "S0013"),
    ("Q18", "software_techniques", ("zscaler_2021-09-09_cloudfall-targets-researchers-and-scientists",), "S0154"),
    ("Q19", "software_techniques_missing", ("intel471_2021-12-09_emotet-returns-december-2021",), "S0367"),
    ("Q20", "shared_techniques", ("proofpoint_2021-10-28_ta575-uses-squid-game-lures",
                                  "proofpoint_2021-12-09_holiday-feelin-ta575-brings-warm"), None),
    ("Q21", "shared_techniques", ("intel471_2021-12-09_emotet-returns-december-2021",
                                  "zscaler_2021-11-16_return-emotet-malware"), None),
    ("Q22", "shared_techniques", ("lab52_2022-01-10_tokyox-dll-side-loading-an",
                                  "lab52_2022-01-12_tokyox-dll-side-loading-an"), None),
    ("Q23", "groups_all_techniques", ("proofpoint_2022-01-27_oivavoii-active-malicious-hybrid-cloud",), None),
    ("Q24", "groups_all_techniques", ("lab52_2020-08-26_a-twisted-malware-infection-chain",), None),
)

# Answers compared with the ATT&CK website on 19 Oct 2026, which showed ATT&CK v20.0 content: each page was
# fetched and the IDs it lists were compared with the computed answer, item by item. A Navigator layer file is
# the website's machine-readable list of the techniques on a group or software page. gold.py stops when a
# computed answer no longer equals the answer the website check confirmed.
WEBSITE_CHECKS = {
    "Q11": {"page": "https://attack.mitre.org/groups/G0134/", "date": "2026-10-19", "answer": ["S0115", "S0643"],
            "found": "The page lists five software for Transparent Tribe: S0115 Crimson, S0334 DarkComet, S0385 njRAT, "
                     "S0644 ObliqueRAT and S0643 Peppy. Both software the report mentions, S0115 Crimson and S0643 "
                     "Peppy, are among them."},
    "Q13": {"page": "https://attack.mitre.org/groups/G0114/G0114-enterprise-layer.json", "date": "2026-10-19",
            "answer": ["T1027", "T1036", "T1059", "T1071", "T1083"],
            "found": "The Navigator layer of Chimera's page (ATT&CK version 20) lists 82 technique entries, among them "
                     "T1027 with T1027.010, T1036 with T1036.005, T1059 with T1059.001 and T1059.003, T1071 with "
                     "T1071.001 and T1071.004, and T1083; it lists no T1547 entry."},
    "Q17": {"page": "https://attack.mitre.org/software/S0013/S0013-enterprise-layer.json", "date": "2026-10-19",
            "answer": ["T1071", "T1140", "T1204", "T1574", "T1620"],
            "found": "The Navigator layer of PlugX's page (ATT&CK version 20) lists 66 technique entries, among them "
                     "T1071 with T1071.001 and T1071.004, T1140, T1204 with T1204.002, T1574 with T1574.001, and "
                     "T1620; it lists no entry for T1055, T1129 or T1546."},
}

HOPS = {
    "mitigations_top": "Report -MENTIONS-> Technique <-[SUBTECHNIQUE_OF 0 or 1 step]- Technique <-MITIGATES- "
                       "Mitigation, counted per mitigation",
    "tactics": "Report -MENTIONS-> Technique -PART_OF_TACTIC-> Tactic",
    "group_software": "Report -MENTIONS-> Group -USES-> Software <-MENTIONS- Report",
    "groups_using_report_software": "Report -MENTIONS-> Group -USES-> Software <-MENTIONS- Report",
    "group_techniques": "Report -MENTIONS-> Group -USES-> Technique -[SUBTECHNIQUE_OF 0 or 1 step]-> Technique "
                        "<-MENTIONS- Report",
    "groups_most_techniques": "Report -MENTIONS-> Group -USES-> Technique -[SUBTECHNIQUE_OF 0 or 1 step]-> "
                              "Technique <-MENTIONS- Report, counted per group",
    "software_techniques": "Report -MENTIONS-> Software -USES-> Technique -[SUBTECHNIQUE_OF 0 or 1 step]-> "
                           "Technique <-MENTIONS- Report",
    "software_techniques_missing": "Report -MENTIONS-> Technique with no path Software -USES-> Technique "
                                   "-[SUBTECHNIQUE_OF 0 or 1 step]-> that Technique",
    "shared_techniques": "Report A -MENTIONS-> Technique <-MENTIONS- Report B",
    "groups_all_techniques": "Report -MENTIONS-> Technique <-[SUBTECHNIQUE_OF 0 or 1 step]- Technique <-USES- "
                             "Group, for every technique of the report",
}


def describe(report_id: str) -> str:
    """Intel 471's December 2021 report “How the new Emotet differs from previous versions”."""
    info = annoctr.metadata(report_id)
    year, month, _ = info["date"].split("-")
    return f"{VENDORS[info['vendor']]}'s {MONTHS[int(month) - 1]} {year} report “{info['title']}”"


def listed(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Builder:
    """Computes each question's answer from the gold facts and ATT&CK, with its supporting facts."""

    def __init__(self, attack: Attack, gold: dict):
        self.attack = attack
        self.gold = gold_by_report(gold, "test")

    # ------------------------------------------------ small helpers

    def of(self, report_id: str, label: str) -> list[str]:
        if report_id not in self.gold:
            raise GoldError(f"{report_id} is not a test report with gold facts")
        return sorted(attack_id for attack_id in self.gold[report_id] if self.attack.label(attack_id) == label)

    def named(self, attack_id: str) -> str:
        return f"{attack_id} {self.attack.name(attack_id)}"

    def via(self, technique: str, recorded: list[str]) -> str:
        """The technique's name, plus the sub-techniques that carry the relationship when the technique does not."""
        if technique in recorded:
            return self.named(technique)
        word = "sub-technique" if len(recorded) == 1 else "sub-techniques"
        return f"{self.named(technique)} (through its {word} {listed([self.named(s) for s in recorded])})"

    def support(self, source: str, relationship: str, target: str, recorded: list[str]) -> list[str]:
        """The ATT&CK relationship IDs that establish source -relationship-> target at parent level."""
        if target in recorded:
            return [edge_id(source, relationship, target)]
        ids = []
        for sub in recorded:
            ids += [edge_id(source, relationship, sub), edge_id(sub, SUBTECHNIQUE_OF, target)]
        return ids

    @staticmethod
    def gold_facts(report_id: str, attack_ids: list[str]) -> list[str]:
        return [f"{report_id} {MENTIONS} {attack_id}" for attack_id in attack_ids]

    # ------------------------------------------------ the patterns

    def mitigations_top(self, report: str, named: str | None) -> dict:
        techniques = self.of(report, TECHNIQUE)
        hits = {}
        for mitigation in self.attack.ids(MITIGATION):
            reach = self.attack.reach(mitigation, MITIGATES)
            found = {t: reach[t] for t in techniques if t in reach}
            if found:
                hits[mitigation] = found
        most = max(len(found) for found in hits.values())
        best = sorted(m for m, found in hits.items() if len(found) == most)
        runner_up = max((len(found) for found in hits.values() if len(found) < most), default=0)
        addressed = {m: listed([self.via(t, hits[m][t]) for t in sorted(hits[m])]) for m in best}
        if len(best) == 1:
            reference = (f"{self.named(best[0])} addresses {most} of the {len(techniques)} techniques the report "
                         f"describes: {addressed[best[0]]}.")
        else:
            reference = (f"{listed([self.named(m) for m in best])} each address {most} of the {len(techniques)} "
                         f"techniques the report describes. "
                         + " ".join(f"{self.named(m)} addresses {addressed[m]}." for m in best))
        reference += f" No other mitigation addresses more than {runner_up} of them."
        return {
            "question": f"Which MITRE ATT&CK mitigations address the largest number of the techniques described in "
                        f"{describe(report)}?",
            "answer": [{"attack_id": m, "name": self.attack.name(m), "support": sorted(hits[m])} for m in best],
            "reference_answer": reference,
            "gold_facts": self.gold_facts(report, techniques),
            "attack_facts": [e for m in best for t in sorted(hits[m]) for e in self.support(m, MITIGATES, t, hits[m][t])],
            "parameters": {"techniques": techniques},
            "cypher": cypher_mitigations_top(),
        }

    def tactics(self, report: str, named: str | None) -> dict:
        techniques = self.of(report, TECHNIQUE)
        placed = defaultdict(list)
        for technique in techniques:
            for tactic in sorted(self.attack.targets[(technique, PART_OF_TACTIC)]):
                placed[tactic].append(technique)
        parts = [f"{self.named(tactic)} ({listed([self.named(t) for t in placed[tactic]])})" for tactic in sorted(placed)]
        return {
            "question": f"Under which MITRE ATT&CK tactics does ATT&CK place the techniques described in "
                        f"{describe(report)}?",
            "answer": [{"attack_id": tactic, "name": self.attack.name(tactic), "support": placed[tactic]}
                       for tactic in sorted(placed)],
            "reference_answer": f"ATT&CK places the report's {len(techniques)} techniques under {len(placed)} tactics: "
                                f"{listed(parts)}.",
            "gold_facts": self.gold_facts(report, techniques),
            "attack_facts": [edge_id(t, PART_OF_TACTIC, tactic) for tactic in sorted(placed) for t in placed[tactic]],
            "parameters": {"techniques": techniques},
            "cypher": cypher_tactics(),
        }

    def group_software(self, report: str, named: str | None) -> dict:
        groups, software = self.of(report, GROUP), self.of(report, SOFTWARE)
        if len(groups) != 1:
            raise GoldError(f"{report}: group_software needs a report naming one group, found {groups}")
        group = groups[0]
        used = [s for s in software if s in self.attack.targets[(group, USES)]]
        not_used = [s for s in software if s not in used]
        reference = (f"The report names the threat group {self.named(group)} and the software "
                     f"{listed([self.named(s) for s in software])}. ATT&CK records {self.attack.name(group)} as using "
                     f"{'both: ' if len(used) == 2 else ''}{listed([self.named(s) for s in used])}.")
        if not_used:
            reference += f" ATT&CK does not record {self.attack.name(group)} using {listed([self.named(s) for s in not_used])}."
        return {
            "question": f"Which malware or tools mentioned in {describe(report)} does MITRE ATT&CK record as used by "
                        f"the threat group the report names?",
            "answer": [{"attack_id": s, "name": self.attack.name(s), "support": [group]} for s in used],
            "reference_answer": reference,
            "gold_facts": self.gold_facts(report, groups + software),
            "attack_facts": [edge_id(group, USES, s) for s in used],
            "parameters": {"groups": groups, "software": software},
            "cypher": cypher_group_software(),
        }

    def groups_using_report_software(self, report: str, named: str | None) -> dict:
        groups, software = self.of(report, GROUP), self.of(report, SOFTWARE)
        users = {g: [s for s in software if s in self.attack.targets[(g, USES)]] for g in groups}
        answer = [g for g in groups if users[g]]
        others = [g for g in groups if not users[g]]
        reference = (f"Of the {len(groups)} groups the report names ({listed([self.named(g) for g in groups])}), ATT&CK "
                     f"records {'only ' if len(answer) == 1 else ''}{listed([self.named(g) for g in answer])} as using "
                     f"{listed([self.named(s) for s in software])}, the software the report mentions.")
        if others:
            reference += (f" ATT&CK does not record {listed([self.attack.name(g) for g in others])} using "
                          f"{listed([self.attack.name(s) for s in software])}.")
        return {
            "question": f"Which of the threat groups named in {describe(report)} does MITRE ATT&CK record as using the "
                        f"malware or tools the report mentions?",
            "answer": [{"attack_id": g, "name": self.attack.name(g), "support": users[g]} for g in answer],
            "reference_answer": reference,
            "gold_facts": self.gold_facts(report, groups + software),
            "attack_facts": [edge_id(g, USES, s) for g in answer for s in users[g]],
            "parameters": {"groups": groups, "software": software},
            "cypher": cypher_groups_using_report_software(),
        }

    def _used_techniques(self, user: str, techniques: list[str]) -> dict[str, list[str]]:
        reach = self.attack.reach(user, USES)
        return {t: reach[t] for t in techniques if t in reach}

    def _uses_techniques(self, report: str, user: str, label: str) -> dict:
        techniques = self.of(report, TECHNIQUE)
        if user not in self.of(report, label):
            raise GoldError(f"{report}: {user} is not among the report's gold {label} facts")
        used = self._used_techniques(user, techniques)
        rest = [t for t in techniques if t not in used]
        name = self.attack.name(user)
        reference = (f"ATT&CK records {name} as using {len(used)} of the {len(techniques)} techniques the report "
                     f"describes: {listed([self.via(t, used[t]) for t in sorted(used)])}.")
        if rest:
            other = "technique" if len(rest) == 1 else "techniques"
            reference += (f" The report's other {other}, {listed([self.named(t) for t in rest])}, "
                          f"{'is' if len(rest) == 1 else 'are'} not among {name}'s ATT&CK techniques.")
        return {
            "question": f"Which of the techniques described in {describe(report)} does MITRE ATT&CK record {name} as "
                        f"using?",
            "answer": [{"attack_id": t, "name": self.attack.name(t), "support": used[t]} for t in sorted(used)],
            "reference_answer": reference,
            "gold_facts": self.gold_facts(report, [user] + techniques),
            "attack_facts": [e for t in sorted(used) for e in self.support(user, USES, t, used[t])],
            "parameters": {"user": user, "techniques": techniques},
            "cypher": cypher_uses_techniques(label),
        }

    def group_techniques(self, report: str, named: str | None) -> dict:
        return self._uses_techniques(report, named, GROUP)

    def software_techniques(self, report: str, named: str | None) -> dict:
        return self._uses_techniques(report, named, SOFTWARE)

    def groups_most_techniques(self, report: str, named: str | None) -> dict:
        techniques, groups = self.of(report, TECHNIQUE), self.of(report, GROUP)
        if len(groups) < 2:
            raise GoldError(f"{report}: groups_most_techniques needs a report naming two or more groups")
        used = {g: self._used_techniques(g, techniques) for g in groups}
        most = max(len(u) for u in used.values())
        best = [g for g in groups if len(used[g]) == most]
        others = sorted((g for g in groups if g not in best), key=lambda g: (-len(used[g]), g))
        reference = " ".join(
            f"{self.named(g)}: ATT&CK records {self.attack.name(g)} as using {most} of the {len(techniques)} "
            f"techniques the report describes ({listed([self.via(t, used[g][t]) for t in sorted(used[g])])})."
            for g in best)
        reference += " " + " ".join(f"{self.named(g)} uses {len(used[g])}." for g in others)
        return {
            "question": f"Which of the threat groups named in {describe(report)} does MITRE ATT&CK record as using the "
                        f"largest number of the techniques the report describes?",
            "answer": [{"attack_id": g, "name": self.attack.name(g), "support": sorted(used[g])} for g in best],
            "reference_answer": reference.strip(),
            "gold_facts": self.gold_facts(report, groups + techniques),
            "attack_facts": [e for g in best for t in sorted(used[g]) for e in self.support(g, USES, t, used[g][t])],
            "parameters": {"groups": groups, "techniques": techniques},
            "cypher": cypher_groups_most_techniques(),
        }

    def software_techniques_missing(self, report: str, named: str | None) -> dict:
        techniques = self.of(report, TECHNIQUE)
        if named not in self.of(report, SOFTWARE):
            raise GoldError(f"{report}: {named} is not among the report's gold software")
        used = self._used_techniques(named, techniques)
        missing = [t for t in techniques if t not in used]
        name = self.attack.name(named)
        reference = (f"{len(missing)} of the {len(techniques)} techniques the report describes are not among the "
                     f"techniques ATT&CK records for {name}: {listed([self.named(t) for t in missing])}. ATT&CK records "
                     f"{name} as using the other {len(used)}: {listed([self.via(t, used[t]) for t in sorted(used)])}.")
        return {
            "question": f"Which of the techniques described in {describe(report)} are missing from MITRE ATT&CK's "
                        f"record of the techniques {name} uses?",
            "answer": [{"attack_id": t, "name": self.attack.name(t), "support": []} for t in missing],
            "reference_answer": reference,
            "gold_facts": self.gold_facts(report, [named] + techniques),
            "attack_facts": [e for t in sorted(used) for e in self.support(named, USES, t, used[t])],
            "parameters": {"user": named, "techniques": techniques},
            "cypher": cypher_software_techniques_missing(),
        }

    def shared_techniques(self, reports: tuple[str, str], named: str | None) -> dict:
        first, second = reports
        a, b = self.of(first, TECHNIQUE), self.of(second, TECHNIQUE)
        shared = sorted(set(a) & set(b))
        count = "technique" if len(shared) == 1 else f"{len(shared)} techniques"
        return {
            "question": f"Which MITRE ATT&CK techniques are described in both {describe(first)} and "
                        f"{describe(second)}?",
            "answer": [{"attack_id": t, "name": self.attack.name(t), "support": []} for t in shared],
            "reference_answer": (f"The first report describes {len(a)} techniques and the second {len(b)}; both "
                                 f"describe {'one ' if len(shared) == 1 else ''}{count}: "
                                 f"{listed([self.named(t) for t in shared])}."),
            "gold_facts": self.gold_facts(first, shared) + self.gold_facts(second, shared),
            "attack_facts": [],
            "parameters": {"techniques_a": a, "techniques_b": b},
            "cypher": cypher_shared_techniques(),
        }

    def groups_all_techniques(self, report: str, named: str | None) -> dict:
        techniques = self.of(report, TECHNIQUE)
        used = {g: self._used_techniques(g, techniques) for g in self.attack.ids(GROUP)}
        every = [g for g in sorted(used) if len(used[g]) == len(techniques)]
        parts = [f"{self.named(g)} ({listed([self.via(t, used[g][t]) for t in techniques])})" for g in every]
        lead = "Only one group" if len(every) == 1 else f"{len(every)} groups"
        return {
            "question": f"Which threat groups does MITRE ATT&CK record as using every technique described in "
                        f"{describe(report)}?",
            "answer": [{"attack_id": g, "name": self.attack.name(g), "support": techniques} for g in every],
            "reference_answer": (f"{lead} in ATT&CK uses all {len(techniques)} techniques the report describes: "
                                 f"{listed(parts)}."),
            "gold_facts": self.gold_facts(report, techniques),
            "attack_facts": [e for g in every for t in techniques for e in self.support(g, USES, t, used[g][t])],
            "parameters": {"techniques": techniques},
            "cypher": cypher_groups_all_techniques(),
        }

    # ------------------------------------------------ one question

    def question(self, qid: str, pattern: str, reports: tuple[str, ...], named: str | None) -> dict:
        target = reports if pattern == "shared_techniques" else reports[0]
        built = getattr(self, pattern)(target, named)
        if not 1 <= len(built["answer"]) <= MAX_ANSWER_ITEMS:
            raise GoldError(f"{qid}: {len(built['answer'])} answer items, expected 1 to {MAX_ANSWER_ITEMS}")
        for item in built["answer"]:
            if item["attack_id"] not in built["reference_answer"]:
                raise GoldError(f"{qid}: the reference answer does not name {item['attack_id']}")
        check = WEBSITE_CHECKS.get(qid)
        if check and sorted(item["attack_id"] for item in built["answer"]) != check["answer"]:
            raise GoldError(f"{qid}: the computed answer no longer equals the answer checked on the ATT&CK website "
                            f"on {check['date']}")
        return {"id": qid, "pattern": pattern, "hops": HOPS[pattern], "reports": list(reports),
                **{key: built[key] for key in ("question", "answer", "reference_answer")},
                "supporting_facts": {"gold": built["gold_facts"], "attack": built["attack_facts"]},
                "cypher": built["cypher"], "parameters": built["parameters"],
                "checked_on_attack_website": check}


def build_queries(attack: Attack, gold: dict) -> dict:
    builder = Builder(attack, gold)
    questions = [builder.question(*entry) for entry in QUESTIONS]
    return {
        "about": "Multi-hop evaluation questions of the CTRL ALT DEFEAT GraphRAG project, written by gold.py; never "
                 "edited by hand.",
        "licence": LICENCE,
        "rules": [
            "Every question is about the test reports, which the knowledge graph holds beside ATT&CK v20.0.",
            "Answers are computed by gold.py from benchmark/gold_triples.json (test split) and the ATT&CK v20.0 "
            "Enterprise file, never from the system's own graph.",
            "A relationship that ATT&CK records for a sub-technique counts for its parent technique, because the "
            "experts' labels name parent techniques only.",
            "'cypher' computes the same answer inside Neo4j from the ATT&CK part of the graph, with the gold facts "
            "passed as 'parameters'; gold.py --neo4j runs every query and compares the answers.",
            "'reference_answer' states the answer in words (RAGAS context recall needs one); 'supporting_facts' lists "
            "the gold facts and the ATT&CK relationship IDs (as cited in answers: [edge:<id>]) the answer rests on.",
            "'checked_on_attack_website', where present, names the attack.mitre.org page whose list was compared with "
            "the answer item by item, the date, and what the page lists.",
        ],
        "patterns": HOPS,
        "questions": questions,
    }


# ---------------------------------------------------------------- the Cypher queries

def _versioned(query: str) -> str:
    return "CYPHER 25\n" + query


def _rolled(technique: str) -> str:
    """The pattern 'a technique, or one of its sub-techniques' ending in the named technique variable."""
    return f"(:`{TECHNIQUE}`)-[:`{SUBTECHNIQUE_OF}`*0..1]->({technique}:`{TECHNIQUE}`)"


def cypher_mitigations_top() -> str:
    return _versioned(
        f"MATCH (m:`{MITIGATION}`)-[:`{MITIGATES}`]->{_rolled('t')}\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques\n"
        f"WITH m, collect(DISTINCT t.`{ATTACK_ID}`) AS addressed\n"
        f"WITH max(size(addressed)) AS most, collect({{id: m.`{ATTACK_ID}`, addressed: addressed}}) AS rows\n"
        f"UNWIND rows AS row\n"
        f"WITH row, most WHERE size(row.addressed) = most\n"
        f"RETURN row.id AS answer, row.addressed AS support\n"
        f"ORDER BY answer")


def cypher_tactics() -> str:
    return _versioned(
        f"MATCH (t:`{TECHNIQUE}`)-[:`{PART_OF_TACTIC}`]->(a:`{TACTIC}`)\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques\n"
        f"RETURN a.`{ATTACK_ID}` AS answer, collect(DISTINCT t.`{ATTACK_ID}`) AS support\n"
        f"ORDER BY answer")


def cypher_group_software() -> str:
    return _versioned(
        f"MATCH (g:`{GROUP}`)-[u:`{USES}`]->(s:`{SOFTWARE}`)\n"
        f"WHERE g.`{ATTACK_ID}` IN $groups AND s.`{ATTACK_ID}` IN $software AND u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"RETURN s.`{ATTACK_ID}` AS answer, collect(DISTINCT g.`{ATTACK_ID}`) AS support\n"
        f"ORDER BY answer")


def cypher_groups_using_report_software() -> str:
    return _versioned(
        f"MATCH (g:`{GROUP}`)-[u:`{USES}`]->(s:`{SOFTWARE}`)\n"
        f"WHERE g.`{ATTACK_ID}` IN $groups AND s.`{ATTACK_ID}` IN $software AND u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"RETURN g.`{ATTACK_ID}` AS answer, collect(DISTINCT s.`{ATTACK_ID}`) AS support\n"
        f"ORDER BY answer")


def cypher_uses_techniques(label: str) -> str:
    return _versioned(
        f"MATCH (x:`{label}` {{`{ATTACK_ID}`: $user}})-[u:`{USES}`]->{_rolled('t')}\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques AND u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"RETURN DISTINCT t.`{ATTACK_ID}` AS answer\n"
        f"ORDER BY answer")


def cypher_groups_most_techniques() -> str:
    return _versioned(
        f"MATCH (g:`{GROUP}`)-[u:`{USES}`]->{_rolled('t')}\n"
        f"WHERE g.`{ATTACK_ID}` IN $groups AND t.`{ATTACK_ID}` IN $techniques AND u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"WITH g, collect(DISTINCT t.`{ATTACK_ID}`) AS used\n"
        f"WITH max(size(used)) AS most, collect({{id: g.`{ATTACK_ID}`, used: used}}) AS rows\n"
        f"UNWIND rows AS row\n"
        f"WITH row, most WHERE size(row.used) = most\n"
        f"RETURN row.id AS answer, row.used AS support\n"
        f"ORDER BY answer")


def cypher_software_techniques_missing() -> str:
    return _versioned(
        f"MATCH (t:`{TECHNIQUE}`)\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques\n"
        f"  AND NOT EXISTS {{\n"
        f"    MATCH (:`{SOFTWARE}` {{`{ATTACK_ID}`: $user}})-[u:`{USES}`]->(:`{TECHNIQUE}`)-[:`{SUBTECHNIQUE_OF}`*0..1]->(t)\n"
        f"    WHERE u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"  }}\n"
        f"RETURN t.`{ATTACK_ID}` AS answer\n"
        f"ORDER BY answer")


def cypher_shared_techniques() -> str:
    return _versioned(
        f"MATCH (t:`{TECHNIQUE}`)\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques_a AND t.`{ATTACK_ID}` IN $techniques_b\n"
        f"RETURN t.`{ATTACK_ID}` AS answer\n"
        f"ORDER BY answer")


def cypher_groups_all_techniques() -> str:
    return _versioned(
        f"MATCH (g:`{GROUP}`)-[u:`{USES}`]->{_rolled('t')}\n"
        f"WHERE t.`{ATTACK_ID}` IN $techniques AND u.`{SOURCE}` = '{SOURCE_ATTACK}'\n"
        f"WITH g, count(DISTINCT t) AS covered\n"
        f"WHERE covered = size($techniques)\n"
        f"RETURN g.`{ATTACK_ID}` AS answer\n"
        f"ORDER BY answer")


# ---------------------------------------------------------------- files

def dumps(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def build() -> dict[Path, str]:
    """Both benchmark files' text, built from the data."""
    attack, live, replaced = load()
    gold = build_gold(attack, live, replaced)
    return {GOLD_FILE: dumps(gold), QUERIES_FILE: dumps(build_queries(attack, gold))}


def write(files: dict[Path, str]) -> None:
    BENCHMARK.mkdir(exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")


def differences(files: dict[Path, str]) -> list[str]:
    """The benchmark files whose committed text differs from a fresh build."""
    problems = []
    for path, text in files.items():
        if not path.is_file():
            problems.append(f"{path.relative_to(datafiles.REPO)} is missing")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"{path.relative_to(datafiles.REPO)} differs from a fresh build (edited by hand, or the "
                            f"data or gold.py changed without rerunning gold.py)")
    return problems


def summary(files: dict[Path, str]) -> None:
    gold = json.loads(files[GOLD_FILE])
    queries = json.loads(files[QUERIES_FILE])
    for split, counts in gold["counts"].items():
        print(f"  ok       {split}: {counts['facts']} gold triples from {counts['reports']} reports ("
              + ", ".join(f"{label} {counts[label]}" for label in GOLD_LABELS) + ")")
    print(f"  ok       expert IDs revoked in v20.0 and replaced: {len(gold['revoked'])}; left out as missing from the "
          f"v20.0 Enterprise file: {len(gold['dropped'])}")
    patterns = Counter(question["pattern"] for question in queries["questions"])
    reports = {report for question in queries["questions"] for report in question["reports"]}
    print(f"  ok       {len(queries['questions'])} questions in {len(patterns)} patterns, about {len(reports)} test reports")
    checked = [question["id"] for question in queries["questions"] if question["checked_on_attack_website"]]
    print(f"  ok       answers checked on attack.mitre.org: {len(checked)} ({', '.join(checked)})")


# ---------------------------------------------------------------- the Neo4j check

def check_in_neo4j(queries: dict) -> list[str]:
    """Run every question's Cypher query in Neo4j and compare its answer (and support lists) with the file."""
    import db  # the Neo4j driver is needed only here
    from neo4j.exceptions import DriverError, Neo4jError

    problems = []
    try:
        driver, database = db.connect()
    except db.SetupError as exc:
        return [str(exc)]
    with driver:
        for question in queries["questions"]:
            qid = question["id"]
            try:
                rows = db.run_plain(driver, database, question["cypher"], **question["parameters"])
            except (Neo4jError, DriverError) as exc:
                problems.append(f"{qid}: {db.explain(exc)}")
                print(f"  PROBLEM  {problems[-1]}")
                continue
            found = {row["answer"]: sorted(row.get("support") or []) for row in rows}
            wanted = {item["attack_id"]: sorted(item["support"]) for item in question["answer"]}
            if sorted(found) != sorted(wanted):
                problems.append(f"{qid}: Neo4j answered {sorted(found)}, gold.py {sorted(wanted)}")
            elif any("support" in row for row in rows) and any(found[key] != wanted[key] for key in wanted):
                problems.append(f"{qid}: same answer, different support: Neo4j {found}, gold.py {wanted}")
            else:
                print(f"  ok       {qid} {question['pattern']:28} Neo4j's answer equals gold.py's: "
                      f"{', '.join(sorted(found))}")
                continue
            print(f"  PROBLEM  {problems[-1]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or check the gold triples and the evaluation questions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare the committed files with a fresh build")
    mode.add_argument("--neo4j", action="store_true", help="run every question's Cypher query in Neo4j")
    args = parser.parse_args(argv)
    print("Benchmark: gold triples and evaluation questions")
    problem = datafiles.check(datafiles.ATTACK_FILE)
    if problem:
        print(f"  PROBLEM  {problem}\n\nRESULT: FAILED - {problem}")
        return 1
    try:
        files = build()
    except GoldError as exc:
        print(f"  PROBLEM  {exc}\n\nRESULT: FAILED - {exc}")
        return 1
    summary(files)
    if args.check or args.neo4j:
        problems = differences(files)
        for text in problems:
            print(f"  PROBLEM  {text}")
        if args.neo4j and not problems:
            problems = check_in_neo4j(json.loads(files[QUERIES_FILE]))
        if problems:
            print(f"\nRESULT: {len(problems)} problem(s)")
            for text in problems:
                print(f"  - {text}")
            return 1
        what = "every question's Cypher query gives the same answer in Neo4j" if args.neo4j else \
            "both files equal a fresh build"
        print(f"\nRESULT: OK - {what}")
        return 0
    write(files)
    for path in files:
        print(f"  ok       wrote {path.relative_to(datafiles.REPO)} ({path.stat().st_size:,} bytes)")
    print("\nRESULT: OK - both benchmark files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
