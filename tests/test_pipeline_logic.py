"""The parts of extraction, linking and answering that need neither OpenAI nor Neo4j."""
import numpy as np
import pytest

import annoctr
import extract
import link
import rag


# ---------------------------------------------------------------- AnnoCTR

def test_chunk_spans_match_the_test_report():
    assert annoctr.chunk_spans(5838) == [(0, 4000), (3600, 5838)]
    assert annoctr.chunk_spans(4000) == [(0, 4000)]
    assert annoctr.chunk_spans(100) == [(0, 100)]


def test_dev_and_test_chunk_counts_and_image_removal():
    assert sum(len(annoctr.chunk_spans(len(annoctr.text(r)))) for r in annoctr.report_ids("dev")) == 50
    assert sum(len(annoctr.chunk_spans(len(annoctr.text(r)))) for r in annoctr.report_ids("test")) == 99
    removed = sum(len(annoctr.IMAGE_DATA.findall(annoctr.raw_text(r))) for r in annoctr.report_ids("dev"))
    assert removed == 7
    assert all("base64," not in annoctr.text(r) for r in annoctr.report_ids("dev"))


def test_split_sizes():
    assert [len(annoctr.report_ids(split)) for split in annoctr.SPLITS] == [70, 16, 34]


def test_expert_facts_of_the_test_report_and_the_two_splits():
    index, live, replaced = link.load_index()
    assert replaced["T1562"] == "T1685"
    assert sorted(annoctr.expert_facts(annoctr.TRACER, live, replaced)) == [
        "G0006", "G0129", "S0013", "S0154", "T1001", "T1036", "T1055", "T1071", "T1083", "T1140", "T1203",
        "T1574", "TA0002", "TA0011"]
    assert sum(len(annoctr.expert_facts(r, live, replaced)) for r in annoctr.report_ids("dev")) == 213
    assert sum(len(annoctr.expert_facts(r, live, replaced)) for r in annoctr.report_ids("test")) == 469


def test_attack_ids_from_links():
    assert annoctr.attack_id_from_link("https://attack.mitre.org/techniques/T1574") == "T1574"
    assert annoctr.attack_id_from_link("https://attack.mitre.org/techniques/T1574/001") == "T1574.001"
    assert annoctr.attack_id_from_link("http://www.bosch.com#node1fvup9pj1x44") is None


# ---------------------------------------------------------------- extraction checks

def test_prompt_example_comes_from_a_train_report_and_passes_every_check():
    assert annoctr.split_of(extract.EXAMPLE_REPORT) == "train"
    passage = extract.example_passage()
    reply = extract.example_reply()
    record = {"finish_reason": "stop", "refusal": None, "content": reply.model_dump_json()}
    mentions, relations, dropped = extract.check_reply(record, passage, "example#0")
    assert (len(mentions), len(relations), dict(dropped)) == (12, 2, {})


def test_messages_end_with_the_chunk():
    messages = extract.messages_for("Some chunk text.")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "Chunk:\nSome chunk text."


CHUNK = "APT1 used PlugX. It decoded files after download.\nThe group used Cobalt Strike."


def reply_with(mentions, uses):
    return {"finish_reason": "stop", "refusal": None,
            "content": extract.ChunkExtraction(mentions=mentions, uses=uses).model_dump_json()}


def m(id_, text, category, evidence, name=""):
    return extract.Mention(id=id_, text=text, category=category, attack_name=name, evidence=evidence, confidence=0.9)


def test_every_reason_for_dropping_an_item_is_counted():
    s1, s2, s3 = "APT1 used PlugX.", "It decoded files after download.", "The group used Cobalt Strike."
    mentions = [
        m("m1", "APT1", "group", s1, "APT1"),
        m("m2", "PlugX", "software", s1, "PlugX"),
        m("m2", "PlugX", "software", s1, "PlugX"),                       # id used twice
        m("m3", "Mimikatz", "software", s1),                              # text not in the chunk
        m("m4", "decoded files", "technique", "It decoded every file."),  # evidence not in the chunk
        m("m5", "Cobalt Strike", "software", s2),                         # text not in its evidence sentence
        m("m6", "decoded files", "technique", s2, "Deobfuscate/Decode Files or Information"),
        m("m7", "Cobalt Strike", "software", s3, "Cobalt Strike"),
    ]
    uses = [
        extract.Uses(user="m1", used="m2", evidence=s1),         # kept
        extract.Uses(user="m6", used="m1", evidence=s2),         # technique uses group: wrong way
        extract.Uses(user="m6", used="m6", evidence=s2),         # technique uses technique: impossible
        extract.Uses(user="m1", used="m9", evidence=s1),         # m9 does not exist
        extract.Uses(user="m7", used="m6", evidence="Not here."),  # evidence not in the chunk
    ]
    mentions_kept, relations, dropped = extract.check_reply(reply_with(mentions, uses), CHUNK, "r#0")
    assert [x["local_id"] for x in mentions_kept] == ["m1", "m2", "m6", "m7"]
    assert mentions_kept[0]["mention_id"] == "r#0/m1"
    assert relations == [{"user": "r#0/m1", "used": "r#0/m2", "evidence": s1}]
    assert dict(dropped) == {
        "mention id used twice": 1, "mention text not in the chunk": 1, "evidence not in the chunk": 1,
        "mention text not in its evidence sentence": 1, "relation pointing the wrong way": 1,
        "relation between categories that cannot use each other": 1,
        "relation refers to a missing or dropped mention": 1, "relation evidence not in the chunk": 1}


@pytest.mark.parametrize("record, reason", [
    ({"finish_reason": "length", "refusal": None, "content": "{\"mentions\": ["}, "reply cut off at the token cap"),
    ({"finish_reason": "content_filter", "refusal": None, "content": None}, "reply stopped by the content filter"),
    ({"finish_reason": "stop", "refusal": "I can't help with that.", "content": None}, "model refused"),
    ({"finish_reason": "stop", "refusal": None, "content": "not json"}, "reply not valid JSON in the required format"),
])
def test_unusable_replies_are_counted(record, reason):
    assert extract.check_reply(record, CHUNK, "r#0") == ([], [], {reason: 1})


def test_typographic_quotes_and_spacing_do_not_cause_drops():
    chunk = "The loader’s “LD”  parameter loads a DLL."
    mention = m("m1", "loads a DLL", "technique", "The loader's \"LD\" parameter loads a DLL.")
    kept, _, dropped = extract.check_reply(reply_with([mention], []), chunk, "r#0")
    assert len(kept) == 1 and not dropped


# ---------------------------------------------------------------- linking

@pytest.fixture(scope="module")
def index():
    return link.load_index()[0]


def mention(attack_name, text, category, evidence="", mention_id="r#0/m1"):
    return {"mention_id": mention_id, "attack_name": attack_name, "text": text, "category": category,
            "evidence": evidence or text}


@pytest.mark.parametrize("attack_name, text, category, expected, method", [
    ("PlugX", "PlugX", "software", "S0013", "exact"),
    ("", "Thoper", "software", "S0013", "exact"),                 # an alias, found through the mention text
    ("Mustang Panda", "Mustang Panda", "group", "G0129", "exact"),
    ("Carbanak", "Carbanak", "group", "G0008", "exact, tie broken by category"),
    ("Carbanak", "Carbanak", "software", "S0030", "exact, tie broken by category"),
    ("Hijack Execution Flow: DLL", "side-loading", "technique", "T1574.001", "exact"),
    ("Code Signing", "signed binaries", "technique", "T1553.002", "exact, tie broken by category"),
    ("Defense Evasion", "evade defenses", "tactic", None, "no exact match"),
    ("Cloud Accounts", "cloud accounts", "technique", None, "several exact matches"),
])
def test_exact_matching_and_tie_breaks(index, attack_name, text, category, expected, method):
    entry, how = link.exact_link(mention(attack_name, text, category), index)
    assert (entry.attack_id if entry else None, how) == (expected, method)


def test_embedding_step_links_above_the_threshold_and_leaves_the_rest_unlinked(index):
    target = next(i for i, entry in enumerate(index.entries) if entry.attack_id == "TA0005")
    vectors = np.zeros((len(index.entries), 8), dtype=np.float32)
    vectors[:, 7] = 1.0
    vectors[target] = [1, 0, 0, 0, 0, 0, 0, 0]
    close, far = np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32), np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    mentions = [mention("Defense Evasion", "evade defenses", "tactic", mention_id="r#0/m1"),
                mention("", "a made-up behaviour", "technique", mention_id="r#0/m2"),
                mention("PlugX", "PlugX", "software", mention_id="r#0/m3")]
    decisions = link.link_mentions(mentions, index, vectors, lambda texts: np.stack([close, far][:len(texts)]))
    assert decisions[0]["attack_id"] == "TA0005" and decisions[0]["method"] == "embedding (no exact match)"
    assert decisions[0]["confidence"] == pytest.approx(0.995, abs=0.001)
    assert decisions[1]["attack_id"] is None and decisions[1]["method"].startswith("unlinked (no exact match")
    assert decisions[2]["attack_id"] == "S0013" and decisions[2]["confidence"] == 1.0


def test_descriptions_lose_citations_and_link_targets():
    text = "Uses [PlugX](https://attack.mitre.org/software/S0013).(Citation: Lab52 2020)  Done."
    assert link.clean_description(text) == "Uses PlugX. Done."


# ---------------------------------------------------------------- answering

def test_citations_are_read_from_single_and_shared_brackets():
    answer = "M1040 helps [edge:attack:MITIGATES:M1040:T1055] and [chunk:r#0, edge:text:MENTIONS:r#0/m3]."
    assert rag.citations(answer) == [("edge", "attack:MITIGATES:M1040:T1055"), ("chunk", "r#0"),
                                     ("edge", "text:MENTIONS:r#0/m3")]


def test_citation_check_flags_ids_that_were_not_in_the_context():
    answer = "Yes [chunk:r#0] [edge:attack:USES:G0129:S0013] [edge:attack:MITIGATES:M9999:T1055]"
    check = rag.check_citations(answer, ["r#0"], ["attack:USES:G0129:S0013"])
    assert check == {"cited": 3, "distinct": 3, "invalid": ["edge:attack:MITIGATES:M9999:T1055"]}


def test_graph_facts_rank_seed_links_first_and_take_each_type_in_turn(monkeypatch):
    nodes = [{"element_id": "a", "key": "T1055", "name": "Process Injection", "score": 0.5},
             {"element_id": "d", "key": "S0001", "name": "Some Software", "score": 0.2},
             {"element_id": "b", "key": "M1040", "name": "Behavior Prevention on Endpoint", "score": 0.3},
             {"element_id": "e", "key": "M1038", "name": "Execution Prevention", "score": 0.05},
             {"element_id": "f", "key": "T1105", "name": "Ingress Tool Transfer", "score": 0.1},
             {"element_id": "c", "key": "G0001", "name": "Some Group", "score": 0.01}]
    edges = [{"edge_id": "attack:USES:G0001:T1105", "type": "USES", "a": "c", "b": "f", "text": None},   # two steps out
             {"edge_id": "attack:USES:S0001:T1105", "type": "USES", "a": "d", "b": "f", "text": None},   # one step out
             {"edge_id": "attack:USES:S0001:T1055", "type": "USES", "a": "d", "b": "a", "text": None},   # seed to seed
             {"edge_id": "attack:MITIGATES:M1038:T1055", "type": "MITIGATES", "a": "e", "b": "a", "text": None},
             {"edge_id": "attack:MITIGATES:M1040:T1055", "type": "MITIGATES", "a": "b", "b": "a",
              "text": "Block injection.(Citation: X)"}]
    mentions = [{"chunk_id": "r#0", "edge_id": "text:MENTIONS:r#0/m1", "mention": "injects", "key": "T1055",
                 "name": "Process Injection", "element_id": "a"},
                {"chunk_id": "r#0", "edge_id": "text:MENTIONS:r#0/m2", "mention": "injection", "key": "T1055",
                 "name": "Process Injection", "element_id": "a"},
                {"chunk_id": "r#0", "edge_id": "text:MENTIONS:r#0/m3", "mention": "the tool", "key": "S0001",
                 "name": "Some Software", "element_id": "d"}]
    lines, edge_ids = rag.graph_facts(mentions, nodes, edges, {"a", "d"})
    # one mention fact per (chunk, entity); then USES and MITIGATES in turn: seed to seed, one step, two steps
    assert edge_ids == ["text:MENTIONS:r#0/m1", "text:MENTIONS:r#0/m3", "attack:USES:S0001:T1055",
                        "attack:MITIGATES:M1040:T1055", "attack:USES:S0001:T1105", "attack:MITIGATES:M1038:T1055",
                        "attack:USES:G0001:T1105"]
    assert lines[3] == ("[edge:attack:MITIGATES:M1040:T1055] Behavior Prevention on Endpoint (M1040) MITIGATES "
                        "Process Injection (T1055) - Block injection.")
    monkeypatch.setattr(rag, "MAX_EDGES", 3)
    assert rag.graph_facts(mentions, nodes, edges, {"a", "d"})[1][2:] == [
        "attack:USES:S0001:T1055", "attack:MITIGATES:M1040:T1055", "attack:USES:S0001:T1105"]
