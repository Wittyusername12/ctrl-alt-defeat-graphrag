"""Read the AnnoCTR reports: their text, their chunks, and the experts' ATT&CK labels.

AnnoCTR (Lange et al., LREC-COLING 2024) is a set of cyber threat reports in which domain experts marked
mentions of techniques, tactics, groups and software and linked many of them to MITRE ATT&CK. The 120
reports with the cybersecurity layer are split train 70, dev 16, test 34. Train reports supply the prompt's
example; changes are tried on dev reports; the test reports are run once for the reported numbers.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from datafiles import DATA

ANNOCTR = DATA / "annoctr"
TEXT_DIR = ANNOCTR / "text"
LINKING_DIR = ANNOCTR / "linking_mitre_only"
SPLITS = ("train", "dev", "test")
TRACER = "lab52_2020-06-02_mustang-panda-recent-activity-dll"  # the dev report Phase 2 runs end to end

CHUNK_SIZE = 4000      # characters
CHUNK_OVERLAP = 400    # characters shared by neighbouring chunks

# Embedded images, for example ![](data:image/png;base64,iVBOR...): the base64 data is removed and the
# alt text and title around it are kept, so no human-readable text is lost.
IMAGE_DATA = re.compile(r"data:image/[A-Za-z.+-]+;base64,[A-Za-z0-9+/=]+")
ATTACK_LINK = re.compile(r"attack\.mitre\.org/(?:techniques|tactics|groups|software|mitigations)/([A-Z]+\d+)(?:/(\d+))?")


def report_ids(split: str) -> list[str]:
    return sorted(path.stem for path in (TEXT_DIR / split).glob("*.txt"))


@lru_cache(maxsize=None)
def split_of(report_id: str) -> str:
    for split in SPLITS:
        if (TEXT_DIR / split / f"{report_id}.txt").is_file():
            return split
    raise KeyError(f"no AnnoCTR report named {report_id!r} in {TEXT_DIR}")


def raw_text(report_id: str) -> str:
    return (TEXT_DIR / split_of(report_id) / f"{report_id}.txt").read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    return IMAGE_DATA.sub("", text)


def text(report_id: str) -> str:
    """The report text the pipeline uses: the file's text with embedded image data removed."""
    return clean_text(raw_text(report_id))


def chunk_spans(length: int, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """(start, end) character positions of each chunk; a 5,838-character text gives (0, 4000), (3600, 5838)."""
    return [(start, min(start + size, length)) for start in range(0, max(length - overlap, 1), size - overlap)]


def metadata(report_id: str) -> dict:
    """Vendor and date from the file name (vendor_YYYY-MM-DD_slug), and the title (the first line)."""
    vendor, date, _ = report_id.split("_", 2)
    title = raw_text(report_id).split("\n", 1)[0].strip()
    return {"vendor": vendor, "date": date, "title": title, "split": split_of(report_id)}


def attack_id_from_link(url: str) -> str | None:
    """T1574 from .../techniques/T1574, T1574.001 from .../techniques/T1574/001; None for other targets."""
    match = ATTACK_LINK.search(url)
    if match is None:
        return None
    return match.group(1) + (f".{match.group(2)}" if match.group(2) else "")


@lru_cache(maxsize=None)
def linking_rows(split: str) -> tuple[dict, ...]:
    """Every row of linking_mitre_only/<split>.jsonl (the train file holds three blank lines, skipped)."""
    with (LINKING_DIR / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def expert_labels(report_id: str) -> list[dict]:
    """The experts' ATT&CK-linked mentions in one report: mention text, entity type and ATT&CK ID."""
    labels = []
    for row in linking_rows(split_of(report_id)):
        if row["document"] != report_id:
            continue
        attack_id = attack_id_from_link(row["label_link"])
        if attack_id:
            labels.append({"mention": row["mention"], "entity_type": row["entity_type"], "attack_id": attack_id,
                           "label_title": row["label_title"]})
    return labels


def expert_facts(report_id: str, live_ids: set[str], replaced_by: dict[str, str]) -> set[str]:
    """The ATT&CK IDs the experts linked in a report, mapped to ATT&CK v20.0.

    An ID revoked in v20.0 is replaced by the ID that revoked it; an ID absent from v20.0 is dropped.
    """
    facts = set()
    for label in expert_labels(report_id):
        attack_id = replaced_by.get(label["attack_id"], label["attack_id"])
        if attack_id in live_ids:
            facts.add(attack_id)
    return facts
