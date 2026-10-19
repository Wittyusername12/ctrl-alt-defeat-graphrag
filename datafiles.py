"""Where the data files live, and the check of each file against its SHA-256 in data/SHA256SUMS.

A SHA-256 is a 64-character fingerprint computed from a file's bytes; any change to the file changes it.
data/SHA256SUMS lists one line per data file: the fingerprint, two spaces, the path inside data/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
MANIFEST = DATA / "SHA256SUMS"
OUT = REPO / "out"  # files the scripts produce; never committed

ATTACK_FILE = DATA / "attack" / "enterprise-attack.json"
ATTACK_TURTLE = OUT / "attack-v20.ttl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict[str, str]:
    """Path inside data/ (with forward slashes) -> expected SHA-256."""
    expected = {}
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"{MANIFEST} line {number} not understood: {line!r}")
        expected[name.strip()] = digest
    return expected


def check(path: Path) -> str | None:
    """None when the file matches its listed SHA-256; otherwise a sentence saying what is wrong."""
    name = path.relative_to(DATA).as_posix()
    expected = manifest().get(name)
    if expected is None:
        return f"{name} is not listed in {MANIFEST.name}"
    if not path.is_file():
        return f"{name} is missing from {DATA}"
    actual = sha256(path)
    if actual != expected:
        return f"{name} does not match its SHA-256 in {MANIFEST.name} (expected {expected[:12]}…, found {actual[:12]}…)"
    return None
