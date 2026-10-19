# Data

Each data source in this folder keeps its own licence, which covers that source's files only.

## MITRE ATT&CK v20.0, Enterprise domain

| | |
|---|---|
| File | `attack/enterprise-attack.json`, 49,253,343 bytes |
| Source | `enterprise-attack/enterprise-attack.json` at tag `ATT&CK-v20.0` of <https://github.com/mitre/cti> (commit `50be388e93819a5d20e71cbaaeccfda157abc846`, 13 October 2026) |
| Changes | none: the file is an unmodified copy |
| SHA-256 | `66f3fe3f07ef1a2de0bd2fe4b3e8a9ea06c1d0bb2c46171bdbb57c7f7b666524` |
| Licence | `attack/LICENSE.txt`, copied unchanged from the same tag |
| Used by | `load_attack.py` |

© 2026 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation.

## AnnoCTR

The AnnoCTR reports and annotations are added with the extraction step, together with their CC BY-SA 4.0 licence and attribution notice.

## SHA-256 list

`SHA256SUMS` lists the SHA-256 (a 64-character fingerprint of a file's bytes) of every data file, one line per file: the fingerprint, two spaces, the path inside this folder. `scripts/check_setup.py` (section 6) checks every listed file, and `load_attack.py` checks the ATT&CK file before reading it, so a damaged, changed or removed file is caught.
