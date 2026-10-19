# AnnoCTR: attribution and licence

The files in this folder come from the AnnoCTR corpus:

> Lukas Lange, Marc Müller, Ghazaleh Haratinezhad Torbati, Dragan Milchevski, Patrick Grau, Subhash Pujari, Annemarie Friedrich. **AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and Techniques in Cyber Threat Reports.** LREC-COLING 2024. <https://arxiv.org/abs/2404.07765>

Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries.

| | |
|---|---|
| Source | <https://github.com/boschresearch/anno-ctr-lrec-coling-2024>, folder `AnnoCTR`, commit `d510b6949e1938d47c93a43eedd562dc538439dc` (12 April 2024) |
| Files copied | `text/train`, `text/dev`, `text/test` (120 report texts) and `linking_mitre_only/train.jsonl`, `dev.jsonl`, `test.jsonl` (the experts' links to MITRE ATT&CK) |
| Changes | none: every file is an unmodified copy. When the project's code reads a report, it removes embedded base64 image data (the image's alt text and title are kept). |
| Licence | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0): the full text is in `LICENSE.txt`, copied unchanged from the source; summary at <https://creativecommons.org/licenses/by-sa/4.0/> |

The reports were donated to AnnoCTR by Intel471, Lab52, the threat intelligence division of S2 Grupo, Proofpoint, QuoIntelligence and ZScaler.

The label descriptions in `linking_mitre_only` come from MITRE ATT&CK®. © 2026 The MITRE Corporation. This work is reproduced and distributed with the permission of The MITRE Corporation (licence: `../attack/LICENSE.txt`).

Files this project derives from AnnoCTR, such as its gold-standard facts, are adaptations of the corpus and are shared under CC BY-SA 4.0 as well. Neither the AnnoCTR authors nor the report donors endorse this project. The material is provided as is, without warranties, as Section 5 of the licence states.
