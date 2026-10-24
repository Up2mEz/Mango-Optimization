# Original Git history

The source package was prepared in a separate submission checkout.
Original source commit: `5a89ff34847bda1d82fb643c50108ede06e4dce5`.

The GitHub publication uses the already-authorized Up2mEz connector because
GitHub CLI device authorization could not be completed. Source files are
published as a snapshot; the original commits and authorship are retained in
[greenmind-submission.bundle](greenmind-submission.bundle).

To inspect or recover the original repository after downloading the bundle:

```bash
git bundle verify greenmind-submission.bundle
git clone greenmind-submission.bundle original-submission
```

No model weights or credentials are included. The pitch PDF is submitted
separately.

## Original commit history

```text
d98fe02e9a05f69c5405a81a2976105291e30f14
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T19:17:51+07:00
Subject: GreenMind submission package: serving code, request logs, sweeps, docs

387351d2665c9dcfe109b074bd6afb3d4839f69e
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T20:13:14+07:00
Subject: Fill grader numbers for all Finals, document chunk-sizing failure and power sweep, add Apache-2.0 LICENSE

7b3f0b0285eb378f0da1ebb16b939c3853557ab7
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T21:30:22+07:00
Subject: Final #14 (S 100.0, 300/300): batched 2560, long-prefill 600, load-adaptive MTP K; Finals 12-13 added

3d31af18ad05801886f96d40d3867a55de9e3d8c
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T21:46:01+07:00
Subject: model card: base_model_relation quantized (organizer requirement), apache-2.0 licence from base model

9d9e7fc8f720a8b711a37fbb76a3fa1dc7785c7f
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:14:57+07:00
Subject: fill HF repo (Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8) and GitHub URL placeholders

5726ac7cf6271ba309421607c3f3c124837ccf5e
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:16:49+07:00
Subject: checklist: HF INT8 repo published and verified public

25ea36939432183956fbb7ce78bef8a686b83192
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:23:11+07:00
Subject: slides: deck submitted separately; remove outdated draft PDF

b433adcd58870675a2d663f82afa97e7f51b619a
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:40:52+07:00
Subject: team: ThunderPermanent (Eakkachat Sathaphon, Paradee Pimoleakaksorn, Pongsura Pongpisan, Thanakrit Samoena)

7c83391cc6cda3c97690a248bfc41a0fc5e57f8f
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:53:11+07:00
Subject: docs: architecture overview page (request path, components, general vs tuned, compliance)

5a89ff34847bda1d82fb643c50108ede06e4dce5
Author: eggchad <67070309@kmitl.ac.th>
AuthorDate: 2026-10-24T22:54:32+07:00
Subject: docs: fix clone command, public model links and separate slides submission

```
