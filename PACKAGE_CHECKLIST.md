# Package checklist (remaining TODOs before submission)

Deadline per briefing: GitHub link, Hugging Face link(s) and slides PDF by Day 2 (Oct 25) 11:30.

## Placeholders to fill (`grep -rn '{{' .`)
- [ ] `Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8`: README.md, docs/MODEL_CARD.md, scripts/setup.sh (after upload, see docs/HF_UPLOAD.md)
- [ ] `{{TEAM_MEMBERS}}`: README.md, NOTICE.md, docs/MODEL_CARD.md
- [ ] `{{LICENCE_OF_BASE_MODEL}}`, `https://github.com/Up2mEz/Mango-Optimization`: docs/MODEL_CARD.md
- [x] `{{CODE_LICENCE}}`: resolved as Apache-2.0 (LICENSE, NOTICE.md); owner to confirm

## Owner actions
- [x] Upload INT8 weights to Hugging Face (done 2026-10-24 22:30: https://huggingface.co/Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8, public, 15 files, 12.42 GB, anonymous download verified, base_model_relation: quantized), make the repo public, test `scripts/setup.sh --download-int8` from a clean directory.
- [ ] Final slides PDF into `slides/` (current `pitch_deck_DRAFT.pdf` predates Finals 5-14 and still shows placeholders); remove the DRAFT file.
- [ ] Create the public GitHub repo and push (we made local commits only; nothing was pushed). The central competition repository
      decision is in the main working repo (`docs/greenmind/SUBMISSION_REPOSITORY.md`); check history/authorship policy before pushing.

## Content decisions (done by lead)
- [x] Grader History numbers filled for all 14 Finals in results/RESULTS.md (Finals 12-14 added; submitted config = Final #14, S 100.0) (baseline eta 0.00163, reference accuracy 87.8 %).
- [x] Final #14 config (batched 2560, threshold 600, adaptive MTP K) documented in README, RESULTS.md 2.9, EXPERIMENTS.md E10, MODEL_CARD; `sweep13/summary.tsv` and request logs f12-f14 added (f12 log includes a preceding Validation run).
- [x] `GM_SCHED_CHUNK` not adopted (replay 293 and 292 vs 295 of 300); documented in EXPERIMENTS.md E7b and RESULTS.md 2.8.
- [x] No Final used the Thai-tuned MTP head (documented in README, RESULTS.md 2.5 and the table note).
- [x] INT8 power sweep wording: Finals at fixed 400 W (sE saturated); 250 W is the production recommendation (replay).
- [x] `LICENSE` added (Apache-2.0, copied from the vLLM 0.32.0 wheel). **Owner: confirm the licence choice** (and that the base model's
      licence allows it, `{{LICENCE_OF_BASE_MODEL}}`).

## Still open (owner)
- [ ] EXPERIMENTS.md E5d (two API servers): cause not isolated; update only if investigated.
- [ ] Not verified: a clean-machine run of `scripts/setup.sh` (no GPU or network used while packaging); `scripts/test_slo_scheduler.py`
      was only syntax-checked.
- [ ] Team agrees to publish the request logs (`results/reqlog_*.jsonl`: metadata only, no prompt text).

## Automated checks done at packaging time
- no Hugging Face tokens, no private repo names, no absolute home paths in docs/scripts (see the final message of the packaging run)
- `bash -n` on all `.sh`, `python -m py_compile` on all `.py`
