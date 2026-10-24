# Uploading the INT8 weights to Hugging Face (owner steps)

Do this yourself; nothing here has been run for you and no token is stored in this repository.

```bash
# 0. the model dir: built by scripts/setup.sh --convert (default models/w8a8-int8)
MODEL_DIR=${GM_MODELS:-models}/w8a8-int8
REPO=<your-user-or-org>/MANGO1.5-Qwen3.5-9B-W8A8-INT8      # choose the final name

# 1. login (paste a write-access token when prompted; do not put it in a file in this repo)
.venv-vllm/bin/hf auth login

# 2. model card as README.md of the HF repo (fill the placeholders first, see below)
cp docs/MODEL_CARD.md "$MODEL_DIR/README.md"

# 3. create and upload (about 12 GB)
.venv-vllm/bin/hf repo create "$REPO" --repo-type model            # add --private first if you want to check before publishing
.venv-vllm/bin/hf upload "$REPO" "$MODEL_DIR" --repo-type model
```

Afterwards, replace the placeholders (search with `grep -rn '{{' .`):

| placeholder | where | value |
|---|---|---|
| `Eggchad/MANGO1.5-Qwen3.5-9B-W8A8-INT8` | README.md, docs/MODEL_CARD.md, scripts/setup.sh, PACKAGE_CHECKLIST.md | `<user>/<repo>` from step 0 |
| `{{LICENCE_OF_BASE_MODEL}}` | docs/MODEL_CARD.md | licence id from the base model card |
| `https://github.com/Up2mEz/Mango-Optimization` | docs/MODEL_CARD.md | final public GitHub URL |
| `{{TEAM_MEMBERS}}` | README.md, NOTICE.md, docs/MODEL_CARD.md | names |
| (resolved) code licence | NOTICE.md, LICENSE | Apache-2.0 (copied from vLLM 0.32.0); owner to confirm |

Check: the repo must be public at submission time; verify with a clean download, then a server start:
`GM_HF_INT8=<user>/<repo> scripts/setup.sh --download-int8 && scripts/serve_final.sh start`.
Do not upload any other weights (the submitted config only uses these and the unchanged base model).
