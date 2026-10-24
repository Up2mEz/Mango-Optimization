#!/usr/bin/env python
"""Manual RTN W8A8-INT8 converter (compressed-tensors int-quantized) for MANGO1.5-Qwen3.5-9B.
Weights: symmetric per-output-channel int8 (absmax/127). Activations: dynamic per-token (config only).
Streams shard by shard (low RAM, CPU only).  Usage: quant_w8a8_int8.py SRC_DIR DST_DIR
"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.quantization import preset_name_to_scheme

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
torch.set_num_threads(8)

# Linear weights to quantize: text-tower projections only.
QUANT = re.compile(r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|"
                   r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$")

idx = json.load(open(f"{src}/model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
wmap, total, nq = {}, 0, 0
for sh in shards:
    out, meta = {}, {"format": "pt"}
    with safe_open(f"{src}/{sh}", "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if QUANT.match(k):
                w = t.float()
                s = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
                q = torch.round(w / s).clamp_(-128, 127).to(torch.int8)
                base = k[: -len(".weight")]
                out[k] = q.contiguous(); out[base + ".weight_scale"] = s.float().contiguous()
                nq += 1
            else:
                out[k] = t.contiguous()
    for k, v in out.items():
        wmap[k] = sh; total += v.numel() * v.element_size()
    save_file(out, f"{dst}/{sh}", metadata=meta)
    print(sh, len(out), "tensors", flush=True)
    del out
json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
          open(f"{dst}/model.safetensors.index.json", "w"), indent=2)

cfg = json.load(open(f"{src}/config.json"))
scheme = preset_name_to_scheme("W8A8", ["Linear"])
scheme.format = "int-quantized"
ignore = ["lm_head", "re:.*visual.*", "re:.*mtp.*", "re:.*linear_attn\\.in_proj_[ab]$"]
cfg["quantization_config"] = {
    "config_groups": {"group_0": json.loads(scheme.model_dump_json())},
    "format": "int-quantized", "quant_method": "compressed-tensors",
    "quantization_status": "compressed", "ignore": ignore,
    "global_compression_ratio": None, "kv_cache_scheme": None, "sparsity_config": {},
    "transform_config": {}, "version": "0.17.0",
}
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for fn in os.listdir(src):
    if fn.endswith((".json", ".jinja")) and fn not in ("config.json", "model.safetensors.index.json") and not os.path.exists(f"{dst}/{fn}"):
        shutil.copy(f"{src}/{fn}", dst)
print("quantized linears:", nq)
