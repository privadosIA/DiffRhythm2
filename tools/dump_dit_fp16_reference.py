"""
INFORMATIONAL fp16 reference (change add-dit-block-flow-matching, task 2.4). It reproduces the production precision
(`model.half()`, half style embedding, the unpatched `.half()` cache times) with the SAME tokens, style and noise as the fp32 golden run
(tests/fixtures/dit/inputs.safetensors), so the two can be compared. No gating test may read this directory.

Caveat: it runs on CPU (the dev machine's reference image has no GPU in this setup), so the half-precision rounding is the CPU's, not CUDA's.
Writes run.safetensors (time_grid, block_start, block_states, latent), meta.json and MANIFEST.md to --out (default tests/fixtures/dit_fp16).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch
from safetensors.torch import load_file, save_file

import dr2_ref as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--inputs", default="/work/tests/fixtures/dit")
    ap.add_argument("--out", default="/work/tests/fixtures/dit_fp16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    meta = json.load(open(os.path.join(args.inputs, "meta.json")))
    inp = load_file(os.path.join(args.inputs, "inputs.safetensors"))
    model, _ = R.load_model(args.ckpt, "fp16")
    sample, rec = R.patched_sampler(model, inp["noise"], "fp16")
    latent = sample(text=inp["tokens"][None], duration=meta["frames"], style_prompt=inp["style"].half(), steps=meta["steps"],
                    cfg_strength=meta["cfg_strength"], process_bar=False)
    st = rec.stacked()
    fp32 = load_file(os.path.join(args.inputs, "run.safetensors"))
    n = min(latent.shape[1], fp32["latent"].shape[1])
    dev = (latent[:, :n].float() - fp32["latent"][:, :n]).abs()
    info = {"device": "cpu", "frames": int(latent.shape[1]), "vs_fp32_max_abs": dev.max().item(), "vs_fp32_mean_abs": dev.mean().item()}
    print(info)
    save_file({**{k: v.contiguous() for k, v in st.items()}, "latent": latent.float().contiguous()}, os.path.join(args.out, "run.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"precision": "fp16 (model.half(), half style)", "inputs": "tests/fixtures/dit/inputs.safetensors", **info}, f, indent=1)
    R.write_manifest(args.out, "DiT fp16 reference (informational)",
                     "Dumped by `tools/dump_dit_fp16_reference.py`. NOT a gate: it only feeds the informational fp16 test.",
                     ["run.safetensors", "meta.json"])


if __name__ == "__main__":
    main()
