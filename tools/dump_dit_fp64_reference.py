"""
Calibration aid (change add-dit-block-flow-matching, task 6.1): the SAME network, tokens, style and noise as the fp32 golden run, in float64.
It runs 45 frames (5 blocks, like tests/fixtures/dit/run_45.safetensors, whose first 4 blocks are the 40-frame golden run).
The difference between this and the fp32 reference is how much the sampler amplifies rounding by itself (Euler steps with guidance), i.e. the noise
floor under which no port can be compared. Only the C# calibration test reads it; no gate does.
Writes run.safetensors (time_grid, block_start, block_states, latent; float64), meta.json and MANIFEST.md to --out (default tests/fixtures/dit_fp64).
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
    ap.add_argument("--out", default="/work/tests/fixtures/dit_fp64")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    meta = json.load(open(os.path.join(args.inputs, "meta.json")))
    inp = load_file(os.path.join(args.inputs, "inputs.safetensors"))
    model, _ = R.load_model(args.ckpt, "fp64")
    frames = 45
    sample, rec = R.patched_sampler(model, inp["noise"].double(), "fp64")
    latent = sample(text=inp["tokens"][None], duration=frames, style_prompt=inp["style"].double(), steps=meta["steps"],
                    cfg_strength=meta["cfg_strength"], process_bar=False)
    st = rec.stacked()
    fp32 = load_file(os.path.join(args.inputs, "run_45.safetensors"))
    per_block = []
    for b in range(st["block_states"].shape[0]):
        d = (st["block_states"][b, -1] - fp32["block_states"][b, -1].double()).abs()
        per_block.append({"block": b, "fp32_vs_fp64_max_abs": d.max().item(), "mean_abs": d.mean().item()})
    print(json.dumps(per_block, indent=1))
    save_file({**{k: v.contiguous() for k, v in st.items()}, "latent": latent.contiguous()}, os.path.join(args.out, "run.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"precision": "float64", "inputs": "tests/fixtures/dit/inputs.safetensors", "frames": frames, "fp32_reference_vs_this": per_block}, f, indent=1)
    R.write_manifest(args.out, "DiT float64 reference (calibration)",
                     "Dumped by `tools/dump_dit_fp64_reference.py`. NOT a gate: it measures the sampler's own sensitivity to rounding.",
                     ["run.safetensors", "meta.json"])


if __name__ == "__main__":
    main()
