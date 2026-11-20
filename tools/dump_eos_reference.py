"""
End-of-song rule fixtures (change add-dit-block-flow-matching, task 2.3). Controlled blocks are fed through the REAL
CFM.sample_block_cache (only the ODE result of each block is scripted, see tools/dr2_ref.py), so the lengths come from the
reference code itself, not from a copy of its logic. The copy in dr2_ref.end_pos is checked against it here.

Frame kinds (every channel of a frame has the same value): N = 0.0 (MSE to all-ones 1.0), E = 1.0 (MSE 0),
IN = 1 + sqrt(0.049) (MSE 0.049, inside the 0.05 threshold), OUT = 1 + sqrt(0.051) (MSE 0.051, just outside).

Writes eos.safetensors (case{i}_block{j}: [1,10,64]) and eos.json (names, block layouts, lengths from the reference) to --out.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch
from safetensors.torch import save_file

import dr2_ref as R

N, E = 0.0, 1.0
IN, OUT = 1 + math.sqrt(0.049), 1 + math.sqrt(0.051)

CASES = [
    ("two_ending_frames", [[N] * 8 + [E, E]]),
    ("all_ones", [[E] * 10]),
    ("just_above_threshold", [[N] * 8 + [E, OUT]]),
    ("just_below_threshold", [[N] * 8 + [E, IN]]),
    ("single_last_frame", [[N] * 9 + [E]]),
    ("second_block_ends", [[N] * 10, [N] * 5 + [E] * 5]),
    ("ones_only_in_the_middle", [[N] * 3 + [E] * 3 + [N] * 4]),
    ("no_ending", [[N] * 10, [N] * 10]),
]


def block(values):
    return torch.tensor(values, dtype=torch.float32)[None, :, None].expand(1, R.BLOCK, R.CHANNELS).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", default="/work/tests/fixtures/eos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    model, _ = R.load_model(args.ckpt, "fp32")
    tokens = torch.tensor(R.lyric_tokens("[verse]\nHello world\n"), dtype=torch.long)[None]
    style = torch.randn(1, 512, generator=torch.Generator().manual_seed(7))
    tensors, info = {}, []
    for i, (name, layout) in enumerate(CASES):
        blocks = [block(v) for v in layout]
        noise = R.make_noise(99, len(blocks))
        sample, rec = R.patched_sampler(model, noise, "fp32", scripted_blocks=blocks)
        latent = sample(text=tokens, duration=R.BLOCK * len(blocks), style_prompt=style, steps=2, cfg_strength=0.0, process_bar=False)
        helper_end, helper_blocks = R.end_pos(blocks)
        assert helper_end == latent.shape[1] and helper_blocks == len(rec.start), f"{name}: copy of the rule disagrees with the reference"
        for j, b in enumerate(blocks):
            tensors[f"case{i}_block{j}"] = b
        info.append({"index": i, "name": name, "blocks_given": len(blocks), "blocks_sampled": len(rec.start), "frames": int(latent.shape[1])})
        print(info[-1], flush=True)
    save_file(tensors, os.path.join(args.out, "eos.safetensors"))
    with open(os.path.join(args.out, "eos.json"), "w") as f:
        json.dump({"threshold_mse": 0.05, "kinds": {"N": N, "E": E, "IN": IN, "OUT": OUT}, "cases": info}, f, indent=1)
    R.write_manifest(args.out, "End-of-song rule (reference)",
                     "Dumped by `tools/dump_eos_reference.py`: scripted blocks through the real `sample_block_cache`; lengths in `eos.json`.",
                     ["eos.safetensors", "eos.json"])


if __name__ == "__main__":
    main()
