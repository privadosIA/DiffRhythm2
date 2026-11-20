"""
INFORMATIONAL fp16 reference (change add-bigvgan-decoder, task 2.4): the production precision (`Generator(...).half()`, half latent) decoding the same
40-frame golden-run latent as tests/fixtures/bigvgan/decode_40.safetensors. No gating test may read this directory.

Caveat: it runs on CPU, so the half-precision rounding is the CPU's, not CUDA's. Writes decode_40.safetensors (wave, float32 copy of the half output),
meta.json and MANIFEST.md to --out (default tests/fixtures/bigvgan_fp16).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch
from safetensors.torch import load_file, save_file

import dr2_bigvgan as B
import dr2_ref as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--latent", default="/work/tests/fixtures/latent/latent_40.safetensors")
    ap.add_argument("--fp32", default="/work/tests/fixtures/bigvgan/decode_40.safetensors")
    ap.add_argument("--out", default="/work/tests/fixtures/bigvgan_fp16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    gen = B.load_generator(args.ckpt, "fp16")
    full = load_file(args.latent)["latent"].transpose(1, 2).contiguous().half()
    wave = gen.decode_audio(full, overlap=5, chunk_size=B.CHUNK)
    ref32 = load_file(args.fp32)["wave"]
    diff = (wave.float() - ref32).abs()
    meta = {"dtype": str(wave.dtype), "samples": int(wave.shape[2]), "max_abs_vs_fp32_reference": diff.max().item(), "mean_abs_vs_fp32_reference": diff.mean().item(),
            "nonfinite": int((~torch.isfinite(wave)).sum().item()), "versions": {"torch": torch.__version__}}
    save_file({"wave": wave.float().contiguous()}, os.path.join(args.out, "decode_40.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    R.write_manifest(args.out, "BigVGAN fp16 reference (informational)",
                     "Production-style half precision on CPU for the same latent as `tests/fixtures/bigvgan/decode_40.safetensors`. **Never a gate.**",
                     ["decode_40.safetensors", "meta.json"])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
