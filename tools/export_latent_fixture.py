"""
Exports the golden run's latent as the input fixture of the decoder change (add-bigvgan-decoder). Pure file copy from tests/fixtures/dit/run.safetensors:
no model is loaded. Writes tests/fixtures/latent/latent_40.safetensors (key `latent`) and its MANIFEST.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

from safetensors.torch import load_file, save_file

import dr2_ref as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/work/tests/fixtures/dit/run.safetensors")
    ap.add_argument("--out", default="/work/tests/fixtures/latent")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    latent = load_file(args.src)["latent"].contiguous()
    assert latent.dim() == 3 and latent.shape[0] == 1 and latent.shape[2] == 64, tuple(latent.shape)
    save_file({"latent": latent}, os.path.join(args.out, "latent_40.safetensors"))
    R.write_manifest(args.out, "Golden-run latent (input of the decoder)",
                     "The sampler output of the fp32 golden run (tests/fixtures/dit, 40 frames = 8 s at 5 frames/s), exactly as `CFM.sample_block_cache` returns it: "
                     "`latent` has shape **[1, T, 64]** (batch, frames, channels). The server (and `decode_audio`) work on **[1, 64, T]**, i.e. after "
                     "`latent.transpose(1, 2)`; the decoder change must do that transposition itself. dtype float32, one frame = 9600 output samples at 48 kHz.",
                     ["latent_40.safetensors"])


if __name__ == "__main__":
    main()
