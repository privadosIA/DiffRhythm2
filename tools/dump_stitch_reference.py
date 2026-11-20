"""
Chunk-stitching reference (change add-bigvgan-decoder, task 2.3). It runs the REAL `Generator.decode_audio` with a stand-in decoder that
returns, for the n-th chunk it is asked to decode (1-based), a waveform filled with n. The output then says, for every sample, which chunk
supplied it, with no weights involved. Every frame of the output must be constant (the seams are at frame boundaries); the result stored is one
chunk number per frame. No checkpoint is needed.

Writes stitch.json and MANIFEST.md to --out (default tests/fixtures/stitch).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch

import dr2_bigvgan as B
import dr2_ref as R

LENGTHS = [20, 25, 35, 40, 47, 64]


class TaggingDecoder:
    def __init__(self):
        self.calls = []

    def __call__(self, x):
        self.calls.append(int(x.shape[2]))
        return torch.full((x.shape[0], 1, x.shape[2] * B.SAMPLES_PER_FRAME), float(len(self.calls)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/tests/fixtures/stitch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from bigvgan.model import Generator

    result = {}
    for t in LENGTHS:
        gen = Generator.__new__(Generator)
        torch.nn.Module.__init__(gen)
        gen.decoder = TaggingDecoder()
        # distinct values per frame so that a chunk cut at the wrong place cannot hide behind a constant input
        latents = torch.arange(t, dtype=torch.float32).view(1, 1, t).expand(1, 64, t).contiguous()
        y = gen.decode_audio(latents, overlap=5, chunk_size=B.CHUNK)
        assert y.shape == (1, 1, t * B.SAMPLES_PER_FRAME), tuple(y.shape)
        frames = y[0, 0].view(t, B.SAMPLES_PER_FRAME)
        assert bool((frames == frames[:, :1]).all()), f"T={t}: a frame is not supplied by a single chunk"
        per_frame = [int(v) for v in frames[:, 0].tolist()]
        assert 0 not in per_frame, f"T={t}: a frame was never written"
        result[str(t)] = {"chunks_decoded": len(gen.decoder.calls), "chunk_frames": gen.decoder.calls, "chunk_per_frame": per_frame}
        print(t, result[str(t)]["chunks_decoded"], per_frame)

    with open(os.path.join(args.out, "stitch.json"), "w") as f:
        json.dump({"overlap": 5, "chunk_size": B.CHUNK, "samples_per_frame": B.SAMPLES_PER_FRAME, "lengths": result}, f, indent=1)
    R.write_manifest(args.out, "Chunk stitching reference",
                     "Which chunk (1-based) supplies each latent frame in the real `Generator.decode_audio` (chunk 20, overlap 5), by running it with a "
                     "stand-in decoder that tags its output with the chunk number. Dumped by `tools/dump_stitch_reference.py`; no weights involved.",
                     ["stitch.json"])


if __name__ == "__main__":
    main()
