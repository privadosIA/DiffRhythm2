"""
CUDA noise of the reference (change add-song-pipeline, task 7.1). Run inside the reference image WITH a GPU (docker run --gpus device=N ...), see tools/README.md.
Draws what `sample_block_cache` draws on CUDA after `torch.manual_seed(1234)`: `torch.randn(1, 10, 64, device="cuda", dtype=...)` once per block, 4 blocks,
in float16 (the service's dtype) and in float32. Writes noise_cuda.safetensors (values stored as float32 on CPU) and meta.json (GPU name) to --out.
Run it once per card and compare: the stream should not depend on the card.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HOME", "/tmp")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dr2_ref as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/tests/fixtures/song_cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this dump needs a GPU (docker run --gpus ...)"
    os.makedirs(args.out, exist_ok=True)
    out = {}
    for name, dtype in (("f16", torch.float16), ("f32", torch.float32)):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        out[f"noise_cuda_{name}"] = torch.stack([torch.randn(1, 10, 64, device="cuda", dtype=dtype) for _ in range(4)]).float().cpu()
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(args.out, f"noise_cuda{args.tag}.safetensors"))
    meta = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__, "cuda": torch.version.cuda}
    with open(os.path.join(args.out, f"meta{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
