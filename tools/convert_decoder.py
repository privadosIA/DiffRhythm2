"""
One-off conversion of the BigVGAN decoder checkpoint: decoder.bin (a Python pickle, torch.save({"generator": state_dict})) ->
decoder.safetensors, which the C# code reads. Runs INSIDE the reference image (see tools/README.md); the pickle is only ever opened here
(weights_only=True), never by the C# side.

Values are copied bit for bit and NOTHING is folded: the weight_norm pairs stay as two tensors, so folding is done (and tested) in C# by
the core's WeightNorm. The only change is a pure rename of the pair, `X.weight_g` / `X.weight_v` ->
`X.parametrizations.weight.original0` / `original1`, the spelling the core's StrictLoader folds. Every other name is unchanged.

Usage: convert_decoder.py [--src /ckpt/decoder.bin] [--out /ckpt/decoder.safetensors]   (the output directory must be writable)
"""
import argparse
import hashlib
import json

import torch
from safetensors.torch import save_file

RENAMES = {".weight_g": ".parametrizations.weight.original0", ".weight_v": ".parametrizations.weight.original1"}


def rename(key):
    for old, new in RENAMES.items():
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/ckpt/decoder.bin")
    ap.add_argument("--out", default="/ckpt/decoder.safetensors")
    args = ap.parse_args()

    state = torch.load(args.src, map_location="cpu", weights_only=True)["generator"]
    out = {}
    for key, tensor in state.items():
        assert tensor.dtype == torch.float32, f"{key} is {tensor.dtype}, expected float32"
        new = rename(key)
        assert new not in out, f"duplicate key after renaming: {new}"
        out[new] = tensor.contiguous()
    save_file(out, args.out, metadata={"format": "pt"})   # a single metadata entry: more would make the header (and so the sha256) depend on hash order

    h = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    pairs = sum(1 for k in out if k.endswith(".original0"))
    print(json.dumps({"tensors": len(out), "weight_norm_pairs": pairs, "parameters": sum(t.numel() for t in out.values()), "sha256": h.hexdigest()}, indent=1))


if __name__ == "__main__":
    main()
