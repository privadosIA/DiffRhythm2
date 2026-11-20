"""
One-off conversion of the MuQ-MuLan audio tower (change add-muq-mulan-audio). ORACLE ONLY: `diffrhythm2 convert --audio` does this in C#, and its output
is compared with this file byte for byte. Runs INSIDE the reference image (see tools/README.md); the pickle is only ever opened here
(weights_only=True), never by the C# side.

Reads  <cache>/models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin   (2.5 GB, the whole MuLan)
Writes <out>/mulan_audio.safetensors   the 531 audio-tower tensors, named as in the checkpoint minus the leading `mulan.`:
                                         audio.model.model.* (MuQ-large, 525), audio.proj.* (2), audio.aggregator.* (2), audio_to_latents.* (2)
Values are copied bit for bit. While both are at hand it reports (informational) whether the 525 MuQ-large tensors equal the ones of
`OpenMuQ/MuQ-large-msd-iter`, the repository `MuQ.from_pretrained` starts from before the MuLan checkpoint overwrites it.

Usage: convert_mulan_audio.py [--cache /ckpt] [--out /ckpt]   (the output directory must be writable)
"""
import argparse
import glob
import hashlib
import json

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/ckpt")
    ap.add_argument("--out", default="/ckpt")
    args = ap.parse_args()

    bin_path = glob.glob(f"{args.cache}/models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin")[0]
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    out = {}
    for key, tensor in state.items():
        if key.startswith("mulan.audio.") or key.startswith("mulan.audio_to_latents."):
            assert tensor.dtype == torch.float32 or key.endswith("num_batches_tracked"), f"{key} is {tensor.dtype}"
            out[key[len("mulan."):]] = tensor.contiguous()
    groups = {g: sum(k.startswith(g) for k in out) for g in ("audio.model.model.", "audio.proj.", "audio.aggregator.", "audio_to_latents.")}
    assert groups == {"audio.model.model.": 525, "audio.proj.": 2, "audio.aggregator.": 2, "audio_to_latents.": 2}, groups
    assert len(out) == 531

    dtypes = sorted({str(t.dtype) for t in out.values()})
    save_file(out, f"{args.out}/mulan_audio.safetensors", metadata={"format": "pt"})   # one metadata entry: more would make the header depend on hash order

    muq_equal = None
    candidates = glob.glob(f"{args.cache}/models--OpenMuQ--MuQ-large-msd-iter/snapshots/*/model.safetensors")
    if candidates:
        base = load_file(candidates[0])
        prefix = "audio.model.model."
        same = 0
        missing = 0
        for k, t in out.items():
            if not k.startswith(prefix):
                continue
            other = base.get("model." + k[len(prefix):])
            if other is None:
                missing += 1
            elif torch.equal(other, t):
                same += 1
        muq_equal = {"equal_tensors": same, "missing_in_msd_iter": missing, "total": groups["audio.model.model."]}

    print(json.dumps({
        "tensors": len(out), "groups": groups, "dtypes": dtypes, "parameters": sum(t.numel() for t in out.values() if t.dtype == torch.float32),
        "muq_large_msd_iter_comparison": muq_equal,
        "mulan_audio.safetensors": sha256(f"{args.out}/mulan_audio.safetensors"),
    }, indent=1))


if __name__ == "__main__":
    main()
