"""
One-off conversion of the MuQ-MuLan text tower (change add-muq-mulan-text). Runs INSIDE the reference image (see tools/README.md); the pickle is
only ever opened here (weights_only=True), never by the C# side.

Reads  <cache>/models--OpenMuQ--MuQ-MuLan-large/snapshots/*/pytorch_model.bin   (2.5 GB, the whole MuLan: audio tower included)
Writes <out>/mulan_text.safetensors          the 273 text-tower tensors, named as in the checkpoint minus the leading `mulan.`:
                                             text.model.*  (XLM-RoBERTa-base, 199), text.transformer.* (72), text_to_latents.* (2)
       <out>/mulan_text_tokenizer.json       the tokenizer.json of xlm-roberta-base (what AutoTokenizer builds the fast tokenizer from)
Values are copied bit for bit. While both are at hand it asserts that the 199 encoder tensors equal xlm-roberta-base's.

Usage: convert_mulan_text.py [--cache /ckpt] [--out /ckpt]   (the output directory must be writable)
"""
import argparse
import glob
import hashlib
import json
import shutil

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
        if key.startswith("mulan.text.") or key.startswith("mulan.text_to_latents."):
            assert tensor.dtype == torch.float32, f"{key} is {tensor.dtype}"
            out[key[len("mulan."):]] = tensor.contiguous()
    groups = {g: sum(k.startswith(g) for k in out) for g in ("text.model.", "text.transformer.", "text_to_latents.")}
    assert groups == {"text.model.": 199, "text.transformer.": 72, "text_to_latents.": 2}, groups
    assert len(out) == 273

    base_path = glob.glob(f"{args.cache}/models--xlm-roberta-base/snapshots/*/model.safetensors")[0]
    base = load_file(base_path)
    for key, tensor in out.items():
        if key.startswith("text.model."):
            expected = base["roberta." + key[len("text.model."):]]
            assert torch.equal(expected, tensor), f"{key} differs from xlm-roberta-base"

    save_file(out, f"{args.out}/mulan_text.safetensors", metadata={"format": "pt"})   # one metadata entry: more would make the header depend on hash order
    tokenizer_src = glob.glob(f"{args.cache}/models--xlm-roberta-base/snapshots/*/tokenizer.json")[0]
    shutil.copyfile(tokenizer_src, f"{args.out}/mulan_text_tokenizer.json")

    print(json.dumps({
        "tensors": len(out), "groups": groups, "parameters": sum(t.numel() for t in out.values()),
        "encoder_equals_xlm_roberta_base": True,
        "mulan_text.safetensors": sha256(f"{args.out}/mulan_text.safetensors"),
        "mulan_text_tokenizer.json": sha256(f"{args.out}/mulan_text_tokenizer.json"),
    }, indent=1))


if __name__ == "__main__":
    main()
