"""
Fixtures for the C# torch.save reader (change add-weights-tools). Tiny archives written by the REAL torch.save, plus the values torch.load
returns for them, so the reader is checked against PyTorch itself. No checkpoint is needed.

Archives (all in --out, default tests/fixtures/torch_archive):
  plain.pt         OrderedDict with float32, float16, bfloat16, int64, int32, uint8 and bool tensors
  wrapped.pt       {"generator": state_dict} with a weight_g / weight_v pair, like decoder.bin
  shared.pt        two tensors that are slices of ONE storage (offsets 0 and 4)
  noncontig.pt     a transposed view saved without .contiguous()
  hostile.pt       a pickle whose GLOBAL is os.system (never executed: the C# reader must refuse it by name)
  unsupported.pt   a pickle that uses an opcode the reader does not implement (bytes -> SHORT_BINBYTES)
  deflated.pt      plain.pt with its zip entries deflated instead of stored
  short_storage.pt plain.pt with the data file of `a.weight` cut short
  cut.pt           plain.pt cut in half (the zip directory is lost)
  legacy.pt        the legacy (non-zip) serialization
expected.json: for every readable archive, the tensors in pickle order: name, dtype, shape and sha256 of the raw little-endian bytes.
Writes MANIFEST.md too.
"""
import argparse
import collections
import hashlib
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch

import dr2_ref as R

DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32", torch.uint8: "U8", torch.bool: "BOOL"}


def plain():
    g = torch.Generator().manual_seed(7)
    d = collections.OrderedDict()
    d["a.weight"] = torch.randn(2, 3, generator=g)
    d["b.bias"] = torch.randn(3, generator=g)
    d["half"] = torch.randn(2, 2, generator=g).half()
    d["brain"] = torch.randn(5, generator=g).bfloat16()
    d["ids"] = torch.tensor([3, -1, 4, 1, 5], dtype=torch.int64)
    d["small_ints"] = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    d["bytes"] = torch.tensor([0, 7, 255], dtype=torch.uint8)
    d["flags"] = torch.tensor([True, False, True])
    d["scalar"] = torch.tensor(2.5)
    return d


def rewrite(src, dst, compression=zipfile.ZIP_STORED, cut=None):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", compression) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if cut and info.filename.endswith(cut[0]):
                data = data[: cut[1]]
            zout.writestr(info.filename, data, compress_type=compression)


class Hostile:
    def __reduce__(self):
        return (os.system, ("echo should-never-run",))


def describe(path, key=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if key:
        state = state[key]
    out = []
    for name, t in state.items():
        raw = t.contiguous().flatten().view(torch.uint8).numpy().tobytes() if t.numel() else b""
        out.append({"name": name, "dtype": DTYPES[t.dtype], "shape": list(t.shape), "sha256": hashlib.sha256(raw).hexdigest()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/tests/fixtures/torch_archive")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = lambda n: os.path.join(args.out, n)

    torch.save(plain(), p("plain.pt"))
    g = torch.Generator().manual_seed(11)
    torch.save({"generator": collections.OrderedDict([
        ("conv.weight_g", torch.randn(4, 1, 1, generator=g)), ("conv.weight_v", torch.randn(4, 3, 5, generator=g)), ("conv.bias", torch.randn(4, generator=g)),
    ])}, p("wrapped.pt"))
    base = torch.arange(10, dtype=torch.float32)
    torch.save(collections.OrderedDict([("first", base[:4]), ("second", base[4:])]), p("shared.pt"))
    torch.save(collections.OrderedDict([("t", torch.arange(6, dtype=torch.float32).reshape(2, 3).t())]), p("noncontig.pt"))
    torch.save({"x": Hostile()}, p("hostile.pt"))
    buf = io.BytesIO()
    import pickle
    pickle.dump({"a": b"x"}, buf, protocol=3)
    with zipfile.ZipFile(p("unsupported.pt"), "w", zipfile.ZIP_STORED) as z:
        z.writestr("unsupported/data.pkl", buf.getvalue())
    rewrite(p("plain.pt"), p("deflated.pt"), zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(p("plain.pt")) as z:   # the data file of a.weight is the first tensor stored
        first = [i.filename for i in z.infolist() if "/data/" in i.filename][0]
    rewrite(p("plain.pt"), p("short_storage.pt"), cut=(first.split("/data/")[1] and "/data/" + first.split("/data/")[1], 8))
    with open(p("plain.pt"), "rb") as f:
        raw = f.read()
    with open(p("cut.pt"), "wb") as f:
        f.write(raw[: len(raw) // 2])
    torch.save(plain(), p("legacy.pt"), _use_new_zipfile_serialization=False)

    expected = {
        "plain.pt": describe(p("plain.pt")), "deflated.pt": describe(p("deflated.pt")), "wrapped.pt": describe(p("wrapped.pt"), "generator"),
        "shared.pt": describe(p("shared.pt")), "short_storage_first_tensor": "a.weight", "first_data_entry": first.split("/data/")[1],
    }
    with open(p("expected.json"), "w") as f:
        json.dump(expected, f, indent=1)
    R.write_manifest(args.out, "torch.save archives for the C# reader",
                     "Tiny archives written by the real `torch.save` (and variants made from them), with the tensors `torch.load` returns. Dumped by `tools/dump_torch_archive_fixtures.py`; no weights involved.",
                     [n for n in os.listdir(args.out) if n != "MANIFEST.md"])


if __name__ == "__main__":
    main()
