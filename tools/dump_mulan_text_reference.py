"""
Golden run for the MuQ-MuLan text tower (change add-muq-mulan-text). Runs INSIDE the reference image (see tools/README.md), fp32 on CPU, through the
real `MuQMuLan`. Writes to --out (default tests/fixtures/mulan_text):

  tokens.json         for every text of tools/mulan_text_corpus.txt plus the special ones built here: ids and pieces of `tokenizer(text)`
  normalization.json  `changed`: how the reference normalizer maps every code point of planes 0 to 2 taken ALONE (only the changed ones);
                      `sequences`: input/output pairs of multi-character strings (combining marks, jamo, emoji sequences, every kind of space,
                      ligatures, Indic and Arabic text, controls) plus seeded random strings drawn from an interesting pool
  tower.safetensors   for each prompt p{i}: ids, embeddings (output of the encoder's embedding block), l0 (encoder layer 0), enc (last hidden
                      state), t0 (MuLan transformer layer 0), tr (transformer output), mean, emb (the final embedding)
  max_length.safetensors  the embedding of a text of exactly 512 positions (the longest the encoder accepts)
  meta.json           prompts, versions, sha256 of the tokenizer file, and the check that the stages chained by hand equal `mulan(texts=[t])`
  MANIFEST.md         sha256 of every file (verified by the core's FixtureDirectory)
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import dr2_ref as R  # noqa: E402

PROMPTS = [
    "Pop, Piano, Female vocals, Happy, Bright",
    "Rock, Electric guitar, Male vocals, Energetic",
    "Lo-fi, Chill",
    "música alegre con guitarra y voz femenina",
    "流行音乐, 钢琴, female vocals",
    "Electronic, Synthwave, 80s, Retro, Female vocals, Dreamy, " * 6 + "Warm pads",
    "",
]


def read_corpus(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.rstrip("\n") != "" and not l.startswith("##")]


def special_texts():
    c = chr
    return [
        "a" + c(0) + "b", "x" + c(0x1FA00) + c(0x1FA01) + " y", "a" + c(0xE000) + c(0xE001) + c(0xE002) + "b", "a" + c(0x200B) + "b", "a" + c(0xA0) + "b",
        "e" + c(0x301), "a" + c(0x308) + c(0x301) + "b", c(0x1100) + c(0x1161), c(0xAC00), c(0xD55C) + c(0xAE00), "tab" + c(9) + "and" + c(10) + "newline",
        "  " + c(0x3000) + "ideographic space" + c(0x2003), c(0x1F468) + c(0x200D) + c(0x1F469) + c(0x200D) + c(0x1F467), c(0x1F1EA) + c(0x1F1F8) + " flag",
        c(0x1F44D) + c(0x1F3FD) + " thumbs", "1" + c(0xFE0F) + c(0x20E3) + " keycap", c(0xFF21) + c(0xFF22) + c(0xFF23) + c(0xFF11) + c(0xFF12),
        c(0x2460) + c(0x2461) + c(0x2162) + c(0x00BD) + c(0x00B2), c(0xFB01) + c(0xFB02) + c(0x0133), "z" + c(0x0329) + "n", c(0x0928) + c(0x094D) + c(0x0924),
        c(0x0627) + c(0x064E) + c(0x0644), "A" * 700, "word " * 300,
        # the special tokens are recognized in the raw text before normalization, and the word marker U+2581 splits words itself
        "hello <mask> world", "<s>", "</s>", "<pad>x<unk>", "a" + c(0x2581) + "b", c(0x2581) * 2 + "lead", c(0x2581), "x<s>y", "<s", "< /s>",
    ]


def interesting_pool(norm):
    pool = [c for c in range(0x20, 0x7F)] + [0x20, 0x20, 0x0A, 0x09, 0xA0, 0x200B, 0x200D, 0x3000, 0x2003, 0xFE0F]
    pool += list(range(0x300, 0x340)) + list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))
    pool += [0xAC00, 0xAC01, 0xD7A3, 0x1F600, 0x1F3FB, 0x1F468, 0x1F469, 0x1F1E6, 0x1F1F8, 0x0928, 0x094D, 0x0924, 0x0E01, 0x0E31, 0x0627, 0x064B]
    changed = [cp for cp in range(0x80, 0x3000) if cp not in range(0xD800, 0xE000) and norm.normalize_str(chr(cp)) != chr(cp)]
    pool += random.Random(3).sample(changed, min(250, len(changed)))
    return pool


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--corpus", default="/work/tools/mulan_text_corpus.txt")
    ap.add_argument("--out", default="/work/tests/fixtures/mulan_text")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    snap = glob.glob(f"{args.ckpt}/models--xlm-roberta-base/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    assert tok.is_fast
    norm = tok.backend_tokenizer.normalizer

    # --- tokenizer corpus
    texts = read_corpus(args.corpus) + special_texts() + [p for p in PROMPTS if p not in read_corpus(args.corpus)]
    tokens = []
    for t in texts:
        ids = tok([t])["input_ids"][0]
        tokens.append({"text": t, "ids": ids, "pieces": tok.convert_ids_to_tokens(ids)})

    # --- normalization: every code point of planes 0..2 alone, then sequences
    changed = {}
    for cp in range(0, 0x30000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        out = norm.normalize_str(chr(cp))
        if out != chr(cp):
            changed[str(cp)] = out
    seqs = [t for t in texts if len(t) < 400]
    rnd = random.Random(11)
    pool = interesting_pool(norm)
    for _ in range(4000):
        seqs.append("".join(chr(rnd.choice(pool)) for _ in range(rnd.randint(1, 12))))
    sequences = [{"in": s, "out": norm.normalize_str(s)} for s in seqs]

    # --- the tower, through the real MuQMuLan (loads the audio tower too; only the text side is used)
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=args.ckpt).eval().float()
    text = mulan.mulan.text
    stages = {}
    hooks = []
    hooks.append(text.model.embeddings.register_forward_hook(lambda m, a, o: stages.__setitem__("embeddings", o.detach().clone())))
    hooks.append(text.model.encoder.layer[0].register_forward_hook(lambda m, a, o: stages.__setitem__("l0", (o[0] if isinstance(o, tuple) else o).detach().clone())))
    tensors, chained_ok, lengths = {}, [], []
    for i, p in enumerate(PROMPTS):
        inputs = text.tokenizer([p], return_tensors="pt", padding=True)
        enc = text.model(**inputs).last_hidden_state
        tr, layers = text.transformer(enc, return_all_layers=True)
        mean = tr.mean(dim=-2)
        lat = mulan.mulan.text_to_latents(mean)
        emb = torch.nn.functional.normalize(lat, p=2, dim=-1)
        direct = mulan(texts=[p])
        chained_ok.append(bool(torch.equal(direct, emb)) or float((direct - emb).abs().max()))
        t0 = text.transformer.layers[0][1](text.transformer.layers[0][0](enc) + enc) + (text.transformer.layers[0][0](enc) + enc)
        assert torch.equal(t0, layers[0]) or torch.allclose(t0, layers[0], atol=1e-6), "layer 0 of the transformer not reproduced by hand"
        tensors.update({
            f"p{i}_ids": inputs["input_ids"][0].to(torch.int64), f"p{i}_embeddings": stages["embeddings"][0], f"p{i}_l0": stages["l0"][0],
            f"p{i}_enc": enc[0], f"p{i}_t0": layers[0][0], f"p{i}_tr": tr[0], f"p{i}_mean": mean[0], f"p{i}_emb": emb[0],
        })
        lengths.append(int(inputs["input_ids"].shape[1]))
        assert abs(float(emb.norm()) - 1.0) < 1e-5 and emb.shape == (1, 512)
    for h in hooks:
        h.remove()

    longest = "a " * 510
    ids = tok([longest])["input_ids"][0]
    assert len(ids) == 512, len(ids)
    max_len = {"ids": torch.tensor(ids, dtype=torch.int64), "emb": mulan(texts=[longest])[0].detach()}

    def contiguous(d):
        return {k: v.detach().cpu().contiguous() for k, v in d.items()}

    def dump(name, obj):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=True, indent=0)

    dump("tokens.json", tokens)
    dump("normalization.json", {"changed": changed, "sequences": sequences})
    save_file(contiguous(tensors), os.path.join(args.out, "tower.safetensors"))
    save_file(contiguous(max_len), os.path.join(args.out, "max_length.safetensors"))
    import tokenizers, transformers
    meta = {
        "prompts": PROMPTS, "prompt_lengths": lengths, "texts": len(tokens), "changed_code_points": len(changed), "sequences": len(sequences),
        "chained_by_hand_equals_mulan": chained_ok,
        "tokenizer_json_sha256": sha256(os.path.join(snap, "tokenizer.json")),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    }
    dump("meta.json", meta)
    R.write_manifest(
        args.out, "MuQ-MuLan text tower golden run (fp32, CPU)",
        "Dumped by `tools/dump_mulan_text_reference.py` from the reference image `veritmo/diffrhythm2:1` through the real `MuQMuLan`. "
        "`normalization.json` maps every changed code point of planes 0 to 2 alone, plus multi-character strings; `tower.safetensors` holds the stage outputs.",
        ["tokens.json", "normalization.json", "tower.safetensors", "max_length.safetensors", "meta.json"])
    print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}, indent=1))


if __name__ == "__main__":
    main()
