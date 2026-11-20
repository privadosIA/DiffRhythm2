"""
Golden run for the MuQ-MuLan audio tower (change add-muq-mulan-audio). Runs INSIDE the reference image (see tools/README.md), fp32 on CPU, through the
real `MuQMuLan`. Writes to --out (default tests/fixtures/mulan_audio):

  signals.safetensors  the deterministic test signals (24 kHz, float32): s4 (4 s, wraps into one clip), s25 (25 s, three clips, the last wrapped),
                       noise10 (10 s of seeded noise). The silent signal and the 20 s signal (the first 20 s of s25) are made in C#.
  clips.safetensors    the reference's own clip lists (`_get_all_clips`) where they are not a plain slice: s4 clip 0 and s25 clip 2.
  frontend.safetensors the mel filter bank (1025 x 128) and, for the first clip of each signal, the decibel spectrogram (128 x 1001, before the last
                       frame is dropped) and the normalized spectrogram (128 x 1000).
  stages.safetensors   for the first clip of s25 (the "primary" clip): conv (subsampling output), the outputs of Conformer layers 1, 5, 6, 11, 12
                       (before the encoder's final layer norm), the final normalized state, proj, mean, latent (before L2), emb.
  embeddings.safetensors  per-clip embeddings and per-signal embeddings of every signal; the text embeddings of the prompts of the text-tower
                       fixture (one by one: what the C# side computes) and as one padded batch (what the owner's study did), and the similarities.
  meta.json            versions, shapes, the checks done here (chained stages equal the direct call, the depth-0 transformer is the identity,
                       pos_conv_embed is never called), and `reference_nondeterminism` (filled by `--verify`).
  MANIFEST.md          sha256 of every file (verified by the core's FixtureDirectory)

`--verify DIR` recomputes everything in a fresh process and records in DIR/meta.json the largest difference with what DIR holds (the reference is
expected to be bit-for-bit reproducible across processes), then rewrites the manifest.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import dr2_ref as R  # noqa: E402

SR = 24000
CLIP = 10 * SR
FILES = ["signals.safetensors", "clips.safetensors", "frontend.safetensors", "stages.safetensors", "embeddings.safetensors", "meta.json"]


def make_signals():
    """Deterministic test signals, built in float64 and rounded to float32."""
    g = torch.Generator().manual_seed(20261105)

    def t(seconds):
        return torch.arange(int(seconds * SR), dtype=torch.float64) / SR

    n25 = t(25)
    env = 0.55 + 0.45 * torch.sin(2 * math.pi * 0.37 * n25)
    s25 = (0.30 * torch.sin(2 * math.pi * 220.0 * n25) + 0.20 * env * torch.sin(2 * math.pi * 440.5 * n25 + 0.7)
           + 0.10 * torch.sin(2 * math.pi * 1234.5 * n25 + 2.1) * (0.5 + 0.5 * torch.sin(2 * math.pi * 2.0 * n25))
           + 0.02 * torch.randn(len(n25), generator=g, dtype=torch.float64))
    n4 = t(4)
    s4 = 0.40 * torch.sin(2 * math.pi * 330.0 * n4) * (0.6 + 0.4 * torch.sin(2 * math.pi * 1.5 * n4)) + 0.02 * torch.randn(len(n4), generator=g, dtype=torch.float64)
    noise10 = 0.10 * torch.randn(10 * SR, generator=g, dtype=torch.float64)
    return {"s4": s4.float(), "s25": s25.float(), "noise10": noise10.float()}


def get_all_clips(mulan, wav):
    return mulan._get_all_clips(wav)


def run(ckpt):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    from muq import MuQMuLan
    from dump_mulan_text_reference import PROMPTS

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=ckpt).eval().float()
    top = mulan.mulan                         # MuLanModel
    audio = top.audio                         # AudioSpectrogramTransformerPretrained
    mq = audio.model.model                    # MuQModel
    conformer = mq.conformer
    assert audio.use_layer_idx == -1 and len(audio.transformer.layers) == 0, "the published configuration has no audio transformer layers"
    assert not mulan.training and not mq.training

    # hooks: layer outputs, and a counter proving pos_conv_embed is never called
    layer_out = {}
    hooks = [conformer.layers[k].register_forward_hook(lambda m, a, o, k=k: layer_out.__setitem__(k + 1, (o[0] if isinstance(o, tuple) else o).detach().clone()))
             for k in range(len(conformer.layers))]
    pos_calls = [0]
    hooks.append(conformer.pos_conv_embed.register_forward_hook(lambda m, a, o: pos_calls.__setitem__(0, pos_calls[0] + 1)))

    signals = make_signals()
    clip_lists = {name: get_all_clips(mulan, w) for name, w in signals.items()}
    frontend, stages, emb_out, clips_out = {}, {}, {}, {}
    checks = {"chained_equals_direct": {}, "clips": {}}
    fb = mq.preprocessor_melspec_2048.mel_stft.mel_scale.fb.detach().clone()
    frontend["filter_bank"] = fb
    t0 = time.time()

    for name, clips in clip_lists.items():
        checks["clips"][name] = int(clips.shape[0])
        wav = signals[name]
        # the clip lists: where a clip is a plain slice the C# side checks it against the signal; where it wraps, the reference's own clip is stored
        for i in range(clips.shape[0]):
            plain = (i + 1) * CLIP <= len(wav)
            if plain:
                assert torch.equal(clips[i], wav[i * CLIP:(i + 1) * CLIP])
            else:
                clips_out[f"{name}_{i}"] = clips[i].clone()
        per_clip = []
        for i in range(clips.shape[0]):
            clip = clips[i:i + 1]                                   # [1, 240000]
            direct = top.get_audio_latents(clip)[0]
            per_clip.append(direct)
            emb_out[f"{name}_clip{i}"] = direct.clone()
            if i == 0:
                db_full = mq.preprocessor_melspec_2048(clip)       # [1, 128, 1001]
                pre = mq.preprocessing(clip, features=["melspec_2048"])
                assert torch.equal(pre["melspec_2048"], db_full[..., :-1])
                norm = mq.normalize({"melspec_2048": pre["melspec_2048"].clone()})["melspec_2048"]   # [1, 128, 1000]
                frontend[f"{name}_db"] = db_full[0].clone()
                frontend[f"{name}_norm"] = norm[0].clone()
                if name == "s25":
                    layer_out.clear()
                    conv = mq.conv(norm)                           # [1, 250, 1024]
                    out = conformer(conv, output_hidden_states=True)
                    hs = out["hidden_states"]
                    assert len(hs) == 13 and torch.equal(hs[-1], out["last_hidden_state"])
                    last = out["last_hidden_state"]
                    proj = audio.proj(last)
                    tr, _ = audio.transformer(proj, return_all_layers=True)
                    checks["depth0_transformer_is_identity"] = bool(torch.equal(tr, proj))
                    mean = tr.mean(dim=-2)
                    latent = top.audio_to_latents(mean)
                    emb = torch.nn.functional.normalize(latent, p=2, dim=-1)
                    checks["chained_equals_direct"][name] = bool(torch.equal(emb[0], direct)) or float((emb[0] - direct).abs().max())
                    stages.update({"conv": conv[0], "last": last[0], "proj": proj[0], "mean": mean[0], "latent": latent[0], "emb": emb[0]})
                    for k in (1, 5, 6, 11, 12):
                        stages[f"layer{k}"] = layer_out[k][0]
                    # hidden_states[k] is the INPUT of layer k + 1 and the output of layer k (index 12 is the final-normed output)
                    for k in (1, 5, 6, 11):
                        assert torch.equal(hs[k], layer_out[k]), f"hidden_states[{k}] is not the output of layer {k}"
                    assert torch.equal(conformer.layer_norm(layer_out[12]), last)
                    # the whole encoder fed by hand must equal the direct call
                    h = conv
                    rope = conformer.embed_positions(h)
                    for layer in conformer.layers:
                        h = layer(h, attention_mask=None, relative_position_embeddings=rope)[0]
                    assert torch.equal(conformer.layer_norm(h), last)
        stack = torch.stack(per_clip)
        sig = mulan(wavs=wav.unsqueeze(0))[0]                       # the public call: clips one by one, mean of latents
        assert torch.equal(sig, stack.mean(dim=0)), "the signal embedding is not the plain mean of the clip embeddings"
        emb_out[name] = sig.clone()
        emb_out[f"{name}_length"] = sig.norm().reshape(1)
        checks.setdefault("clip_embeddings_unit_length", {})[name] = float(max(abs(float(c.norm()) - 1.0) for c in per_clip))
    elapsed = time.time() - t0

    # text side: one by one (what the C# side computes) and as a padded batch (what the owner's study did)
    one = torch.stack([mulan(texts=[p])[0] for p in PROMPTS])
    batch = mulan(texts=PROMPTS)
    emb_out["text_one"] = one
    emb_out["text_batch"] = batch.detach().clone()
    sig_names = list(signals)
    audio_embs = torch.stack([emb_out[n] for n in sig_names])
    emb_out["sim_one"] = mulan.calc_similarity(audio_embs, one)
    emb_out["sim_batch"] = mulan.calc_similarity(audio_embs, batch)
    emb_out["audio_audio"] = mulan.calc_similarity(audio_embs, audio_embs)

    for h in hooks:
        h.remove()
    import transformers
    meta = {
        "signals": {n: int(len(w)) for n, w in signals.items()}, "signal_order": sig_names, "prompts": PROMPTS,
        "sample_rate": SR, "clip_samples": CLIP, "checks": checks, "pos_conv_embed_called": pos_calls[0] > 0, "pos_conv_embed_calls": pos_calls[0],
        "mel": {"stat": dict(mq.stat) if hasattr(mq, "stat") else None, "hop_length": mq.hop_length, "n_fold": mq.n_fold,
                "filter_bank_shape": list(fb.shape)},
        "hidden_states": 13, "elapsed_seconds_all_signals": round(elapsed, 1), "reference_nondeterminism": None,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "threads": torch.get_num_threads()},
    }
    tensors = {
        "signals.safetensors": signals, "clips.safetensors": clips_out, "frontend.safetensors": frontend,
        "stages.safetensors": stages, "embeddings.safetensors": emb_out,
    }
    return {f: {k: v.detach().cpu().contiguous().clone() for k, v in d.items()} for f, d in tensors.items()}, meta


def write(out, tensors, meta):
    os.makedirs(out, exist_ok=True)
    for f, d in tensors.items():
        save_file(d, os.path.join(out, f))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    manifest(out)


def manifest(out):
    R.write_manifest(
        out, "MuQ-MuLan audio tower golden run (fp32, CPU)",
        "Dumped by `tools/dump_mulan_audio_reference.py` from the reference image `mediaorquestator/diffrhythm2:1` through the real `MuQMuLan`. "
        "`signals` are deterministic test signals, `frontend` the mel filter bank and spectrograms, `stages` the stage outputs of the first clip of `s25`, "
        "`embeddings` the clip, signal and text embeddings and their similarities.",
        FILES)


def verify(out, tensors, meta):
    worst = 0.0
    for f, d in tensors.items():
        old = load_file(os.path.join(out, f))
        assert set(old) == set(d), f"{f}: different tensor names"
        for k, v in d.items():
            worst = max(worst, float((old[k].double() - v.double()).abs().max())) if v.numel() else worst
    with open(os.path.join(out, "meta.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    stored["reference_nondeterminism"] = worst
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(stored, fh, indent=1, sort_keys=True)
    manifest(out)
    print("largest difference between two processes:", worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", default="/work/tests/fixtures/mulan_audio")
    ap.add_argument("--verify", default=None, help="recompute and record the difference with the files of this directory")
    args = ap.parse_args()
    tensors, meta = run(args.ckpt)
    if args.verify:
        verify(args.verify, tensors, meta)
    else:
        write(args.out, tensors, meta)
        print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}, indent=1))


if __name__ == "__main__":
    main()
