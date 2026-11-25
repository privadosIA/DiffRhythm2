"""
Golden run for the DiT and the block-wise sampler (change add-dit-block-flow-matching): fp32, CPU, original DiffRhythm2 code
with the in-memory patches described in tools/dr2_ref.py. Run it inside the reference image (see tools/README.md).

Writes to --out (default tests/fixtures/dit):
  inputs.safetensors  tokens [L] int64, style [1,512], noise [blocks,1,10,64]
  cases.safetensors   stage-level cases:
      a_*  one DiT forward WITHOUT cache (noisy block at t=0.3, positions 0..9)
      b_*  forwards WITH a populated cache (text prefill, then one clean block at t=1, then a noisy block at t=0.5, positions 10..19),
           conditional and unconditional branch; text/clean key caches of layers 0 and 15
  run.safetensors     the whole sampler run: time_grid, block_start, block_states [blocks,steps,1,10,64], latent
  run_empty / run_cfg0 / run_45 .safetensors  the same layout for: empty lyrics (10 frames), guidance disabled (10 frames), 45 frames (5 blocks)
  meta.json           request, lengths, the reference's own nondeterminism (two identical runs), versions
  MANIFEST.md         sha256 of every file (verified by the core's FixtureDirectory)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch
from safetensors.torch import save_file

import dr2_ref as R

LYRICS = """[verse]
Hello world tonight
We sing along
[chorus]
Shine bright like the stars
"""
STYLE = "Pop, Piano, Female vocals, Happy, Bright"


def contiguous(d):
    return {k: v.detach().cpu().contiguous() for k, v in d.items()}


def capture(dit):
    """Forward hooks on layer 0 / layer 15 / final norm; returns (store, handles)."""
    store, handles = {}, []

    def hook(name, with_input=False):
        def fn(module, args, out):
            if with_input:
                store[name + "_in"] = args[0].detach().clone()
            o = out[0] if isinstance(out, (list, tuple)) else out
            store[name + "_out"] = o.detach().clone()
        return fn

    b0, b15 = dit.transformer_blocks[0], dit.transformer_blocks[len(dit.transformer_blocks) - 1]
    handles.append(b0.input_layernorm.register_forward_hook(hook("l0_ln", True)))
    handles.append(b0.mlp.register_forward_hook(hook("l0_mlp", True)))
    handles.append(b0.register_forward_hook(hook("l0")))
    handles.append(b15.register_forward_hook(hook("l15")))
    handles.append(dit.norm_out.register_forward_hook(hook("norm_out")))
    return store, handles


def case_a(dit, noise0, style, out):
    x = dit.latent_embed(noise0)
    time_t = torch.full((1, R.BLOCK), 0.3)
    pos = torch.arange(R.BLOCK)[None]
    mask = torch.ones(1, 1, R.BLOCK, R.BLOCK).bool()
    store, hs = capture(dit)
    pred, _, _ = dit(x=x, time=time_t, attn_mask=mask, position_ids=pos, style_prompt=style, use_cache=False)
    for h in hs:
        h.remove()
    out["a_noisy"] = noise0
    out["a_latent_emb"] = x
    out["a_time_emb"] = dit.time_embed(time_t)
    out["a_input_emb"] = dit.input_embed(x, style, dit.time_embed(time_t))
    out["a_pred"] = pred
    out.update({f"a_{k}": v for k, v in store.items()})


def cache_case(dit, tokens, style, clean_block, noisy, prefix, out, record_keys):
    """Replays sample_block_cache by hand: text prefill, one clean-block write, one noisy evaluation, for one branch."""
    from diffrhythm2.cache_utils import BlockFlowMatchingCache

    text = tokens if prefix == "b" else torch.zeros_like(tokens)
    st = style if prefix == "b" else torch.zeros_like(style)
    text_emb = dit.text_embed(text)
    L = text_emb.shape[1]
    cache = BlockFlowMatchingCache(text_lengths=torch.LongTensor([L]), num_history_block=None)
    text_time = torch.full((1, L), -1.0)
    with cache.cache_text():
        dit(x=text_emb, time=text_time, attn_mask=torch.ones(1, 1, L, L).bool(), position_ids=torch.arange(L)[None],
            style_prompt=st, use_cache=True, past_key_value=cache)
    # clean block 0 at time 1, positions 0..9, attending to text + itself
    ce = dit.latent_embed(clean_block)
    with cache.cache_context():
        dit(x=ce, time=torch.ones(1, R.BLOCK), attn_mask=torch.ones(1, 1, R.BLOCK, L + R.BLOCK).bool(),
            position_ids=torch.arange(R.BLOCK)[None], style_prompt=st, use_cache=True, past_key_value=cache)
    # noisy block 1 at t=0.5, positions 10..19, attending to text + clean block + itself
    x = dit.latent_embed(noisy)
    store, hs = capture(dit)
    pred, _, _ = dit(x=x, time=torch.full((1, R.BLOCK), 0.5), attn_mask=torch.ones(1, 1, R.BLOCK, L + 2 * R.BLOCK).bool(),
                     position_ids=torch.arange(R.BLOCK, 2 * R.BLOCK)[None], style_prompt=st, use_cache=True, past_key_value=cache)
    for h in hs:
        h.remove()
    out[f"{prefix}_pred"] = pred
    out.update({f"{prefix}_{k}": v for k, v in store.items()})
    if record_keys:
        for layer in (0, 15):
            out[f"{prefix}_text_k{layer}"] = cache.text_key_cache[layer]
            out[f"{prefix}_text_v{layer}"] = cache.text_value_cache[layer]
            out[f"{prefix}_clean_k{layer}"] = cache.key_cache[layer]
            out[f"{prefix}_clean_v{layer}"] = cache.value_cache[layer]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", default="/work/tests/fixtures/dit")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    t0 = time.time()
    model, cfg = R.load_model(args.ckpt, "fp32")
    dit = model.transformer
    tokens = torch.tensor(R.lyric_tokens(LYRICS), dtype=torch.long)
    style = R.style_embedding(args.ckpt, STYLE)
    nblocks = args.frames // R.BLOCK + (args.frames % R.BLOCK > 0)
    noise = R.make_noise(args.seed, nblocks + 1)  # one spare block
    print(f"loaded in {time.time() - t0:.0f}s; tokens {tokens.shape[0]}, style {tuple(style.shape)}, blocks {nblocks}", flush=True)

    # --- whole run, twice: the second one measures the reference's own nondeterminism
    results = []
    for rep in range(2):
        sample, rec = R.patched_sampler(model, noise, "fp32")
        lat = sample(text=tokens[None], duration=args.frames, style_prompt=style, steps=args.steps, cfg_strength=args.cfg, process_bar=False)
        results.append((lat.clone(), rec.stacked()))
        print(f"run {rep}: latent {tuple(lat.shape)}, blocks sampled {len(rec.start)}", flush=True)
    latent, run = results[0]
    drift = {
        "latent_max_abs": (results[0][0] - results[1][0]).abs().max().item() if results[0][0].shape == results[1][0].shape else None,
        "states_max_abs": (results[0][1]["block_states"] - results[1][1]["block_states"]).abs().max().item(),
    }
    sampled = [run["block_states"][i, -1] for i in range(run["block_states"].shape[0])]  # last state of each block [1,10,64]
    ep, nb = R.end_pos(sampled)
    assert ep == latent.shape[1], f"EOS helper disagrees with the reference run ({ep} vs {latent.shape[1]})"

    # --- extra runs for the spec scenarios: empty lyrics, guidance off, length that is not a multiple of the block
    extra = {}
    for name, toks, frames, cfg_s in (("run_empty", tokens[:0], 10, args.cfg), ("run_cfg0", tokens, 10, 0.0), ("run_45", tokens, 45, args.cfg)):
        sample, rec = R.patched_sampler(model, noise, "fp32")
        lat = sample(text=toks[None], duration=frames, style_prompt=style, steps=args.steps, cfg_strength=cfg_s, process_bar=False)
        extra[name] = {**rec.stacked(), "latent": lat}
        print(f"{name}: latent {tuple(lat.shape)}, blocks sampled {len(rec.start)}", flush=True)

    # --- stage-level cases
    cases = {}
    case_a(dit, noise[0], style, cases)
    clean0 = sampled[0]
    cases["b_clean_block"] = clean0
    cases["b_noisy"] = noise[1]
    cache_case(dit, tokens[None], style, clean0, noise[1], "b", cases, record_keys=True)
    cache_case(dit, tokens[None], style, clean0, noise[1], "n", cases, record_keys=False)  # unconditional: zero text tokens, zero style

    save_file(contiguous({"tokens": tokens, "style": style, "noise": noise}), os.path.join(args.out, "inputs.safetensors"))
    save_file(contiguous(cases), os.path.join(args.out, "cases.safetensors"))
    save_file(contiguous({**run, "latent": latent}), os.path.join(args.out, "run.safetensors"))
    for name, d in extra.items():
        save_file(contiguous(d), os.path.join(args.out, name + ".safetensors"))
    import transformers, torchdiffeq
    meta = {
        "lyrics": LYRICS, "style": STYLE, "frames": args.frames, "steps": args.steps, "cfg_strength": args.cfg, "seed": args.seed,
        "text_len": int(tokens.shape[0]), "blocks_sampled": nb, "latent_frames": int(latent.shape[1]),
        "extra_runs": {k: {"latent_frames": int(v["latent"].shape[1]), "blocks_sampled": int(v["block_states"].shape[0])} for k, v in extra.items()},
        "reference_nondeterminism": drift,
        "case_a": {"t": 0.3, "positions": "0..9"}, "case_b": {"t": 0.5, "positions": "10..19", "clean_t": 1.0},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "torchdiffeq": torchdiffeq.__version__},
        "config": cfg,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    R.write_manifest(
        args.out, "DiT golden run (fp32, CPU)",
        "Dumped by `tools/dump_dit_reference.py` from the reference image `veritmo/diffrhythm2:1` (upstream 7804f82), "
        "with the in-memory patches listed in `tools/dr2_ref.py`. See `meta.json` for the request and the reference's own run-to-run drift.",
        ["inputs.safetensors", "cases.safetensors", "run.safetensors", "run_empty.safetensors", "run_cfg0.safetensors", "run_45.safetensors", "meta.json"])
    print(json.dumps({k: meta[k] for k in ("text_len", "blocks_sampled", "latent_frames", "reference_nondeterminism")}, indent=1))


if __name__ == "__main__":
    main()
