"""
Golden run for the BigVGAN decoder (change add-bigvgan-decoder): fp32, CPU, deterministic algorithms, the original `Generator` code.
Run it inside the reference image (see tools/README.md). Writes to --out (default tests/fixtures/bigvgan):

  chunk_real.safetensors    in [1,64,20] (frames 0..19 of the golden-run latent, transposed), stage signals, rms per channel, wave [1,1,192000]
  chunk_random.safetensors  seeded randn [1,64,20]: the same, so a divergence cannot hide in a latent that happens to be smooth
  chunk_loud.safetensors    randn * 10: the unclipped output exceeds 1, so the clamp (not a tanh) is exercised; wave only
  decode_40.safetensors     the whole golden-run latent [1,64,40] and its decode_audio waveform [1,1,384000] (two seams plus an aligned final chunk)
  activation.safetensors    the anti-aliased activation in isolation (activation_post and resblocks.17.activations.5) on fixed signals, plus
                            the bare SnakeBeta of activation_post
  weights.safetensors       reference-folded weights of ups.3.0, ups.5.0, resblocks.17.convs1.2 and conv_post (to check the C# weight-norm fold)
  meta.json                 shapes, the reference's own run-to-run drift (two decodes), versions
  MANIFEST.md               sha256 of every file (verified by the core's FixtureDirectory)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code

import torch
from safetensors.torch import load_file, save_file

import dr2_bigvgan as B
import dr2_ref as R


def contiguous(d):
    return {k: v.detach().cpu().contiguous() for k, v in d.items()}


def chunk_case(gen, x, with_stages=True):
    bv = gen.decoder
    wave, stages = B.staged_forward(bv, x)
    assert torch.equal(wave, bv(x)), "the staged copy of BigVGAN.forward diverges from the model"
    out = {"input": x, "wave": wave}
    if with_stages:
        out.update(B.stage_entries("s", stages))
    return out


def activation_case(module, channels, seed):
    g = torch.Generator().manual_seed(seed)
    n = 64
    x = torch.randn(1, channels, n, generator=g)
    x[:, :, :4] = torch.tensor([3.0, -2.0, 1.5, 0.5])       # large, non-zero samples at both edges: the replicate padding is what is tested
    x[:, :, -4:] = torch.tensor([-1.0, 2.5, -3.0, 2.0])
    up = module.upsample(x)
    act = module.act(up)
    return x, up, act, module.downsample(act)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--latent", default="/work/tests/fixtures/latent/latent_40.safetensors")
    ap.add_argument("--out", default="/work/tests/fixtures/bigvgan")
    ap.add_argument("--seed", type=int, default=4321)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)

    gen = B.load_generator(args.ckpt, "fp32")
    bv = gen.decoder
    latent = load_file(args.latent)["latent"]                       # [1, 40, 64]
    full = latent.transpose(1, 2).contiguous()                      # [1, 64, 40]
    g = torch.Generator().manual_seed(args.seed)

    real = chunk_case(gen, full[:, :, :B.CHUNK].contiguous())
    rand = chunk_case(gen, torch.randn(1, 64, B.CHUNK, generator=g))
    loud = chunk_case(gen, torch.randn(1, 64, B.CHUNK, generator=g) * 10.0, with_stages=False)
    unclipped_peak = None
    # peak of the unclipped output of the loud chunk: proves the case really drives it beyond 1
    _, st = B.staged_forward(bv, loud["input"])
    unclipped_peak = st["conv_post"].abs().max().item()
    assert unclipped_peak > 1.0, f"the loud chunk does not exceed 1 before the clamp ({unclipped_peak})"
    loud_only = {"input": loud["input"], "wave": loud["wave"]}

    dec_a = gen.decode_audio(full, overlap=5, chunk_size=B.CHUNK)
    dec_b = gen.decode_audio(full, overlap=5, chunk_size=B.CHUNK)
    drift = (dec_a - dec_b).abs().max().item()
    assert dec_a.shape == (1, 1, 40 * B.SAMPLES_PER_FRAME), tuple(dec_a.shape)

    act = {}
    for name, mod, ch in (("post", bv.activation_post, 24), ("res17_5", bv.resblocks[17].activations[5], 24)):
        x, up, a, down = activation_case(mod, ch, args.seed + ch)
        act.update({f"{name}_in": x, f"{name}_up": up, f"{name}_snake": a, f"{name}_out": down})
        act[f"{name}_up_filter"] = mod.upsample.filter
        act[f"{name}_down_filter"] = mod.downsample.lowpass.filter
        act[f"{name}_alpha"] = mod.act.alpha
        act[f"{name}_beta"] = mod.act.beta
    sg = torch.Generator().manual_seed(args.seed + 1)
    xs = torch.randn(1, 24, 200, generator=sg) * 2.0
    act["snake_in"] = xs
    act["snake_out"] = bv.activation_post.act(xs)

    weights = {
        "ups.3.0.weight": bv.ups[3][0].weight, "ups.3.0.bias": bv.ups[3][0].bias,
        "ups.5.0.weight": bv.ups[5][0].weight, "ups.5.0.bias": bv.ups[5][0].bias,
        "resblocks.17.convs1.2.weight": bv.resblocks[17].convs1[2].weight, "resblocks.17.convs1.2.bias": bv.resblocks[17].convs1[2].bias,
        "conv_post.weight": bv.conv_post.weight,
    }
    assert bv.conv_post.bias is None

    save_file(contiguous(real), os.path.join(args.out, "chunk_real.safetensors"))
    save_file(contiguous(rand), os.path.join(args.out, "chunk_random.safetensors"))
    save_file(contiguous(loud_only), os.path.join(args.out, "chunk_loud.safetensors"))
    save_file(contiguous({"input": full, "wave": dec_a}), os.path.join(args.out, "decode_40.safetensors"))
    save_file(contiguous(act), os.path.join(args.out, "activation.safetensors"))
    save_file(contiguous(weights), os.path.join(args.out, "weights.safetensors"))
    import transformers  # noqa: F401  (version only; the image ships it)
    meta = {
        "seed": args.seed, "chunk_frames": B.CHUNK, "samples_per_frame": B.SAMPLES_PER_FRAME, "keep_channels": B.KEEP_CHANNELS,
        "latent_frames": int(full.shape[2]),
        "reference_nondeterminism": {"decode_40_max_abs": drift},
        "loud_unclipped_peak": unclipped_peak,
        "wave_peak": {"real": real["wave"].abs().max().item(), "random": rand["wave"].abs().max().item(), "decode_40": dec_a.abs().max().item()},
        "stages": ["conv_pre"] + [f"{p}{i}" for i in range(6) for p in ("up", "stage")] + ["act_post", "conv_post"],
        "raw_stages": sorted(B.RAW_STAGES),
        "versions": {"torch": torch.__version__},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    R.write_manifest(
        args.out, "BigVGAN golden run (fp32, CPU)",
        "Dumped by `tools/dump_bigvgan_reference.py` from the reference image `veritmo/diffrhythm2:1` (upstream 7804f82). Stage signals keep "
        f"only the first {B.KEEP_CHANNELS} channels (`conv_pre` keeps all) and only for the stages in `RAW_STAGES` of `tools/dr2_bigvgan.py`; `*_rms` is the RMS over time of every channel, for every stage. See `meta.json` for the reference's own run-to-run drift.",
        ["chunk_real.safetensors", "chunk_random.safetensors", "chunk_loud.safetensors", "decode_40.safetensors", "activation.safetensors",
         "weights.safetensors", "meta.json"])
    print(json.dumps({k: meta[k] for k in ("reference_nondeterminism", "loud_unclipped_peak", "wave_peak")}, indent=1))


if __name__ == "__main__":
    main()
