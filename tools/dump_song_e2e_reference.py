"""
The reference song (change add-song-pipeline): the steps of /opt/server.py `Engine._sing`, on CPU in float32, with a real seed and WITHOUT injecting noise,
so that the C# pipeline can be compared with a whole song made from the same request. Run it inside the reference image (see tools/README.md).

In memory only, as in the other dumpers, `CFM.sample_block_cache` is patched: the two `.half()` become `.float()` (CPU float32 instead of the service's CUDA half)
and the noise draw is wrapped by a recorder, so the fixture holds the noise the reference REALLY drew after `torch.manual_seed(seed)`.

Writes to --out (default tests/fixtures/song_e2e):
  song.safetensors  tokens [N] int64, style [1,512], noise [blocks,1,10,64], latent [1,T,64], latent64 [1,T,64] (a float64 run of the same network with that
                    noise injected: the yardstick for how strongly the sampler amplifies rounding), wave [1,1,T*9600] (decode_audio), stereo [2,T*9600] (make_fake_stereo)
  song.wav          the WAV the service code writes (pedalboard, 16 bit, 48 kHz, 2 channels)
  meta.json, MANIFEST.md
"""
import argparse
import gc
import inspect
import json
import os
import sys
import textwrap
import types
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code
os.environ.setdefault("HOME", "/tmp")
sys.path.insert(0, "/opt")

import numpy as np  # noqa: E402
import pedalboard  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dr2_bigvgan as B  # noqa: E402
import dr2_ref as R  # noqa: E402

LYRICS = "[verse]\nHello world tonight\nWe sing along\n[chorus]\nShine bright like the stars\n"
STYLE = "Pop, Piano, Female vocals, Happy, Bright"
SEED, MAX_SECS, STEPS, CFG = 1234, 4.0, 16, 2.0


def sampler(model, precision, injected=None, recorded=None):
    from diffrhythm2 import cfm

    src = textwrap.dedent(inspect.getsource(cfm.CFM.sample_block_cache))
    assert src.count(".half()") == 2, "upstream changed: expected 2 .half() in sample_block_cache"
    src = src.replace(".half()", ".double()" if precision == "fp64" else ".float()")
    rand = "noisy_emb = torch.randn(batch, self.block_size, self.num_channels, device=device, dtype=style_prompt.dtype)"
    assert src.count(rand) == 1, "upstream changed: the noise line of sample_block_cache is not the expected one"
    if injected is None:
        src = src.replace(rand, "noisy_emb = RECORD(" + rand.split(" = ", 1)[1] + ")")
    else:
        src = src.replace(rand, "noisy_emb = INJECT[bid].to(device=device, dtype=style_prompt.dtype)")

    def record(t):
        recorded.append(t.detach().clone())
        return t

    ns = dict(cfm.__dict__)
    ns.update(RECORD=record, INJECT=injected)
    exec(compile(src, "<patched sample_block_cache>", "exec"), ns)
    return types.MethodType(ns["sample_block_cache"], model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", default="/work/tests/fixtures/song_e2e")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    import server
    import inference as dr2

    lyrics = server.normalize_lyrics(LYRICS)
    tokens = torch.tensor(R.lyric_tokens(lyrics), dtype=torch.long)
    style = R.style_embedding(args.ckpt, STYLE)                       # [1, 512] float32, the service's mulan(texts=[style]) (fp32)
    duration = int(MAX_SECS * 5)

    model, _ = R.load_model(args.ckpt, "fp32")
    drawn = []
    torch.manual_seed(SEED)                                            # after the tokens and the style, right before sampling, as the service does
    latent = sampler(model, "fp32", recorded=drawn)(text=tokens[None], duration=duration, style_prompt=style, steps=STEPS, cfg_strength=CFG, process_bar=False)
    del model
    gc.collect()
    assert latent.shape[1] >= 20, f"the song ended early ({latent.shape[1]} frames): the reference decoder cannot decode fewer than 20"
    noise = torch.stack(drawn)

    gen = B.load_generator(args.ckpt, "fp32")
    wave_mono = gen.decode_audio(latent.transpose(1, 2), overlap=5, chunk_size=20)
    audio = wave_mono.float().cpu().numpy().squeeze()[None, :]
    stereo = np.ascontiguousarray(dr2.make_fake_stereo(audio, gen.h.sampling_rate))
    wav_path = os.path.join(args.out, "song.wav")
    with pedalboard.io.AudioFile(wav_path, "w", gen.h.sampling_rate, 2) as f:
        f.write(stereo)
    del gen
    gc.collect()

    model64, _ = R.load_model(args.ckpt, "fp64")
    latent64 = sampler(model64, "fp64", injected=noise)(text=tokens[None], duration=duration, style_prompt=style.double(), steps=STEPS, cfg_strength=CFG, process_bar=False)

    w = wave.open(wav_path)
    meta = {
        "lyrics": LYRICS, "prepared_lyrics": lyrics, "style": STYLE, "seed": SEED, "max_secs": MAX_SECS, "steps": STEPS, "cfg_strength": CFG,
        "frames": int(latent.shape[1]), "blocks_drawn": len(drawn), "wav": {"channels": w.getnchannels(), "rate": w.getframerate(), "width": w.getsampwidth(), "frames": w.getnframes()},
        "latent_vs_fp64_max_abs_per_block": [float((latent[:, i * 10:(i + 1) * 10].double() - latent64[:, i * 10:(i + 1) * 10]).abs().max()) for i in range(latent.shape[1] // 10)],
        "versions": {"torch": torch.__version__, "pedalboard": pedalboard.__version__},
    }
    save_file({"tokens": tokens, "style": style, "noise": noise, "latent": latent, "latent64": latent64,
               "wave": wave_mono.float().cpu(), "stereo": torch.from_numpy(stereo)}, os.path.join(args.out, "song.safetensors"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    R.write_manifest(args.out, "Reference song (CPU, float32, seed 1234)",
                     "The steps of the service's `_sing` on CPU in float32 with a real seed and the noise it drew, from `tools/dump_song_e2e_reference.py`.",
                     ["song.safetensors", "song.wav", "meta.json"])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
