"""
Small fixtures for the song pipeline (change add-song-pipeline), from the reference image, no checkpoint needed. Writes to --out (default tests/fixtures/song):

  noise.safetensors     the noise the reference draws on CPU with seed 1234: noise_f32 / noise_f16 [4,1,10,64] (f16 drawn WITH that dtype, stored as float32),
                        noise_f16_via_cast [4,1,10,64] (the float32 draw cast to half: it is NOT the same), seed in meta.json
  validate.json         bodies given to server.Engine.validate and what it returns or the exact message it raises; `typed` says whether the
                        C# request (typed fields) can represent the body
  audio.safetensors     stereo_in [1,20000] / stereo_out [2,20000] (inference.make_fake_stereo); pcm_in [300000] float32 and pcm_out [300000] int16
                        (the 16-bit integers pedalboard writes for pcm_in, read back from the WAV)
  meta.json, MANIFEST.md
"""
import argparse
import json
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code
os.environ.setdefault("HOME", "/tmp")
sys.path.insert(0, "/opt/dr2")
sys.path.insert(0, "/opt")

import numpy as np  # noqa: E402
import pedalboard  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dr2_ref as R  # noqa: E402

SEED = 1234


def noise_fixture():
    blocks = 4
    out = {}
    torch.manual_seed(SEED)
    out["noise_f32"] = torch.stack([torch.randn(1, 10, 64) for _ in range(blocks)])
    torch.manual_seed(SEED)
    out["noise_f16"] = torch.stack([torch.randn(1, 10, 64, dtype=torch.float16) for _ in range(blocks)]).float()
    torch.manual_seed(SEED)
    out["noise_f16_via_cast"] = torch.stack([torch.randn(1, 10, 64).half() for _ in range(blocks)]).float()
    assert (out["noise_f16"] - out["noise_f16_via_cast"]).abs().max() > 1.0, "the half draw should differ from the cast float draw"
    return out


def validation_cases():
    c = chr
    lyrics_ok = "[verse]\nHello world\n[chorus]\nLa la la"
    base = {"lyrics": lyrics_ok, "style": "Pop, Piano"}

    def case(name, body, typed=True):
        return {"name": name, "body": body, "typed": typed}

    cases = [
        case("minimal", base),
        case("defaults are applied", {"lyrics": "hello there", "style": "Rock"}),
        case("style as a list", {**base, "style": ["Pop", "Piano", " Female vocals "]}),
        case("style spacing", {**base, "style": "  Pop ,Piano,  Female vocals ,"}),
        case("style non-ascii spaces", {**base, "style": "Pop" + c(0x3000) + "," + c(0xA0) + "Piano" + c(0x2003) + ",Rock"}),
        case("style only commas", {**base, "style": ", ,"}),
        case("style empty", {**base, "style": ""}),
        case("style missing", {"lyrics": lyrics_ok}),
        case("style 500", {**base, "style": "a" * 500}),
        case("style 501", {**base, "style": "a" * 501}),
        case("style 500 after normalization", {**base, "style": "a" * 495 + " , " + "b"}),
        case("style not a string", {**base, "style": 12}, typed=False),
        case("lyrics missing", {"style": "Pop"}),
        case("lyrics empty", {"lyrics": "", "style": "Pop"}),
        case("lyrics blank", {"lyrics": "  \n ", "style": "Pop"}),
        case("lyrics not a string", {"lyrics": 5, "style": "Pop"}, typed=False),
        case("lyrics tags only", {"lyrics": "[verse]\n[chorus]", "style": "Pop"}),
        case("lyrics unknown tags", {"lyrics": "[Pre-Chorus]\nla\n[Verse]\nlo", "style": "Pop"}),
        case("lyrics tag case", {"lyrics": "[VERSE]\nla\n[ChOrUs ]\nlo\n[start]", "style": "Pop"}),
        case("lyrics crlf", {"lyrics": "[verse]\r\nHello\r\nworld\r\n", "style": "Pop"}),
        case("lyrics 20000", {"lyrics": "a" * 20000, "style": "Pop"}),
        case("lyrics 20001", {"lyrics": "a" * 20001, "style": "Pop"}),
        case("lyrics 20000 characters outside the BMP", {"lyrics": c(0x1F600) * 20000, "style": "Pop"}),
        case("body not an object", "x", typed=False),
    ]
    for key, values in (
        ("max_secs", [9.99, 10, 10.0, 12.5, 60, 210, 210.5, 0, -5]),
        ("cfg_strength", [0.49, 0.5, 2.5, 10, 10.1, 0]),
        ("steps", [3, 4, 16, 64, 65, 0]),
        ("seed", [-1, 0, 1234, 2**63 - 1, 2**63]),
    ):
        for v in values:
            cases.append(case(f"{key} = {v!r}", {**base, key: v}))
    cases += [
        case("steps 16.9 is truncated", {**base, "steps": 16.9}, typed=False),
        case("steps as a string", {**base, "steps": "16"}, typed=False),
        case("max_secs boolean", {**base, "max_secs": True}, typed=False),
        case("seed float", {**base, "seed": 1.5}, typed=False),
        case("seed boolean", {**base, "seed": True}, typed=False),
        case("everything given", {**base, "max_secs": 30, "cfg_strength": 3.5, "steps": 32, "seed": 42}),
    ]
    return cases


def run_validation(cases):
    import server

    results = []
    for c in cases:
        try:
            out = server.Engine.validate(c["body"])
            results.append({**c, "ok": out})
        except ValueError as e:
            results.append({**c, "error": str(e)})
    assert any("ok" in r for r in results) and any("error" in r for r in results)
    return results


def audio_fixture():
    import inference as dr2

    rng = np.random.default_rng(5)
    stereo_in = (rng.standard_normal((1, 20000)) * 0.3).astype("float32")
    stereo_out = dr2.make_fake_stereo(stereo_in, 48000)

    rng = np.random.default_rng(9)
    boundary = [0, 1, -1, 0.5, -0.5, 1e-9, -1e-9, 1.5, -1.5, 0.99999994, -0.99999994, 2, -2]
    grid = [(k + f) / 32768.0 for k in range(-40, 40) for f in (-0.5, -0.4999, 0.0, 0.4999, 0.5, 0.5001)]
    signal = np.concatenate([
        rng.standard_normal(150_000) * 0.2, rng.uniform(-1.2, 1.2, 100_000), rng.uniform(-1, 1, 49_000) * 1e-3, np.array(boundary + grid),
    ]).astype("float32")
    signal = np.concatenate([signal, np.zeros(300_000 - len(signal), dtype="float32")]) if len(signal) < 300_000 else signal[:300_000]
    with pedalboard.io.AudioFile("/tmp/pcm.wav", "w", 48000, 1) as f:
        f.write(np.ascontiguousarray(signal[None, :]))
    w = wave.open("/tmp/pcm.wav")
    assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 48000)
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    assert len(pcm) == len(signal)
    return {
        "stereo_in": torch.from_numpy(stereo_in), "stereo_out": torch.from_numpy(np.ascontiguousarray(stereo_out)),
        "pcm_in": torch.from_numpy(signal), "pcm_out": torch.from_numpy(pcm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/tests/fixtures/song")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    noise = noise_fixture()
    cases = run_validation(validation_cases())
    audio = audio_fixture()
    save_file({k: v.contiguous() for k, v in noise.items()}, os.path.join(args.out, "noise.safetensors"))
    save_file({k: v.contiguous() for k, v in audio.items()}, os.path.join(args.out, "audio.safetensors"))
    with open(os.path.join(args.out, "validate.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=True, indent=0)
    meta = {"seed": SEED, "validation_cases": len(cases), "validation_errors": sum("error" in c for c in cases),
            "float16_vs_cast_max_diff": float((noise["noise_f16"] - noise["noise_f16_via_cast"]).abs().max()),
            "versions": {"torch": torch.__version__, "numpy": np.__version__, "pedalboard": pedalboard.__version__}}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    R.write_manifest(args.out, "Song pipeline small fixtures (reference image, CPU)",
                     "Dumped by `tools/dump_song_reference.py`: the reference's CPU noise for seed 1234, what `server.Engine.validate` returns for a list of bodies, "
                     "`make_fake_stereo`, and the 16-bit integers pedalboard writes for a long signal.",
                     ["noise.safetensors", "validate.json", "audio.safetensors", "meta.json"])
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
