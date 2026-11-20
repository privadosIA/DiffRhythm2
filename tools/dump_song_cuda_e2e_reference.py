"""
INFORMATIONAL reference song on CUDA (change add-song-pipeline, task 7.1): the service's `_sing` exactly as it runs in production (DiT and decoder `.half()`, MuLan
float32, the unpatched `sample_block_cache`, seed 1234 for both generators), 4 s, on the GPU. Run inside the reference image with a GPU (docker run --gpus device=N).
Writes song_cuda.safetensors (latent [1,20,64], wave [1,1,192000], stereo [2,192000], all as float32 on CPU) and song_cuda_meta.json to --out (default tests/fixtures/song_cuda).
No gating test may use it: half precision on CUDA is only measured.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HOME", "/tmp")
sys.path.insert(0, "/opt")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dr2_bigvgan as B  # noqa: E402
import dr2_ref as R  # noqa: E402

LYRICS = "[verse]\nHello world tonight\nWe sing along\n[chorus]\nShine bright like the stars\n"
STYLE = "Pop, Piano, Female vocals, Happy, Bright"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--out", default="/work/tests/fixtures/song_cuda")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    os.makedirs(args.out, exist_ok=True)
    import server
    import inference as dr2

    device = torch.device("cuda")
    lyrics = server.normalize_lyrics(LYRICS)
    tokens = torch.tensor(R.lyric_tokens(lyrics), dtype=torch.long, device=device)
    model, _ = R.load_model(args.ckpt, "fp16")
    model = model.to(device)
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=args.ckpt).to(device)
    decoder = B.load_generator(args.ckpt, "fp16").to(device)
    with torch.no_grad():
        style = mulan(texts=[STYLE]).to(device).squeeze(0).half()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    with torch.inference_mode():
        latent = model.sample_block_cache(text=tokens.unsqueeze(0), duration=int(4.0 * 5), style_prompt=style.unsqueeze(0), steps=16, cfg_strength=2.0, process_bar=False)
        audio = decoder.decode_audio(latent.transpose(1, 2), overlap=5, chunk_size=20)
        mono = audio.float().cpu().numpy().squeeze()[None, :]
    stereo = np.ascontiguousarray(dr2.make_fake_stereo(mono, 48000))
    save_file({"latent": latent.float().cpu().contiguous(), "wave": audio.float().cpu().contiguous(), "stereo": torch.from_numpy(stereo)}, os.path.join(args.out, "song_cuda.safetensors"))
    meta = {"gpu": torch.cuda.get_device_name(0), "frames": int(latent.shape[1]), "seed": 1234, "precision": "model and decoder half, mulan float32 (the service)", "lyrics": LYRICS, "style": STYLE}
    with open(os.path.join(args.out, "song_cuda_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
