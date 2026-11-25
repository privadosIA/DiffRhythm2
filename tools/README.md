# tools/

Python dumpers for the golden-run fixtures. They run **inside the reference image** (`veritmo/diffrhythm2:1`, upstream commit 7804f82) and are the only contact with Python in this project. Fixtures are written under `tests/fixtures/<name>/` with a `MANIFEST.md` of sha256 hashes that the tests verify (`FixtureDirectory` from `musicgen-core`).

## Running a dumper

```bash
CKPT=~/Proyectos/veritmo/models/music/diffrhythm2     # config.json, model.safetensors, MuQ-MuLan cache, ...
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
  -v "$PWD":/work -v "$CKPT":/ckpt:ro \
  --entrypoint python3 veritmo/diffrhythm2:1 -I /work/tools/<script>.py [--out DIR]
```

`-I` keeps the interpreter from importing anything from the working directory; each script adds `tools/` (our own code) to `sys.path` explicitly. Run from the repository root.

| script | writes | notes |
|---|---|---|
| `dump_dit_reference.py` | `tests/fixtures/dit/` | fp32 CPU: inputs (tokens, style, per-block noise), stage-level cases, whole sampler run. ~45 s |
| `dump_eos_reference.py` | `tests/fixtures/eos/` | scripted blocks through the real sampler: lengths of the end-of-song rule |
| `dump_torch_archive_fixtures.py` | `tests/fixtures/torch_archive/` | tiny archives written by the real `torch.save` (several dtypes, wrapper key, shared storage, non-contiguous view, a hostile `os.system` pickle, damaged and legacy files) and the values `torch.load` returns, for the C# reader's tests. No checkpoint. ~5 s |
| `convert_decoder.py` | `$CKPT/decoder.safetensors` | **oracle only since add-weights-tools** (`diffrhythm2 convert` does this in C#); one-off: `decoder.bin` (pickle) to safetensors. **Needs a writable checkpoint mount** (drop the `:ro`) or `--out` pointing at a mounted dir. ~15 s |
| `dump_bigvgan_reference.py` | `tests/fixtures/bigvgan/` | fp32 CPU: stage signals of a real and a random 20-frame chunk, a loud (clipped) chunk, the 40-frame `decode_audio` waveform, the anti-aliased activation in isolation, reference-folded weights. ~1 min |
| `dump_stitch_reference.py` | `tests/fixtures/stitch/` | the real `decode_audio` with a stand-in decoder that tags each chunk: which chunk supplies each frame, for T in {20, 25, 35, 40, 47, 64}. No weights |
| `dump_bigvgan_fp16_reference.py` | `tests/fixtures/bigvgan_fp16/` | production-style half precision on CPU for the 40-frame latent. **Informational only, never a gate** |
| `dump_g2p_reference.py` | `tests/fixtures/g2p/` | the English g2p and lyrics tokenization, layer by layer (normalized text, cleaned text, espeak transcripts, phonemes, ids), the `inflect` number table, whole-lyrics cases. Corpus in `tools/g2p_corpus.txt` (a literal `\uXXXX` is replaced by that code point). Needs no checkpoint. ~20 s |
| `fetch_espeak_reference.sh` | any directory outside the repo | copies the reference's exact espeak-ng 1.51 (library, two audio libraries, data) out of the image so the C# tests can run against it. GPL-3: do not commit |
| `convert_mulan_text.py` | `$CKPT/mulan_text.safetensors`, `$CKPT/mulan_text_tokenizer.json` | **oracle only since add-weights-tools** (`diffrhythm2 convert` does this in C#); one-off: the 273 text-tower tensors of MuQ-MuLan (from the 2.5 GB `pytorch_model.bin` pickle, audio tower dropped) plus the XLM-R `tokenizer.json`. Needs a writable checkpoint mount (drop the `:ro`) or `--out`. Asserts the 199 encoder tensors equal `xlm-roberta-base`'s. ~1 min, ~3 GB of RAM |
| `dump_mulan_text_reference.py` | `tests/fixtures/mulan_text/` | the tokenizer corpus (`tools/mulan_text_corpus.txt` + special texts), the normalization of every code point of planes 0 to 2 and of multi-character strings, and the stage outputs of the real `MuQMuLan` for seven prompts. fp32 CPU, ~20 s |
| `convert_mulan_audio.py` | `$CKPT/mulan_audio.safetensors` | **oracle only** (`diffrhythm2 convert --audio` does this in C#); one-off: the 531 audio-tower tensors of MuQ-MuLan (525 of MuQ-large, 513 float32 + 18 int64 batch-norm counters overall) from the `pytorch_model.bin` pickle. Needs a writable checkpoint mount. Reports that only 69 of the 525 MuQ-large tensors equal `MuQ-large-msd-iter`'s: MuLan fine-tuned it. ~5 s, ~3 GB of RAM |
| `dump_mulan_audio_reference.py` | `tests/fixtures/mulan_audio/` | the audio tower through the real `MuQMuLan`: deterministic 24 kHz signals, the reference's clips, the mel filter bank and spectrograms, the stage outputs of the first clip of a 25 s signal (subsampling, Conformer layers 1/5/6/11/12, final state, projections), clip and signal embeddings, text embeddings one by one and as a padded batch, similarities. Records that `pos_conv_embed` is never called. `--verify DIR` recomputes in a fresh process and records the difference (0.0 measured). fp32 CPU, ~12 s |
| `dump_song_reference.py` | `tests/fixtures/song/` | the reference's CPU noise for seed 1234 (float32, float16 drawn with that dtype, and the float32 draw cast to half, which is NOT the same), what `server.Engine.validate` returns for 56 bodies, `make_fake_stereo`, and the 16-bit integers pedalboard writes for a 300,000-sample signal. No checkpoint. ~10 s |
| `dump_song_e2e_reference.py` | `tests/fixtures/song_e2e/` | the reference song: the service's `_sing` steps on CPU in float32 with seed 1234 and 4 s (2 blocks), WITHOUT injecting noise (the noise it drew is recorded), the latent, a float64 run for the sampler's sensitivity, the mono waveform, the stereo waveform and the WAV pedalboard writes. ~1 min |
| `dump_song_cuda_reference.py` | `tests/fixtures/song_cuda/noise_cuda.safetensors` | the reference's CUDA noise for seed 1234 (float16 and float32). Needs a GPU: `docker run --gpus device=N ...`. Run once per card: the stream is identical on an RTX 4070 and an RTX 5060 Ti |
| `dump_song_cuda_e2e_reference.py` | `tests/fixtures/song_cuda/song_cuda.safetensors` | the service's own song on CUDA in half precision (4 s, seed 1234), INFORMATIONAL, never a gate. Needs a GPU and the checkpoint |
| `dump_dit_fp16_reference.py` | `tests/fixtures/dit_fp16/` | production-style half precision on CPU, same inputs. **Informational only, never a gate** |

`dr2_ref.py` is the shared module. It patches `CFM.sample_block_cache` in memory (see its docstring): `.half()` -> `.float()` for the fp32 reference, the random noise replaced by the injected noise, and `odeint` wrapped to record every integration state. Every replacement is asserted, so an upstream change fails loudly.

## What the golden run showed

- The reference is **bit-for-bit deterministic** between two identical fp32 CPU runs (`reference_nondeterminism` in `tests/fixtures/dit/meta.json` is 0.0), so any tolerance the C# port needs comes from the port, not from noise in the reference.
- fp16 (CPU) drifts from fp32 by up to ~0.59 (mean ~0.039) in the latent after 4 blocks, which is why fp32 is the gate and fp16 is only reported.
- The golden run produces 40 frames (4 blocks) without early end-of-song.

## Converting the decoder checkpoint

> Since change `add-weights-tools` the C# command `diffrhythm2 convert` (or `install`) does this with no Python and writes the same bytes; `convert_decoder.py` is kept as the oracle `RealWeightsTests` compares against.

`decoder.bin` is `torch.save({"generator": state_dict})`, a Python pickle that TorchSharp cannot read. The C# code reads `decoder.safetensors`, produced once by:

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
  -v "$PWD":/work -v "$CKPT":/ckpt \
  --entrypoint python3 veritmo/diffrhythm2:1 -I /work/tools/convert_decoder.py     # writes /ckpt/decoder.safetensors
```

The values are copied bit for bit and nothing is folded. The only change is a pure rename of each weight-norm pair, `X.weight_g` / `X.weight_v` to `X.parametrizations.weight.original0` / `original1` (the spelling the core's `StrictLoader` folds); 783 tensors, 116 pairs, 523 MB. The output is deterministic: converting into an empty directory twice gives the same sha256 (`6c39bc55be24...`). `decoder.json` is read as it is.

## What the BigVGAN golden run showed

- **Within one process the reference is bit-for-bit deterministic** (`reference_nondeterminism` in `tests/fixtures/bigvgan/meta.json` is 0.0, two decodes of the 40-frame latent). **Across processes it is not**: regenerating the fixtures gives the same hashes for the activation, weights, stitching and fp16 files but different ones for the chunk and `decode_40` files, by up to 6.3e-7 abs on the 40-frame waveform (1.4e-6 relative), 2.6e-5 abs on the largest stage signal (stage 5, values up to 6.7) and 1.6e-5 on the clipped loud waveform. Thread scheduling changes the summation order of the convolutions. The committed files are one such run; the C# tolerances (`DecoderTolerances.cs`) sit at that variance.
- Stage 5 of the real chunk has per-channel RMS up to ~1000 in one channel (the SnakeBeta output then brings it back to a ~0.46 peak waveform); the RMS check covers every channel for that reason.
- fp16 (CPU) deviates from fp32 by up to 2.8e-3 on the waveform (mean 2.9e-4).

## What the g2p golden run showed

- The reference's espeak is **1.51** (Debian `1.51+dfsg-10+deb12u2`), phonemizer 3.4.0. `transcripts.json` records every raw espeak answer, so the phonemizer post-processing can be replayed in C# without the library.
- `inflect` is **not a dependency of the image**: the module imported is the copy vendored in setuptools 79.0.1. The number table is generated from that copy and the limit is 36 digits (`10**36` raises `NumOutOfRangeError`).
- Lines with no ASCII letter and no Chinese character (`2024`, `?!`, `...`, `1 2 3 4`, the empty string) raise `Unknown language: ...` in the reference; Chinese lines succeed there (they are recorded with `has_chinese`) and are what the C# front end rejects explicitly.
- The whole dump is deterministic: regenerating into an empty directory reproduces every hash.

## What the MuLan text golden run showed

- **Bit-for-bit reproducible, across processes too**: regenerating `tests/fixtures/mulan_text` into an empty directory reproduces every hash (unlike the BigVGAN run, whose convolutions vary with thread scheduling). The tower is short enough that no summation order changes.
- The 199 XLM-RoBERTa tensors inside MuLan are bit-identical to `xlm-roberta-base` (asserted by the converter): MuLan does not fine-tune the encoder.
- The reference normalizer changes 4,837 of the 196,608 code points of planes 0 to 2 when they are taken alone.
- The stages chained by hand (encoder, transformer, mean, projection, L2) equal `mulan(texts=[t])` exactly.
- Converting the pickle twice gives byte-identical files.

## What the song pipeline dumps showed

- **libtorch's CPU generator is PyTorch's**: `randn(1, 10, 64)` after `manual_seed(1234)` is identical in both, in float32 and in float16, for consecutive blocks. The float16 draw is **not** the float32 draw cast to half (largest difference about 5), so the noise has to be drawn in the model's dtype.
- **pedalboard's 16-bit conversion** is `round_half_even(clamp(x, -1, 1) * 2147483647) >> 16`: 0 differences on 300,000 samples including values beyond the interval and quantization boundaries. `round(x * 32767)` (the core's writer) is off by up to 2 steps.
- **Fake stereo**: `right = left * float32(0.8)` delayed by 480 samples, the first 480 zero.
- **The service's validation** has 13 distinct messages; the fixture holds 56 bodies (28 refused) with the exact message of each.
- **The reference song** (seed 1234, 4 s) is 20 frames in 2 blocks; the float32 latent is 1.2e-5 (block 0) and 1.2e-3 (block 1) away from a float64 run of the same network and noise: that sensitivity is the yardstick the C# gate uses.
- Both dumps are reproducible: regenerating into an empty directory gives the same hashes (the reference is bit-for-bit deterministic on these short runs).

## Regenerating

Dumping again into an empty directory must reproduce the hashes of `MANIFEST.md` (same image, same weights, same CPU thread count). If a hash differs, check the BLAS/thread configuration of the machine before assuming a code change: float summation order can vary with the CPU.
