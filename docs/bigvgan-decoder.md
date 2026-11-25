# BigVGAN decoder

C# port of DiffRhythm2's BigVGAN generator and its chunked latent decoder (`src/DiffRhythm2.Net/Decoder`). Python reference: `veritmo/diffrhythm2:1` (upstream commit 7804f82, `bigvgan/model.py`).

## Usage

```csharp
using DiffRhythm2.Decoder;
using TorchSharp;

// decoder.json + decoder.safetensors (converted once from decoder.bin, see tools/README.md), strict, weight-norm folded at load
var generator = BigVganWeights.LoadDirectory("/path/to/checkpoint", torch.CPU);
var decoder = new LatentDecoder(generator);

using var _ = torch.no_grad();
Tensor samplerLatent = ...;                                          // [1, T, 64], what BlockSampler returns
using var latent = LatentDecoder.FromSamplerLayout(samplerLatent);   // [1, 64, T]
using var wave = decoder.Decode(latent);                             // [1, 1, T * 9600], float32, in [-1, 1]
LatentDecoder.WriteWav("song.wav", wave, generator.Config.SamplingRate);   // mono, 48 kHz, 16-bit PCM
```

- One latent frame is 9600 samples (10x10x8x3x2x2 upsampling) at 48 kHz, so 5 frames per second.
- `T >= 20` is decoded exactly like the reference's `decode_audio`: chunks of 20 frames every 15, a final chunk aligned to the end, each overlap cut at its midpoint (2 frames from each edge). `ChunkPlan.Create` is that geometry as a pure function.
- `T < 20`, which the reference cannot decode (it fails), is decoded as one chunk; `T = 0` returns an empty waveform. This is the only deliberate deviation.
- The output buffer is float32 whatever the model precision, as in the reference. Fake stereo and loudness handling are not part of this decoder.

## Checkpoint

`decoder.bin` is a pickle, so `diffrhythm2 convert` (see [weights-tools.md](weights-tools.md); `tools/convert_decoder.py` is the Python oracle it is compared with, byte for byte) converts it to `decoder.safetensors` without folding anything; the fold is done in C# by the core's `WeightNorm` (`StrictLoader` with `FoldWeightNorm`). The anti-alias filters are buffers in the checkpoint and are loaded like parameters; `KaiserSinc` recomputes them and a test proves they are identical. `conv_post` has no bias, so the model has no such parameter and a checkpoint that carries one is rejected by name.

## Parity and tolerances

fp32 on CPU is the gate (`DecoderTolerances.cs`, measured figures and the reference's own cross-process variance in the comments); fp16 is only reported (`BigVganFp16InformationalTests`).

| comparison | measured (relative to the largest magnitude) | gate |
|---|---|---|
| `conv_pre`, `up0`..`up2`, `stage0`..`stage2` (first 2 channels) | 2e-7 .. 1.1e-6 | 5e-5 |
| `stage3`, `stage4` | 1.0e-6 .. 1.1e-6 | 5e-5 |
| `stage5` (values up to 6.7) | 4.3e-6, 5.5e-6 | 5e-5 |
| per-channel RMS of every stage, all channels | <= 5.6e-7 | 1e-5 |
| 20-frame waveform, real / random latent | 1.1e-6 / 1.5e-6 | 5e-5 |
| 40-frame waveform; seam windows | 1.9e-6; 1.5e-6, 9.4e-7 | 5e-5 |
| clipped loud chunk (92074 samples at +-1) | 1.6e-5 | 5e-5 |
| folded weights (4 convolutions) | <= 1.2e-7 | 5e-5 |

The reference differs from itself between processes by about the same amount (thread scheduling changes the summation order of the convolutions): 1.4e-6 on the 40-frame waveform and 1.6e-5 on the loud one. fp16 on CPU deviates from the fp32 reference by 2.8e-3 (mean 2.9e-4); the C# fp16 differs from the CPU half reference by 7.3e-4.

## Performance (CPU, fp32, idle machine, 20 threads)

- Load: 523 MB of tensors, strict.
- 40 frames (8 s of audio, 3 chunks): 9.7 s, about 3.2 s per chunk.
- Memory: the process peaks at about 245 MB above its resident set while decoding, for 40 frames and for 400 (resident set 2.5 GB in the test process with the model, fixtures and libtorch). Growing from 40 to 400 frames added nothing beyond the 14 MB of output buffer (test `LongSequence_DoesNotNeedMoreThanAChunkPlusTheOutputBuffer`); each chunk's activations are freed before the next chunk is decoded.
