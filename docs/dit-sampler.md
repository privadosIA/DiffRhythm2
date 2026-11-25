# DiT and block-wise sampler

C# port of DiffRhythm2's diffusion transformer and its block-wise guided flow-matching sampler (`src/DiffRhythm2.Net`). Python reference: `veritmo/diffrhythm2:1` (upstream commit 7804f82).

## Usage

```csharp
using DiffRhythm2.Model;
using DiffRhythm2.Sampling;
using TorchSharp;

var (dit, report) = DitWeights.LoadDirectory("/path/to/checkpoint", torch.CPU);   // config.json + model.safetensors, strict
var sampler = new BlockSampler(dit);
// tokens: [L] int64 lyric ids; style: [1, 512]; noise: one [1, 10, 64] tensor per block (ceil(frames / 10) of them)
using var _ = torch.no_grad();
Tensor latent = sampler.Sample(tokens, style, noise, frames: 40, new SamplerOptions(Steps: 16, CfgStrength: 2.0));
// latent: [1, T, 64], as the reference returns it; the decoder wants it transposed to [1, 64, T]
```

- Tokens and style come from the later phases (g2p, MuQ-MuLan); for now they are read from the golden run (`tests/fixtures/dit/inputs.safetensors`).
- The initial noise is an input of the sampler, so a run can be checked against a fixed reference. `SongGenerator` draws it the way the reference does (see [song-pipeline.md](song-pipeline.md)): libtorch's CPU generator produces PyTorch's numbers for the same seed, dtype and shape.
- Whole blocks are sampled: 45 frames give 50, like the reference. Only the end-of-song rule shortens the result.
- Loading skips the training-only `projectors.*` tensors of the checkpoint and reports them (`report.IgnoredKeys`); any other discrepancy aborts before reading.

## What differs from stock Llama

`q_norm`/`k_norm` (RMSNorm over the head dimension, before RoPE), a single conditioning injection at the input plus an adaptive final norm, no causal mask, and positions that continue over the clean blocks only (the text shares positions `0..L-1` with the first blocks). See `openspec/changes/archive/2026-10-22-add-dit-block-flow-matching/design.md`.

## Parity and tolerances

| comparison | measured | gate |
|---|---|---|
| single pieces and whole-DiT forward (with and without cache, both guidance branches) | <= 9e-7 relative | 1e-5 |
| first block, every one of its 16 states; 10 frames without lyrics; guidance off | <= 9e-7 relative | 1e-5 |
| blocks 0-4 of the multi-block run | 2.5e-6, 1.4e-4, 2.9e-4, 5.2e-2, 1.0e-2 max abs | <= the reference's own distance to float64 |

The multi-block gate is not a fixed number because the sampler amplifies rounding: the fp32 reference is itself 1.2e-5, 1.2e-3, 2.5e-3, 0.19 and 3.4e-2 away from a float64 run of the same network on blocks 0-4, while the port is 0.12 to 0.29 times that distance from the reference and *closer to float64 than the reference is* (block 3: 0.148 against 0.192). A mismatch that the fixed tight gates on single forwards and the first block do not catch would have to survive those gates.

fp16 (production precision) is only reported, never gated (`Fp16InformationalTests`).

## Fixtures

`tools/README.md` explains how `tests/fixtures/{dit,eos,dit_fp16,dit_fp64,latent}` are produced; all have a `MANIFEST.md` with sha256 hashes verified by the tests.

## Performance (CPU, fp32, 20 logical cores, idle machine)

| | |
|---|---|
| working set after loading the model | 4.24 GiB |
| peak working set (loading + a 4-block run) | 4.34 GiB |
| 4-block (40 frames, 8 s) run, 16 steps, guidance 2.0 | 12.7 s |

The caches hold K/V of 16 layers for the lyrics plus every clean block, twice (conditional and unconditional); they stay negligible next to the weights for songs of this size. A 210 s song has not been measured.
