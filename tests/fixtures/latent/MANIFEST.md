# Golden-run latent (input of the decoder)

The sampler output of the fp32 golden run (tests/fixtures/dit, 40 frames = 8 s at 5 frames/s), exactly as `CFM.sample_block_cache` returns it: `latent` has shape **[1, T, 64]** (batch, frames, channels). The server (and `decode_audio`) work on **[1, 64, T]**, i.e. after `latent.transpose(1, 2)`; the decoder change must do that transposition itself. dtype float32, one frame = 9600 output samples at 48 kHz.

| file | sha256 | bytes |
|---|---|---|
| `latent_40.safetensors` | `30e72c6e5a0bc3aa86d715946c1a566204b8b90f808e6042ee33f5e3de9637df` | 10320 |
