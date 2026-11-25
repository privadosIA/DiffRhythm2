# BigVGAN golden run (fp32, CPU)

Dumped by `tools/dump_bigvgan_reference.py` from the reference image `veritmo/diffrhythm2:1` (upstream 7804f82). Stage signals keep only the first 2 channels (`conv_pre` keeps all) and only for the stages in `RAW_STAGES` of `tools/dr2_bigvgan.py`; `*_rms` is the RMS over time of every channel, for every stage. See `meta.json` for the reference's own run-to-run drift.

| file | sha256 | bytes |
|---|---|---|
| `activation.safetensors` | `0dfd82f33e706590ea3935f6cdace72dd36551c453953a9d8cc91ced3f7be192` | 114072 |
| `chunk_loud.safetensors` | `16ce31d6fd695919437688f66561252dedfdbb0a425e6a131a2030318c002c75` | 773272 |
| `chunk_random.safetensors` | `13fd5fdb2bdd5e9f0b397f5ecd1db21d8e185177548e5b597d3a89fdbdec962a` | 4663692 |
| `chunk_real.safetensors` | `8a1c33002c9abab4dcde70a4e263d6b990813612ee211b3289a9a1f388726873` | 4663692 |
| `decode_40.safetensors` | `fc6c4f0a9e1d63fc4c7e9c0bd6009a95e4b8d083d0955a67ba060d16cc04b41c` | 1546392 |
| `meta.json` | `b6b56017e50ff40b731860f85d2b8be3b36a9a2488b9772726d67ae0e1fbc78f` | 709 |
| `weights.safetensors` | `19a23674c728e9c46c0b701ea540e409af7ed03cba295527cb302cd164b15bd3` | 561696 |
