# Song pipeline small fixtures (reference image, CPU)

Dumped by `tools/dump_song_reference.py`: the reference's CPU noise for seed 1234, what `server.Engine.validate` returns for a list of bodies, `make_fake_stereo`, and the 16-bit integers pedalboard writes for a long signal.

| file | sha256 | bytes |
|---|---|---|
| `audio.safetensors` | `3428e1e9b4d7421b56dd1a961ab9314a6c55242715e2dc7564dff5f07e05aeab` | 2040320 |
| `meta.json` | `7d08efaed2c76c7eeab2c48e3d14b500b360ef55a2be6fd39c201f58ac76a0f2` | 202 |
| `noise.safetensors` | `19ffc96c178696c4ec77908e4a64c70ce85c41e218f26e4311daadac950a7ebe` | 30968 |
| `validate.json` | `8723dce4896871cfea0fae2331e784335249e4935373b0124a886d6a7d2b6e6c` | 555807 |
