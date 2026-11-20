# Chunk stitching reference

Which chunk (1-based) supplies each latent frame in the real `Generator.decode_audio` (chunk 20, overlap 5), by running it with a stand-in decoder that tags its output with the chunk number. Dumped by `tools/dump_stitch_reference.py`; no weights involved.

| file | sha256 | bytes |
|---|---|---|
| `stitch.json` | `ceec99be95da4c27f1dc5d1fe1ce4dda409e1ec7195a83ae5a6a7b411efca9eb` | 2376 |
