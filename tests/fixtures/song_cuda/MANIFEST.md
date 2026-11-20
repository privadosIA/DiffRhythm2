# Reference on CUDA (seed 1234) - INFORMATIONAL, never a gate

`noise_cuda.safetensors`: the CUDA draws after `manual_seed(1234)` (`tools/dump_song_cuda_reference.py`, identical on the RTX 4070 and the RTX 5060 Ti). `song_cuda.safetensors`: the service's own song on CUDA in half precision for the same request as the CPU reference song (`tools/dump_song_cuda_e2e_reference.py`, RTX 4070).

| file | sha256 | bytes |
|---|---|---|
| `meta.json` | `c2d4eca87e63f9d33db483197fb9ad4397167fa27c0f827b63f5a23c55033aec` | 420 |
| `noise_cuda.safetensors` | `cd784b13779f7f51edeb199bd5178c74f07e156c97ee76e90fca0570f0eca2a4` | 20656 |
| `song_cuda.safetensors` | `4184f5efb84f389d7e85ab805ba7bb59db105b8240f0bd6a53e5cfbff75c3ec9` | 2309352 |
| `song_cuda_meta.json` | `f5412b7ee496056252a941e9dc09a0bc3ba8e170b6f2017fca6d03800bf2799c` | 289 |
