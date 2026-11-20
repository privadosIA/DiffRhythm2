# Song pipeline: lyrics and a style to a WAV

`SongGenerator` (namespace `DiffRhythm2.Pipeline`) joins the four stages of DiffRhythm2 and reproduces what the reference service does around them. The `diffrhythm2` executable (`src/DiffRhythm2.Cli`) is a thin command line over it. Python reference: the service in `mediaorquestator/diffrhythm2:1` (`/opt/server.py`, `Engine._sing`).

```
SongRequest --Validate--> lyrics prepared, style normalized
  1 Lyrics   tokens = LyricsTokenizer(EnglishG2p(espeak-ng))                       stays loaded (small)
  2 Style    StyleEncoder (float32, on the device) -> embedding [1,512] -> cast to the network's dtype -> released
  3 Network  seed -> noise (one [1,10,64] block per 10 frames) -> DiT + BlockSampler -> latent [1,T,64] -> released
  4 Decoder  BigVGAN (chunked) -> mono audio, T * 9600 samples at 48 kHz -> released
SongResult.WriteWav -> fake stereo (optional) -> 16-bit PCM WAV
```

Only one model is in memory at any moment (peak: the network plus its attention caches).

## Requirements

- The model directory with six files (below), `libespeak-ng` (ideally 1.51), and a TorchSharp native backend. See the *Requirements* section of the repository's README for where each comes from.
- **CPU** needs about 4.7 GiB of RAM at the peak. **GPU** (float16) needs about 2.6 GiB of card memory over the idle level for a one-minute song, and also about 4.7 GiB of host RAM while the weights are read.

### Model directory

| file | used by |
|---|---|
| `config.json`, `model.safetensors` | diffusion transformer |
| `decoder.json`, `decoder.safetensors` | audio decoder |
| `mulan_text.safetensors`, `mulan_text_tokenizer.json` | style encoder |

`ModelDirectory` lists any missing file, all at once, before anything is loaded. `diffrhythm2 install --model DIR --accept-licenses` downloads and converts all of it ([weights-tools.md](weights-tools.md)). To make songs over HTTP instead of from a terminal, see [http-service.md](http-service.md).

## The library

```csharp
using DiffRhythm2.Pipeline;

var generator = new SongGenerator(new SongOptions(
    ModelDirectory: "/path/to/models",
    Device: "cpu",                        // "cpu", "cuda" or "cuda:N"
    Precision: null));                    // null: float32 on CPU, float16 on CUDA (what the service runs)

var request = new SongRequest(lyrics, "Pop, Piano, Female vocals, Happy, Bright", Seconds: 60, CfgStrength: 2.0, Steps: 16, Seed: 1234);
SongResult song = generator.Generate(request, progress: null, cancellation: default);

song.WriteWav("song.wav");                // 16-bit, 48 kHz, stereo like the service; stereo: false for mono
byte[] bytes = song.ToWavBytes();         // the same file in memory
float[] mono = song.Audio;                // -1..1 at song.SampleRate
```

`samples/MakeSong/Program.cs` is a complete program. To use the library from another repository, reference `src/DiffRhythm2.Net/DiffRhythm2.Net.csproj` by path and add a TorchSharp backend package (`TorchSharp-cpu`, `TorchSharp-cuda-linux` or `TorchSharp-cuda-windows`, version `0.107.*`); the library's own build finds `musicgen-core`.

### The request and its limits

They are the limits of the reference service, with its messages (`SongRequestException.Field` names the field: `lyrics`, `style`, `max_secs`, `cfg_strength`, `steps`, `seed`). Everything is checked before any model is loaded.

| field | rule |
|---|---|
| `Lyrics` | not empty, 20,000 characters at most, at least one line of text besides structure tags. English only (Chinese is refused with an explicit error). Structure tags on their own lines: `[start] [intro] [verse] [chorus] [bridge] [outro] [inst] [solo] [hook] [break] [end] [stop] [space]`, any case; other tags are dropped; `[start]` is added if missing |
| `Style` | comma-separated words, normalized to `a, b, c`; at least one, 500 characters at most. `SongRequest.FromWords` takes a list |
| `Seconds` | 10 to 210 (default 210). A ceiling: the song stops by itself at the end of the lyrics |
| `CfgStrength` | 0.5 to 10 (default 2.0) |
| `Steps` | 4 to 64 (default 16) |
| `Seed` | 0 or more; `null` picks one (`SongResult.Seed`) |

A line without any ASCII letter (`2024`, `1 2 3 4`, `...`) cannot be sung and raises `NoSungTextException` naming the line, as in the reference service, where it also stops the song.

### The result

`SongResult`: `Audio` (mono, `Frames * 9600` samples), `SampleRate` (48,000), `Frames`, `Seed`, `EndedEarly` (the model finished before the requested length), `BlocksSampled`, `BlocksRequested`, `Elapsed`, `Device`, `Precision`.

### Progress and cancellation

`Generate(request, IProgress<SongProgress>, CancellationToken)`. Events, in order: for each stage `StageStarted`, `ModelLoaded`, (`BlockDone` with `Done`/`Total` during `SongStage.Network`), `ModelReleased`, `StageFinished`. Use an `IProgress` that does not post to a context if the order of events matters (`Progress<T>` posts to the thread pool in a console program). The token is checked between blocks (a block takes a few seconds on CPU and a fraction of a second on a GPU); on cancellation `OperationCanceledException` is thrown, the models are released and nothing partial is returned.

### Errors

| exception | when |
|---|---|
| `SongRequestException` | a request rule is broken (`Field` says which) |
| `ModelDirectoryException` | files are missing (`Missing` lists them all) |
| `DeviceUnavailableException` | the device does not exist (for example `cuda` with the CPU backend) |
| `EspeakUnavailableException` | `libespeak-ng` cannot be loaded; the message lists where it looked |
| `NoSungTextException`, `ChineseNotSupportedException`, `LyricsLineException` | a line cannot be tokenized; `LyricsLineException` carries the line number of the lyrics you gave |
| `SongException` | other pipeline failures (for example the model ended the song before any audio) |

`SongException`, `G2pException` and `EspeakUnavailableException` are the families to catch; `OperationCanceledException` signals cancellation.

## Seeds and what they reproduce

The pipeline seeds the global generator (CPU and every CUDA device) after the lyrics and the style are done and right before sampling, then draws one `[1, 10, 64]` block of noise per block of song, in order, in the network's dtype and on its device: the calls the reference service makes.

- **Same seed, same device, same precision: the same song.** Verified byte for byte (CPU; and a GPU against itself).
- **CPU, float32: the reference's song.** libtorch's CPU generator produces PyTorch's numbers. Against a 4-second song made by the reference with seed 1234 the tokens and the noise are identical, the style embedding is within 4e-7, the latent within 5e-7 (block 0) and 1.6e-5 (block 1, where the reference itself is 1.2e-3 away from a float64 run), the waveform within 1.1e-5 of the peak and the WAV within one 16-bit step.
- **CUDA: the reference's CUDA noise.** The draws equal the reference's CUDA stream exactly, in float16 and float32, on an RTX 4070 and an RTX 5060 Ti. A 4-second song from seed 1234 is within 44 dB SNR of the one the service makes on the GPU (the difference is half-precision arithmetic).
- **Not reproduced: the same seed across different GPU models over a long song.** Float16 kernels differ between architectures, the sampler amplifies the difference over many blocks, and a one-minute song from the same seed on an RTX 4070 and on an RTX 5060 Ti are different songs. The same card repeats exactly. The CPU stream and the CUDA stream differ from each other, as in PyTorch.
- Float16 and CUDA results are measured against the reference and reported, but never used as a pass/fail gate in the tests.

## The command line

```
diffrhythm2 generate --model DIR --lyrics FILE|- --style TEXT --out FILE.wav
                     [--seconds N] [--seed N] [--cfg X] [--steps N] [--device cpu|cuda|cuda:N] [--dtype fp32|fp16] [--mono]
                     [--espeak-lib FILE] [--espeak-data DIR]
diffrhythm2 check    --model DIR [--device D]
diffrhythm2 install  --model DIR --accept-licenses     # fetch + convert + check (also: fetch, convert; see weights-tools.md)
diffrhythm2 --help
```

`generate` prints `seed=N` first (always, also when it was picked), progress lines to the error stream, and `wrote ...` at the end. The WAV is written through a temporary file, so a failure leaves nothing. `--lyrics -` reads standard input. `check` lists the files with their sizes, the phonemizer library and its version, and the devices, and exits 0 only if a song can be made on the device.

| exit code | meaning |
|---|---|
| 0 | done |
| 1 | failure while working (missing files, no such device, a line that cannot be sung, library not found) |
| 2 | wrong usage: unknown or missing option, a value outside the limits (the message names the option) |
| 130 | cancelled (Ctrl+C); no file written |

## Performance

Measured on one machine (20 CPU threads, RTX 4070 and RTX 5060 Ti, idle), songs from one-minute lyrics:

| run | time to make | audio | notes |
|---|---|---|---|
| CPU, float32, 10 s | 30 s | 10 s | |
| CPU, float32, 20 s | 62 s | 20 s | peak host RSS 4.7 GiB |
| CPU, float32, 60 s | 183 s | 60 s | about 3 times real time |
| CUDA, float16, 60 s (RTX 4070 or 5060 Ti) | 16 to 17 s | 60 s | model loading included; 2.6 GiB of card memory over idle; host RSS peak 4.7 GiB |

On CUDA, TorchSharp offers no way to give the allocator's cache back to the driver, so memory freed by one stage stays reserved by the process and is reused by the next.

## Troubleshooting

- *"libespeak-ng could not be loaded"*: install espeak-ng (1.51 is the reference's) or set `DIFFRHYTHM2_ESPEAK_LIB` (and `DIFFRHYTHM2_ESPEAK_DATA`) / `--espeak-lib`. If its audio dependencies are missing too, `tools/fetch_espeak_reference.sh DIR` copies a working set out of the reference image; run with `LD_LIBRARY_PATH=DIR`.
- *"cuda was requested but no CUDA device is available"*: the default build has the CPU backend. Build with `-p:TorchBackend=cuda-linux` (or `cuda-windows`); the first restore downloads about 2 GB of libtorch.
- *The song does not sound like the reference's*: check the espeak version (`check` shows it) and that the same device kind and precision are used; see the section on seeds.
- *A line stops the song with "no sung text"*: lines made only of digits or symbols cannot be sung; write them out in words.
- *Out of memory on a GPU*: use float16 (`--dtype fp16`, the default on CUDA); the network needs about 2.3 GiB plus caches that grow with the song.
