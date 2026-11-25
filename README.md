# DiffRhythm2.Net

> Part of [Veritmo](https://veritmo.xuxo.net/): a song made with this port, and examples of every other engine, are on its public page (es/en).

C# port (.NET 8 + TorchSharp) of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2), the diffusion model that writes a whole song (voice and accompaniment) from English lyrics and a style text. Everything runs in .NET: the lyrics front end, the style encoder, the diffusion transformer with its block-wise sampler, and the audio decoder. Python is only used as a development-time oracle (the reference outputs the tests compare against are committed).

```
lyrics ---> phonemes ---> token ids ---------------------------+
style text ---> XLM-R tokenizer ---> MuLan text tower ---------+--> block-wise flow-matching sampler (DiT, Euler + CFG)
                                                               |            latent [T, 64] at 5 frames/s
                                                               +--> BigVGAN decoder ---> 48 kHz audio ---> stereo WAV
```

Status: the whole pipeline is ported and verified stage by stage against the Python reference, and joined end to end (`SongGenerator`, `diffrhythm2 generate`). On CPU in float32 the same seed gives the same song as the reference. The model files are downloaded and converted by the program itself (`diffrhythm2 install`, no Python), and an HTTP service with the API of the orchestrator's Python engine is included (`diffrhythm2-service`, with Swagger UI).

## Requirements

| what | notes |
|---|---|
| .NET SDK 8 or newer | the library targets `net8.0`; newer runtimes work (`RollForward=Major`) |
| the model directory | six files, see below |
| `libespeak-ng` | the phonemizer library. The reference uses **espeak-ng 1.51**; other versions may pronounce some words differently. It is a runtime dependency and is not bundled (GPL-3). Debian 12 has 1.51; `tools/fetch_espeak_reference.sh` copies the exact build out of the reference image |
| a TorchSharp native backend | `cpu` (default), or `-p:TorchBackend=cuda-linux` / `cuda-windows` for an NVIDIA GPU (libtorch 2.7, CUDA 12.8: runs on Ada and Blackwell cards) |
| `musicgen-core` | shared model-agnostic code, expected at `../musicgen-core` (override with `-p:MusicGenCoreDir=...`) |

### The model directory

`diffrhythm2 install --model DIR --accept-licenses` downloads and converts everything that is missing (about 7.7 GB the first time, no Python; see [docs/weights-tools.md](docs/weights-tools.md)); `diffrhythm2 check --model DIR` tells what is missing. MuQ-MuLan is licensed CC-BY-NC-4.0 (non-commercial): the licenses must be accepted explicitly.

| file | what | produced by |
|---|---|---|
| `config.json`, `model.safetensors` | diffusion transformer (4.5 GB, as published) | download from the DiffRhythm2 repository |
| `decoder.json` | audio decoder configuration | download |
| `decoder.safetensors` | audio decoder weights | `diffrhythm2 convert` from the published `decoder.bin` (a `torch.save` pickle) |
| `mulan_text.safetensors`, `mulan_text_tokenizer.json` | style text encoder (1.3 GB) | `diffrhythm2 convert` from MuQ-MuLan-large and XLM-RoBERTa's `tokenizer.json` |

`install` does the download and the conversions in C#. The Python scripts in `tools/` are only the test oracle: the C# outputs are byte for byte what they write.

## Make a song from a terminal

```bash
dotnet build -c Release                                   # add -p:TorchBackend=cuda-linux for the GPU
export DIFFRHYTHM2_ESPEAK_LIB=/path/to/libespeak-ng.so.1  # if it is not in the system library path
dotnet src/DiffRhythm2.Cli/bin/Release/net8.0/diffrhythm2.dll check --model ~/models/diffrhythm2
dotnet src/DiffRhythm2.Cli/bin/Release/net8.0/diffrhythm2.dll generate --model ~/models/diffrhythm2 \
    --lyrics lyrics.txt --style "Pop, Piano, Female vocals, Happy, Bright" --out song.wav --seconds 60 --seed 1234
```

It prints `seed=1234`, progress on the error stream, and writes a 16-bit 48 kHz stereo WAV. Options and exit codes: `diffrhythm2 --help`, or [docs/song-pipeline.md](docs/song-pipeline.md).

## Recognize and compare songs

The same MuQ-MuLan that embeds the style text also embeds **audio** into the same space, so a song can be ranked against tags ("This is a jazz song.") or compared with another song, with no Python. It needs the audio tower, converted once with `diffrhythm2 convert --model DIR --audio` (1.3 GB, nothing more is downloaded), and a 24 kHz WAV (`ffmpeg -i in.mp3 -ac 1 -ar 24000 out.wav`):

```bash
diffrhythm2 classify --model ~/models/diffrhythm2 --audio song.wav --vocab docs/mulan-vocabulary-example.json --top 3
# genre: pop 0.4981, r&b 0.3907, country 0.3666
diffrhythm2 embed --model ~/models/diffrhythm2 --audio song.wav --out song.json     # the 512-value embedding, as JSON
```

The weights are CC-BY-NC-4.0 (non-commercial). Guide: [docs/mulan-audio.md](docs/mulan-audio.md).

## Serve it over HTTP

```bash
DIFFRHYTHM2_CKPT=~/models/diffrhythm2 dotnet src/DiffRhythm2.Service/bin/Release/net8.0/diffrhythm2-service.dll
# diffrhythm2 service 1 listening on :8788 (models: ...)      then open http://localhost:8788/docs
curl -s -X POST localhost:8788/generate -H 'Content-Type: application/json' -d '{"lyrics":"[verse]\nHello world\nWe sing along","style":"Pop, Piano","seed":1}'
```

It makes one song at a time through a queue and offers `GET /health`, `POST /generate`, `GET /jobs/{id}`, `GET /jobs/{id}/audio` and `DELETE /jobs/{id}`, the same calls as the Python engine of Veritmo (which can use it as an external engine by its host and port, with no change). Variables, every call with `curl`, errors, tokens and a polling example are in [docs/http-service.md](docs/http-service.md); the API description is served at `/openapi.yaml`.

`scripts/run-service.sh` starts it (and builds it first if needed), and `scripts/diffrhythm2-service.service` is a systemd user unit; both are explained in the guide.

Every song is made in a worker process of its own, so the memory it used, video memory included, goes back to the system when it ends (`DIFFRHYTHM2_ISOLATION=inprocess` turns that off); see "Memory and speed" in the guide.

The port is **8788** by default and is set with `DIFFRHYTHM2_PORT` (for example `DIFFRHYTHM2_PORT=9000`); an invalid value stops the service at start with a message naming the variable. **It listens on all network interfaces** and speaks plain HTTP, so anyone who can reach the port can ask it for songs. On a machine that is reachable from a network, set `DIFFRHYTHM2_TOKEN` (every API call then needs `Authorization: Bearer <token>`) or put it behind a firewall or a reverse proxy; the token is sent unencrypted without HTTPS, so keep it to a trusted network.

## Use it from your own C# code

Reference the library by path (from another repository too: its build finds `musicgen-core` on its own) and add the TorchSharp backend you want:

```bash
dotnet add reference /path/to/diffrhythm2/src/DiffRhythm2.Net/DiffRhythm2.Net.csproj
dotnet add package TorchSharp-cpu --version "0.107.*"     # or TorchSharp-cuda-linux / TorchSharp-cuda-windows
```

```csharp
using DiffRhythm2.Pipeline;

var generator = new SongGenerator(new SongOptions(ModelDirectory: "/path/to/models"));     // CPU, float32; Device: "cuda" for the GPU
var request = new SongRequest(lyrics, "Pop, Piano, Female vocals", Seconds: 30, Seed: 1234);
SongResult song = generator.Generate(request);        // throws SongRequestException, ModelDirectoryException, ... see the guide
song.WriteWav("song.wav");                            // stereo; WriteWav(path, stereo: false) for mono. song.Audio is the mono float[] at 48 kHz
```

`samples/MakeSong` is this program with progress output; it is built with the solution. The complete guide (request limits, progress and cancellation, errors, seeds, devices, performance, troubleshooting) is [docs/song-pipeline.md](docs/song-pipeline.md). Every public type of `DiffRhythm2.Pipeline` has XML documentation.

## Reproducibility in one paragraph

The seed fixes the random numbers exactly as the reference does (same generator, same draws, same dtype). On the same kind of device the same seed repeats the song, and on CPU it gives the reference's song. On a GPU the noise equals the reference's CUDA noise, but half-precision arithmetic differs between GPU models and the sampler amplifies it, so a long song from the same seed can differ between two different cards (the same card repeats exactly). Details in the guide.

## Repository

```
src/DiffRhythm2.Net/     the library: Text/ (lyrics, espeak, tokenizers), StyleText/ (XLM-R tokenizer, MuLan text tower), AudioTower/ (MuLan audio tower: WAV, mel, MuQ-large, ranking), Model/ + Sampling/ (DiT and sampler),
                         Decoder/ (BigVGAN), Pipeline/ (the joined pipeline and the audio output), Weights/ (download, torch.save reader, conversion)
src/DiffRhythm2.Cli/     the `diffrhythm2` executable (generate, check, install, fetch, convert, embed, classify)
src/DiffRhythm2.Service/ the `diffrhythm2-service` HTTP service (ASP.NET Core; Swagger UI at /docs)
samples/MakeSong/        the smallest program that makes a song
scripts/                 run-service.sh (starts the HTTP service) and a systemd user unit for it
tests/                   xUnit tests and the committed reference fixtures (tests/fixtures/*, with sha256 manifests)
tools/                   Python dumpers that produced the fixtures, run in the reference Docker image; weight conversion tools
docs/                    one guide per component; openspec/ holds the specs and the history of the changes
```

## Tests

```bash
dotnet test -c Release        # without models the tests that need them are reported as skipped, with the reason
DIFFRHYTHM2_CKPT=~/models/diffrhythm2 DIFFRHYTHM2_ESPEAK_LIB=... DIFFRHYTHM2_ESPEAK_DATA=... dotnet test -c Release     # the full run
```

Environment variables and the exact commands are in [CLAUDE.md](CLAUDE.md).

## Guides

| | |
|---|---|
| [docs/song-pipeline.md](docs/song-pipeline.md) | the joined pipeline, the CLI, seeds, devices, performance |
| [docs/weights-tools.md](docs/weights-tools.md) | getting the model directory: download, licenses, conversion |
| [docs/http-service.md](docs/http-service.md) | the HTTP service (same API as the orchestrator's Python engine), Swagger UI, `curl` examples |
| [docs/english-g2p.md](docs/english-g2p.md) | lyrics to token ids, espeak-ng |
| [docs/mulan-text.md](docs/mulan-text.md) | style text to embedding |
| [docs/mulan-audio.md](docs/mulan-audio.md) | recognizing and comparing songs: audio to embedding, tag ranking, `embed` and `classify` |
| [docs/dit-sampler.md](docs/dit-sampler.md) | the diffusion transformer and its sampler |
| [docs/bigvgan-decoder.md](docs/bigvgan-decoder.md) | latent to audio |

## API reference

The public API is documented with XML comments (`///`, the C# counterpart of Javadoc); the `Pipeline` namespace is complete and checked by a test, the rest is
partial. They show up as tooltips in the editor and feed an HTML reference, together with the guides above, built with [DocFX](https://dotnet.github.io/docfx/)
(pinned as a local tool in `.config/dotnet-tools.json`):

```bash
dotnet tool restore
TorchBackend=cpu dotnet docfx docs/docfx.json --serve   # builds docs/_site and serves it on http://localhost:8080
```

`docs/api` and `docs/_site` are generated and ignored by git. The home page is `docs/index.md`; the guides are the Markdown files of `docs/`. The CI builds the
site on every push with `--warningsAsErrors`, so a broken `<see cref>` or a broken link between guides fails that step; the site is not published anywhere.

## Third-party notes

DiffRhythm2 and its published weights come from ASLP-lab (Apache-2.0 code); the style encoder uses MuQ-MuLan and XLM-RoBERTa, and the phoneme vocabulary comes from the reference (derived from Amphion, MIT). Check the license of the model weights you download before using the songs. espeak-ng and phonemizer are GPL-3: this project only calls the espeak-ng library found on the machine and does not redistribute it.

The HTTP service (`src/DiffRhythm2.Service`) bundles **Swagger UI 5.32.0** (`swagger-ui-dist`, Apache-2.0, https://github.com/swagger-api/swagger-ui) as the page at `/docs`, so it works offline; its license and notice are in `src/DiffRhythm2.Service/Docs/swagger-ui/`.

## Related repositories

Depends on `musicgen-core` (shared code) and follows the original DiffRhythm2. Used by `veritmo`, which starts it as a service.

| Repository | What it is | Licence |
|---|---|---|
| [`veritmo`](https://github.com/privadosIA/veritmo) | The server: the HTTP API and its contract, the pipelines (video, images, music, 3D), the run history and the services it starts. | Apache-2.0 |
| [`dashboards-labs`](https://github.com/privadosIA/dashboards-labs) | The dashboards (Angular) that talk to the server through its OpenAPI contract. | Apache-2.0 |
| [`forms-lib`](https://github.com/privadosIA/forms-lib) | `@core/dynamic-forms`, the Angular library that draws forms and tables from JSON. | Apache-2.0 |
| [`heartmula`](https://github.com/privadosIA/heartmula) | C# (.NET 8 + TorchSharp) port of [heartlib](https://github.com/HeartMuLa/heartlib): the HeartMuLa song generator, run by the server as a service. | Apache-2.0 |
| [`DiffRhythm2`](https://github.com/privadosIA/DiffRhythm2) (this repository) | C# port of [DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2): a diffusion model that writes a whole song. | Apache-2.0 |
| [`lion-clap`](https://github.com/privadosIA/lion-clap) | C# port of [LAION-CLAP](https://github.com/LAION-AI/CLAP): audio and text embeddings, used to recognise songs. | Apache-2.0 |
| [`musicgen-core`](https://github.com/privadosIA/musicgen-core) | The TorchSharp building blocks shared by the five ports. | Apache-2.0 |
| [`triposr`](https://github.com/privadosIA/triposr) | C# port of [TripoSR](https://github.com/VAST-AI-Research/TripoSR): one picture in, one 3D model (`.glb`) out. | Apache-2.0 |
| [`stable-fast-3d`](https://github.com/privadosIA/stable-fast-3d) | C# port of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d): one picture in, one textured 3D model (`.glb`) out. | **Stability AI Community License — not Apache-2.0**: free for research and non-commercial use and under US$1M of yearly revenue (register with Stability AI), a commercial licence above |

The links are to the public copies of these repositories. The third-party tools, models and libraries they build on, with their licenses, are listed in [THIRD_PARTY.md](https://github.com/privadosIA/veritmo/blob/main/THIRD_PARTY.md).

## License

Apache 2.0 (see [LICENSE](LICENSE)). This is a port of work by other people; see the origin and licences of the original and of the weights in the sections above and, for the whole set, in the Veritmo `dotnet/THIRD_PARTY.md`.

This licence covers the code of this repository. The model weights are downloaded from their publishers and are under their own licences: MuQ-MuLan is CC-BY-NC-4.0 (non-commercial); for the others, see their model cards.

## Maintenance

Maintained by one person in their free time: issues and pull requests are welcome, with no promised response time. See [CONTRIBUTING.md](CONTRIBUTING.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
