# Weights tools: getting the model directory without Python

`diffrhythm2 install` builds the six-file model directory the pipeline needs, starting from nothing but a network connection. Everything is C#: the files are downloaded from pinned HuggingFace revisions and verified, and the Python pickles (`torch.save` files) are converted to safetensors by a reader that never executes anything contained in them. The Python scripts in `tools/` remain only as the test oracle: the C# outputs are byte for byte the files they write.

## The short version

```bash
diffrhythm2 install --model ~/models/diffrhythm2 --accept-licenses
diffrhythm2 check   --model ~/models/diffrhythm2
```

`install` fetches what is missing (about 7.7 GB the first time), converts it and checks the result. Run it again at any time: it does nothing when everything is in place. Interrupted downloads resume where they stopped.

## Licenses

Three repositories are involved. You must accept their licenses with `--accept-licenses`; without it nothing is downloaded (exit code 2) and the licenses are printed.

| repository | license | what we take |
|---|---|---|
| [ASLP-lab/DiffRhythm2](https://huggingface.co/ASLP-lab/DiffRhythm2) | Apache-2.0 | `config.json`, `model.safetensors`, `decoder.json`, `decoder.bin` |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | MIT | `tokenizer.json` |
| [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) | **CC-BY-NC-4.0** | `pytorch_model.bin` (only its text tower is used) |

MuQ-MuLan is **non-commercial**. A product that includes the style encoder inherits that restriction. The library enforces the acknowledgement too (`ModelDownloader.EnsureAsync` throws `LicenseNotAcceptedException` before any request), so a program built on it cannot skip it by accident.

## Commands

| command | what it does |
|---|---|
| `fetch --model DIR --accept-licenses` | downloads the six files of the manifest. The three that the pipeline reads go in `DIR`; the three checkpoints to convert go in `DIR/sources` |
| `convert --model DIR [--from DIR] [--force] [--delete-sources] [--audio]` | writes `decoder.safetensors`, `mulan_text.safetensors` and `mulan_text_tokenizer.json`; with `--audio` also `mulan_audio.safetensors` (see below) |
| `install --model DIR --accept-licenses [--audio]` | fetches only what is missing (a checkpoint is skipped if its converted output exists or the checkpoint is already somewhere `convert` looks), converts, and checks. Ends with `ready` or `missing: ...`. With `--audio` it also converts the audio tower and the check includes it |

Output lines (stable, meant to be parsed): `license: REPO is LICENSE`, `step: REPO/FILE` (download) or `step: OUTPUT` (conversion), `present: FILE`, indented progress lines, `done` / `ready`, `missing: A, B`. Errors go to standard error. Exit codes: 0 done, 1 failure, 2 wrong usage or licenses not accepted, 130 cancelled (Ctrl+C).

### Downloads

* Revisions are fixed commits, and every file has a recorded size and SHA-256 (`ModelManifest`). A file whose hash differs is deleted and the error shows the expected and the actual hash.
* A download goes to `NAME.part` and gets its name only after verification. If `NAME.part` exists, the next run asks the server for the rest (`Range`) and still verifies the whole file; a server that ignores ranges makes it start over.
* Transient failures (connection errors, stalls of 60 s, HTTP 5xx and 429) are retried up to 5 times with delays of 1, 2, 4 and 8 s. A refusal (404, 401, 403) fails at once with the address and the status.
* Behind a proxy the usual `HTTPS_PROXY` / `HTTP_PROXY` variables are honored. `HF_ENDPOINT` points to a mirror of huggingface.co (the path layout `/{repo}/resolve/{revision}/{file}` must be the same).
* No token is sent: none of these repositories is gated.

### Conversion

| output | made from | how |
|---|---|---|
| `decoder.safetensors` | `decoder.bin` (`{"generator": state_dict}`) | 783 float32 tensors, copied bit for bit; each weight-norm pair is renamed `X.weight_g/weight_v` to `X.parametrizations.weight.original0/original1` (nothing is folded) |
| `mulan_text.safetensors` | MuQ-MuLan `pytorch_model.bin` | the 273 text-tower tensors (`mulan.text.*`, `mulan.text_to_latents.*`), written without the leading `mulan.`; the audio tower is dropped |
| `mulan_text_tokenizer.json` | XLM-RoBERTa `tokenizer.json` | copied unchanged |
| `mulan_audio.safetensors` (only with `--audio`) | MuQ-MuLan `pytorch_model.bin` | the 531 audio-tower tensors (`mulan.audio.*`, `mulan.audio_to_latents.*`), written without the leading `mulan.`: 525 of MuQ-large, 4 of the audio projection and layer aggregator, 2 of the projection to the latent space. 513 are float32 and 18 are the int64 `num_batches_tracked` counters of the batch normalizations; the file is 1.3 GB |

### The audio tower (`--audio`)

Making songs needs only the text tower, so `convert` and `install` do **not** produce `mulan_audio.safetensors` unless you ask with `--audio`; it is only used to recognize and compare songs (see [mulan-audio.md](mulan-audio.md)). Asking for it **downloads nothing more**: the audio tower is in the same `pytorch_model.bin` the style encoder comes from (2.5 GB, already in the manifest), and `install --audio` fetches that checkpoint if it is not in the directory yet. The conversion takes a few seconds and the working set grows by about 13 MiB.

The weights come from that pickle on purpose, not from the `OpenMuQ/MuQ-large-msd-iter` repository: MuLan fine-tuned MuQ-large, and only 69 of the 525 MuQ-large tensors in the MuLan checkpoint are still equal to that repository's (the frozen random quantizer and the counters), so reading the standalone MuQ-large file would give different embeddings. `tools/convert_mulan_audio.py` is the Python oracle: the C# output is byte for byte its file.

The files list their tensors **by data type first (int64 before float32) and by name within a type**, which is how the Python `safetensors` library orders them; with only float32 tensors, as in the other two files, that is the same as sorting by name.

Where `convert` looks for each checkpoint: the `--from` directory (if given), the model directory, `DIR/sources`, then the HuggingFace cache layout (`models--ORG--NAME/snapshots/REVISION/FILE`) under those. This means a models folder that already holds checkpoints downloaded by other tools needs no moving. When nothing is found, the error lists every place that was searched.

Safety and repeatability:

* Outputs are written to `NAME.tmp`, verified against the source (every tensor: name, type, shape and bytes) and only then renamed. A failure leaves no file under the final name.
* A valid output that is already there is left alone (`up to date`) unless `--force`. The output is deterministic: converting twice gives the same bytes.
* The source checkpoints are **kept**. `--delete-sources` removes them once every output is written and verified (for HuggingFace cache layouts only the file in the snapshot, which is a link, is removed).
* Memory does not depend on the checkpoint size: tensors are streamed one at a time (the working set grew by about 14 MiB converting 3.1 GB of checkpoints).

## Using it from your own code

```csharp
using DiffRhythm2.Weights;

var downloader = new ModelDownloader();
await downloader.EnsureAsync("/models/diffrhythm2", licensesAccepted: true,
                             only: ModelManifest.Needed("/models/diffrhythm2"),
                             progress: new Progress<DownloadEvent>(e => Console.WriteLine(e)));

WeightConverter.Convert("/models/diffrhythm2", new ConversionOptions(), step: name => Console.WriteLine(name));
```

The pieces are public and usable on their own: `TorchArchive` (list and stream the tensors of a `torch.save` file), `SafeTensorsStreamWriter`, `SafeTensorsHeader`.

### What the checkpoint reader accepts

`TorchArchive` reads the zip format of `torch.save` (PyTorch 1.6 and later) and interprets its pickle with a closed allow-list: an ordered dictionary, the tensor rebuild function and the storage types of float32/16, bfloat16, float64 and the integer and boolean types. A pickle that asks for anything else (`os.system`, a class, a function) is refused by name before any data is read, and nothing is ever run; a pickle opcode outside the supported set is refused naming the opcode and its offset. Tensors must be contiguous (shared storages and storage offsets are fine). The legacy non-zip format, a damaged zip and a tensor whose data is cut short are reported with the tensor's name.

## Development

`tools/dump_torch_archive_fixtures.py` makes the small archives under `tests/fixtures/torch_archive/` with the real `torch.save`. `tools/convert_decoder.py` and `tools/convert_mulan_text.py` stay as the oracle: `RealWeightsTests` converts the real checkpoints and compares the outputs with the files those tools wrote (sha256), and is skipped without `DIFFRHYTHM2_CKPT`.
