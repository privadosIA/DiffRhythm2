# MuQ-MuLan audio tower (recognizing and comparing songs)

C# port of the audio side of MuQ-MuLan: it turns a piece of music into the same 512-dimensional space as the style text of [mulan-text.md](mulan-text.md), so a song can be compared with text ("This is a jazz song.") or with another song. Nothing runs in Python, at run time or at conversion time; the Python scripts in `tools/` are only the oracle the fixtures come from. Python reference: `veritmo/diffrhythm2:1` (`muq` 0.1.0, transformers 4.47.1, torchaudio 2.7.0).

> **License.** The MuQ-MuLan weights are CC-BY-NC-4.0: recognition with them is non-commercial too. (LAION-CLAP is Apache-2.0.)

This page grows with the change `add-muq-mulan-audio`; the sections below are the ones that exist so far.

## Input: WAV files and clips

`WavReader.Read(path)` (namespace `DiffRhythm2.AudioTower`) reads 16, 24 and 32-bit PCM and 32-bit float WAV files (also WAVE_FORMAT_EXTENSIBLE), mixing channels to mono by their mean. Integer samples are divided by the full range of their type. **The sample rate must be 24,000 Hz**: any other rate is refused, before a single sample is decoded, with a message that names both rates and the command that fixes it, and nothing is resampled (the reference leaves resampling to the caller, and different resamplers give different embeddings):

```bash
ffmpeg -i song.mp3 -ac 1 -ar 24000 song.wav
```

`AudioClips.Split(signal)` cuts the signal as the reference does (`_get_all_clips`):

- consecutive clips of exactly 240,000 samples (10 s) from the start;
- a remainder becomes one more clip: the remainder followed by the first `240,000 - remainder` samples of the signal. A 25 s song gives three clips, the last one being seconds 20 to 25 followed by seconds 0 to 5; an exact multiple of 10 s has no extra clip;
- **a signal shorter than 5 s gives one clip of twice its length**, not a 10 s clip: the reference's Python slice stops at the end of the signal, and that is reproduced (a 4 s signal gives an 8 s clip, measured on the reference). Between 5 and 10 s the clip is exactly 10 s;
- a signal of fewer than 513 samples is refused: its clip would be shorter than the 1,025 samples the spectrogram's reflection padding needs, and the reference cannot process it either.

## Front end: log-mel

`MelFrontEnd` produces the encoder's input from a clip, in float32, with the operations and the order torchaudio uses:

| step | what |
|---|---|
| power spectrogram | `torch.stft`, 2048-point periodic Hann window, hop 240, centered with reflection padding; `abs().pow(2)` |
| mel bands | 128 HTK bands from 0 to 12 kHz, no band normalization; the filter bank `[1025, 128]` is built exactly as `melscale_fbanks` does (`FilterBank()`) |
| decibels | `10 * log10(max(x, 1e-10))`, no dynamic-range limit |
| last frame | dropped (a clip of n samples gives `n / 240` frames: 1000 for 10 s) |
| normalization | `(x - 6.768444971712967) / 18.417922652295623`: the mean and standard deviation of the MuQ-large training set, the `stat` entry of its configuration (they are not in the weights, so they are constants of the code) |

```csharp
using DiffRhythm2.AudioTower;

var wav = WavReader.Read("song.wav");                       // mono floats at 24 kHz
var clips = AudioClips.Split(wav.Samples);                  // 10 s clips
using var front = new MelFrontEnd();                        // on CPU; pass a TorchSharp Device for another
using var features = front.Features(clips[0]);              // [1, 128, 1000], float32
```

## The MuQ-large encoder

`MuLanAudioTower` (namespace `DiffRhythm2.AudioTower`) holds the audio side of MuQ-MuLan under the tensor names of the checkpoint (`audio.model.model.conv.*`, `audio.model.model.conformer.*`, `audio.proj.*`, `audio_to_latents.*`), so loading needs no mapping. It runs in evaluation mode, fp32, without gradients.

```
normalized spectrogram [1, 128, 1000]
  -> subsampling: 2 residual blocks (3x3 convs, BatchNorm, ReLU, strided shortcut), channels x frequency flattened, Linear 16384 -> 1024   [1, 250, 1024]
  -> 12 Conformer layers, 1024 wide, 16 heads of 64, rotary base 10,000
  -> LayerNorm                                                                                                                        [1, 250, 1024]
```

One Conformer layer is: `x + 0.5 * ffn1(LN(x))`, then attention, then the convolution module, then `x + 0.5 * ffn2(LN(x))`, then a LayerNorm. The attention module adds `rotary` to its **input** before `linear_q` and `linear_k` (not to the projected queries and keys, as most implementations do), `linear_v` sees the unrotated input, scores are divided by 8, plain softmax, no mask. The convolution module is LayerNorm, pointwise conv to 2048, GLU over the channels, depthwise conv (kernel 31), BatchNorm (stored statistics), swish, pointwise conv. All of this was read from the source of transformers 4.47.1 in the reference image and checked tensor by tensor.

**What is loaded and what is not.** The converted file has 531 tensors; the model reads 459. The other 72 are never evaluated by the reference and are reported by name when loading (`MuLanAudioLoadReport.IgnoredKeys`): the 64 of MuQ-large's random-projection quantizer (`rvq`, a training target), its classification token and its prediction layer to 8,192 codebook logits (3), the 3 tensors of the convolutional positional embedding (`pos_conv_embed`: HuggingFace's `Wav2Vec2ConformerEncoder` constructs it but its `forward` never calls it, positions come only from the rotary embedding) and MuLan's 2 layer-aggregator tensors (not used when the last layer is taken). Batch-norm statistics and the rotary frequencies (`inv_freq`) are buffers and are loaded. Loading is strict: a name or shape that differs aborts before anything is assigned and lists every discrepancy.

**Why the weights come from the MuLan pickle.** `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")` is called first, but MuLan's checkpoint then overwrites it, and MuLan fine-tuned MuQ-large: only 69 of its 525 tensors are still equal to that repository's. The converted file is therefore made from MuLan's own `pytorch_model.bin` (see [weights-tools.md](weights-tools.md)).

### Verification

Fixtures in `tests/fixtures/mulan_audio` (`tools/dump_mulan_audio_reference.py`, through the real `MuQMuLan`, fp32 on CPU): three deterministic 24 kHz signals (4 s, 25 s, 10 s of noise), the reference's own wrapped clips, the filter bank, and the decibel and normalized spectrograms of the first clip of each signal. The reference is bit-for-bit reproducible across processes (two runs in separate processes differ by 0.0).

| comparison | measured (relative to the largest magnitude) | gate |
|---|---|---|
| filter bank `[1025, 128]` | 0 (identical) | 1e-6 |
| decibel spectrogram, 3 signals | 0 (identical) | 1e-6 |
| normalized features, 3 signals | 0 (identical) | 1e-6 |
| wrapped clips (s4, s25) | identical | identical |

The front end is bit-identical because the port runs the same libtorch kernels in the same order; the gate leaves room for a few float32 units on another CPU's FFT or matrix multiply code.

The encoder's fixtures are the stage outputs of the first clip of the 25 s signal; each stage is fed the reference's previous stage (the Conformer layers 6 and 12 start from the reference's layers 5 and 11), so an error is attributed to the module that introduced it:

| comparison | measured (relative to the largest magnitude of the stage) | gate |
|---|---|---|
| subsampling `[1, 250, 1024]` | 0 (identical) | 1e-5 |
| Conformer layer 1 / 6 / 12 | 2.4e-7 / 2.2e-7 / 2.2e-7 | 1e-5 |
| final layer norm | 1.9e-7 | 1e-5 |
| all 12 layers chained from the spectrogram | 3.7e-7 (cosine 1.000000000) | 1e-5 |

The error does not grow through the layers. The strict loader reads 459 tensors and ignores exactly 72: 64 of the `rvq` quantizer, 3 of `pos_conv_embed`, 2 of the layer aggregator, 2 of the prediction layer and the classification token (64 + 3 + 2 + 2 + 1).

The number of steps for a clip of `n` frames is `ceil(ceil(n / 2) / 2)`: 250 for 10 s (1000 frames), 200 for the 8 s clip a 4 s signal makes (800 frames).

## The embedding of a clip and of a song

`MuLanAudioEncoder` (`LoadDirectory(dir, device)` reads `mulan_audio.safetensors`; it is created by `diffrhythm2 convert --audio`) turns audio into the 512 values MuQ-MuLan shares with text:

```
clip -> front end -> MuQ-large [1, 250, 1024] -> Linear 1024 -> 768 -> mean over the 250 steps -> Linear 768 -> 512 -> L2 normalization
```

MuLan's own audio transformer has **no layers** in the published configuration (`tf_depth` is 0), so nothing sits between the projection and the mean. The embedding of a clip has length 1.

```csharp
using DiffRhythm2.AudioTower;
using TorchSharp;

using var encoder = MuLanAudioEncoder.LoadDirectory("/path/to/models", torch.CPU);
var wav = WavReader.Read("song.wav");                                   // 24 kHz, see above
AudioEmbedding song = encoder.Embed(wav);                               // song.Values: 512 floats; song.Clips; song.Seconds
```

A song is cut into 10 s clips, **each clip is embedded alone** (memory does not grow with the length of the song) and the embeddings are averaged. As in the reference, **the mean is not normalized again**: a song of several different clips has an embedding shorter than 1 (the 25 s test signal: 0.997341), and a single clip has exactly 1. Scores from different songs therefore compare as dot products, not as cosines; divide by the lengths if you want cosines.

## Comparing: similarity and ranking

`TagRanker.Dot(a, b)` is the dot product of two embeddings, the reference's `calc_similarity`: audio with audio, or audio with text. `TagRanker.Rank(audio.Values, texts, styleEncoder, top)` embeds each text **alone** with the text tower (`StyleEncoder`, see [mulan-text.md](mulan-text.md)), and returns the texts with their scores, highest first (ties keep the order of the list), optionally the first `top`. A text that cannot be embedded (over 512 tokens) fails the ranking and is named.

```csharp
using var style = StyleEncoder.LoadDirectory("/path/to/models", torch.CPU);
foreach (var r in TagRanker.Rank(song.Values, new[] { "This is a jazz song.", "This is a rock song." }, style, top: 3))
    Console.WriteLine($"{r.Text}: {r.Score:F3}");
```

### Why the numbers can differ from the Python study

`muq_genres.py` of the song-recognition study embeds **all the tags of a category as one padded batch** (`m(texts=texts)`). The text tower averages over every position including the padding, so in a batch each text's embedding depends on how much padding its neighbours impose. Embedding each tag alone (what this library does, and what the `style-text-embedding` spec requires) is the cleaner measure, but it is not the same number. Measured on the seven prompts of the fixture (one at a time against the same seven as one padded batch):

| prompt | cosine between the two embeddings |
|---|---|
| "Pop, Piano, Female vocals, Happy, Bright" | 0.942 |
| "Rock, Electric guitar, Male vocals, Energetic" | 0.885 |
| "Lo-fi, Chill" | 0.894 |
| "música alegre con guitarra y voz femenina" | 0.864 |
| "流行音乐, 钢琴, female vocals" | 0.895 |
| the long "Electronic, Synthwave, ..." prompt (the longest: no padding) | 1.000 |
| "" (empty) | 0.960 |

The largest change of a similarity score between the two ways was 0.10. Rankings can therefore change when the tags of a category have very different lengths. If you need to reproduce the old numbers exactly, embed the tags as a batch in Python; for new comparisons the one-by-one numbers are the reliable ones.

## From a terminal: `embed` and `classify`

Both commands need the audio tower (`diffrhythm2 convert --model DIR --audio`, or `install --audio`) and a **24 kHz WAV**; `classify` and `embed --text` also need the text tower. The audio is checked before any model is loaded: a file of another rate fails at once with the ffmpeg command that fixes it.

```bash
ffmpeg -i song.mp3 -ac 1 -ar 24000 song.wav
diffrhythm2 embed --model ~/models/diffrhythm2 --audio song.wav --out song.json
diffrhythm2 embed --model ~/models/diffrhythm2 --text "Pop, Piano, Female vocals"
diffrhythm2 classify --model ~/models/diffrhythm2 --audio song.wav --vocab docs/mulan-vocabulary-example.json --top 3
```

`embed` prints one JSON object (or writes it with `--out`, atomically): `{"kind":"audio","dim":512,"clips":3,"seconds":25,"length":0.955422,"embedding":[...]}`; for a text, `kind` is `text` and there are no `clips`, `seconds` or `length` (a text embedding always has length 1). Progress (`clip 2/3`) goes to the error stream.

`classify` ranks a **vocabulary** of tags for the audio. The vocabulary is a JSON object of categories; each category is a list of tags, or an object with a `template` (a text with `{}`, replaced by the tag) and the `tags`. A list uses the tag itself as the text. [`docs/mulan-vocabulary-example.json`](mulan-vocabulary-example.json) is the tag list of Veritmo with the prompt templates `muq_genres.py` uses:

```json
{ "genre": { "template": "This is a {} song.", "tags": ["pop", "rock", "jazz"] },
  "mood":  ["happy", "sad", "calm"] }
```

The output is one line per category, the best `--top` tags (default 3) with their dot product with the audio, highest first; with `--json`, one object with `clips`, `seconds` and, per category, the `tag`, the `text` that was embedded and the `score`:

```
genre: pop 0.4981, r&b 0.3907, country 0.3666
mood: uplifting 0.1697, playful 0.1515, chill 0.1366
instrument: ukulele 0.3073, drums 0.3072, bass 0.2663
```

Errors: a wrong or missing option, an invalid vocabulary (naming the category), or a wrong `--top` end with exit code 2 before anything is read; an unreadable audio file, a rate other than 24 kHz, an audio of fewer than 513 samples or a missing model file (naming the command that creates it) end with exit code 1; Ctrl+C ends with 130. No partial output file is left.

### Comparing with LAION-CLAP and Whisper

The three analyses answer different questions, so compare them on the same songs by what each one outputs:

* **MuQ-MuLan (this) and LAION-CLAP** both put audio and text in one space and rank tags by similarity. Run `classify --json` with the same vocabulary you give CLAP and compare the rankings song by song. MuQ-MuLan is trained on music only and CLAP on general audio; the earlier study found CLAP's scores nearly flat between genres (0.5 to 0.67 for everything) on generated music. Scores are dot products (a multi-clip song has an embedding shorter than 1), so compare **rankings and gaps between tags**, not absolute values across models.
* **Whisper** transcribes speech and detects the language; it has no notion of genre or mood. It adds the lyrics and the language to what the other two say about style.
* **License.** MuQ-MuLan's weights are CC-BY-NC-4.0 (non-commercial); LAION-CLAP's are Apache-2.0. For a commercial product that difference alone may decide.

## Verification of the embedding

| comparison | measured | gate |
|---|---|---|
| projection 1024 to 768 (fed the reference's encoder output) | 1.2e-7 relative | 1e-5 |
| mean over time, projection to 512, L2 normalization (each fed the reference's previous stage) | 0 (identical) | 1e-5 |
| clip embeddings from raw samples (5 clips of 3 signals) | 1.2e-7 to 2.8e-7 relative, cosine 1.000000000 | 1e-5 |
| signal embeddings (4 s, 25 s, 10 s of noise) | 1.4e-7 to 1.7e-7 relative | 1e-5 |
| length of the embedding of the 25 s signal | 0.997341 (reference 0.997341) | 1e-5 |
| similarities audio to audio (3 x 3) and audio to the 7 prompts (3 x 7) | 1e-7 at most | 1e-6 absolute |

The ranking of the seven prompts for each test signal has the reference's order and scores.

## Performance (CPU, fp32, 20 threads)

Measured on the 25 s test signal (three clips) with `Timing_AndMemory_OfEmbeddingA25SecondSignal_AreMeasured`:

```bash
DIFFRHYTHM2_CKPT=~/models/diffrhythm2 dotnet test -c Release --filter "FullyQualifiedName~Timing_AndMemory" --logger "console;verbosity=detailed"
```

| what | measured |
|---|---|
| embedding 25 s of audio (3 clips) | 1.4 s, the same on the second call (the Python reference takes 1.3 s for the same 25 s on this machine: the speed is the same) |
| the front end of one clip | 3 ms |
| memory while embedding | +440 MiB over the loaded model (the model itself is 1.3 GB in fp32) |

The test process held the audio tower and the text tower together (3.3 GiB resident).

### From the terminal, end to end

Whole commands, including starting the process and loading the models (a 25 s song, 20 threads):

| command | time |
|---|---|
| `embed --audio` (25 s) | 2.4 s |
| `embed --text` | 1.2 s |
| `classify` (25 s, 106 tags in 7 categories) | 4.9 s |

## Checked on real music

The tests above use synthetic signals. As a last check the seven songs of the owner's recognition study that still exist (mp3 made by the project, 10 to 25 s each, converted with `ffmpeg -ac 1 -ar 24000`) were embedded by the Python reference and by `diffrhythm2 embed`, and their 106-tag vocabulary ranked both ways:

| comparison | result |
|---|---|
| audio embeddings, C# against Python, 7 songs (1 to 3 clips each) | 1.1e-7 to 2.3e-7 relative, cosine 1.000000000, same length |
| `classify` top 3 against Python embedding the tags one at a time | identical in 49 of 49 (song, category) pairs, scores within the 4 printed decimals |
| `classify` top 3 against the original `muq_genres.py` (tags as one padded batch) | identical in 31 of 49 pairs; the same best tag in 45 of 49 |

The first two rows are the port against the reference. The third is the effect described under "Why the numbers can differ": by category, the genre top 3 was identical in 6 of 7 songs and the era in 7 of 7, but the production top 3 in only 1 of 7. In the one genre difference (the tavern song) the same three tags come back in another order: country, folk, pop from the old script and folk, country, pop from this tool. If you compare a ranking from this tool with a number from the old script, expect small reorderings of close tags, not different songs.

