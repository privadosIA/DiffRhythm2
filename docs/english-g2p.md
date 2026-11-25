# English g2p and lyrics tokenization

C# port of the front end that turns lyrics into the token ids the DiT takes as text (`src/DiffRhythm2.Net/Text`). Python reference: `veritmo/diffrhythm2:1` (upstream commit 7804f82: `inference.parse_lyrics`, `g2p/g2p_generation.chn_eng_g2p`, `g2p/g2p/english.py`, `phonemizer` 3.4.0, espeak-ng 1.51). **English only**: a line with Chinese characters is rejected with an explicit error (the Mandarin stack - jieba, pypinyin, a polyphone model - is a separate, parked change).

## Usage

```csharp
using DiffRhythm2.Text;

var tokenizer = new LyricsTokenizer(new EnglishG2p(EspeakEngine.Shared));     // libespeak-ng found on the machine
long[] ids = tokenizer.PrepareAndTokenize("[verse]\nHello world tonight\n[chorus]\nShine bright like the stars");
using var tokens = LyricsTokenizer.ToTensor(ids);                             // int64 [L], what BlockSampler.Sample takes as its text
```

- `LyricsTokenizer.Prepare` is the service's tag normalization (known tags lower-cased, unknown bracketed lines dropped, `[start]` added, error if there is nothing to sing); `Tokenize` is `parse_lyrics` (tag id + `[stop]`, phoneme ids + `[stop]`, `[start] [stop]` first when missing); `PrepareAndTokenize` is both, with errors reported against the line numbers of the text the user wrote.
- `EnglishG2p.Convert(line)` gives the phoneme string and the vocabulary indices of one line; `Encode` adds one (what the sampler sees). Layers can be used on their own: `EnglishNormalizer` (abbreviations, numbers), `EspeakPhonemizer` (cleaning, punctuation, tidying over any `IEspeakEngine`), `EnglishG2p.SpecialMap`, `PhonemeVocabulary`.
- Lines the reference refuses are refused here too: a line with no ASCII letter and no Chinese character (`2024`, `?!`, `...`, `1 2 3 4`) raises `NoSungTextException`, so a count-in such as `1 2 3 4` stops the whole song, as it does in the service. Numbers above 36 digits raise `NumberOutOfRangeException`.

## espeak-ng

The phonemes come from `libespeak-ng` through P/Invoke (`espeak_Initialize`, `espeak_SetVoiceByName("en-us")`, `espeak_TextToPhonemes` in IPA mode), exactly the calls `phonemizer` makes. One engine per process, calls serialized (the library has global state). The library is **not bundled** (espeak-ng is GPL-3): it is found by `DIFFRHYTHM2_ESPEAK_LIB` / `DIFFRHYTHM2_ESPEAK_DATA`, else by the platform's standard names (`libespeak-ng.so.1`) and its own data path, and a missing library fails with the list of what was tried.

**Version matters.** The reference runs espeak-ng **1.51**; pronunciations can differ in other versions. `tools/fetch_espeak_reference.sh <dir>` copies the exact Debian build out of the reference image (the library, `libpcaudio`, `libsonic` and the data), which runs on the host with no install:

```bash
tools/fetch_espeak_reference.sh ~/Proyectos/veritmo/models/music/diffrhythm2/espeak-1.51
E=~/Proyectos/veritmo/models/music/diffrhythm2/espeak-1.51
DIFFRHYTHM2_ESPEAK_LIB=$E/libespeak-ng.so.1.1.51 DIFFRHYTHM2_ESPEAK_DATA=$E/espeak-ng-data LD_LIBRARY_PATH=$E dotnet test -c Release
```

The distribution package (Arch: 1.52.0) was not measured against the reference; if it is installed, `Corpus_DifferencesFromTheReference_AreReported` lists the lines that differ, and the gating comparison is skipped with the version in its reason.

## What was learned from the reference

- With no Chinese, every line is a single English segment: non-letters attach to their neighbours, so the multi-segment logic of the original collapses.
- `inflect` is not a declared dependency of the image: the module imported is the copy vendored in setuptools 79.0.1. Its output has commas between number groups (`one thousand, two hundred thirty-four`), `and` only before a final single word, `and` kept in ordinals (`112th` is `one hundred and twelfth`) and a 36-digit limit; `NumberWords` reproduces it by following its regex passes and is checked against a table of 20,820 cardinals, 1,999 years, 1,900 ordinals and 529 fractions.
- The phonemizer's character classes are Python's (`\w` excludes combining marks and includes `²` and `½`), so `PyText` spells them out instead of using .NET's.
- The apostrophe is a punctuation mark: `Don't` becomes `d|ɑː|n|'|t|iː`.
- Spoken numbers add spaces and commas that espeak turns into pauses, so normalization keeps the reference's exact spacing.

## Verification

Fixtures in `tests/fixtures/g2p` (`tools/dump_g2p_reference.py`, corpus in `tools/g2p_corpus.txt`): per line the normalized text, the cleaned text, the phonemizer output, the `special_map` output, the phoneme string and the ids, plus the recorded raw espeak answers. The recorded answers let the whole post-processing be tested with a replay engine, so everything except the final comparison with the real library runs on any machine. With espeak-ng 1.51 all 103 corpus lines are identical to the reference at every layer, and the golden-run lyrics reproduce the 55 sampler tokens of `tests/fixtures/dit/inputs.safetensors`.
