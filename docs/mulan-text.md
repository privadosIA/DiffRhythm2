# MuQ-MuLan text tower (style embedding)

C# port of the text side of MuQ-MuLan, which turns a free-text style prompt ("Pop, Piano, Female vocals, Happy, Bright") into the 512-dimensional style embedding the DiT is conditioned on (`src/DiffRhythm2.Net/StyleText`). Python reference: `veritmo/diffrhythm2:1` (`muq` package, transformers 4.47.1, tokenizers 0.21.4). The audio tower is not ported.

## Usage

```csharp
using DiffRhythm2.StyleText;
using TorchSharp;

using var style = StyleEncoder.LoadDirectory("/path/to/checkpoint", torch.CPU);   // mulan_text.safetensors + mulan_text_tokenizer.json, strict
using var embedding = style.Embed("Pop, Piano, Female vocals, Happy, Bright");    // [1, 512], float32, unit length
// BlockSampler.Sample takes it as its style; cast it first if the DiT runs in half precision (what the service does): embedding.to(ScalarType.Float16)
```

- One text per call. `Embed(IReadOnlyList<string>)` accepts a list of exactly one text and refuses more: the reference averages over padded positions, so a batch would not reproduce what the service computes.
- A text of more than 512 tokens (`<s>` and `</s>` included) is refused naming the length and the limit (the encoder has 514 position rows and the first two are the padding offset); the reference fails there.
- The tokenizer can be used alone: `XlmrTokenizer.Load(path).Encode(text)` returns the ids, `Normalize(text)` the normalizer's output.

## Weights

`pytorch_model.bin` of MuQ-MuLan-large is a 2.5 GB pickle with the audio tower too. `diffrhythm2 convert` (see [weights-tools.md](weights-tools.md); `tools/convert_mulan_text.py` is the Python oracle it is compared with, byte for byte) writes `mulan_text.safetensors` with the 273 text-tower tensors (329M parameters, 1.3 GB fp32) under their checkpoint names minus the leading `mulan.`, and copies `tokenizer.json` as `mulan_text_tokenizer.json`, both into the checkpoint directory. The loader is strict: any difference in names or shapes aborts before reading and lists everything; only the two unused pooler tensors of the encoder are ignored (and reported). The 199 encoder tensors are bit-identical to `xlm-roberta-base`'s: MuLan does not fine-tune the encoder.

## The tower

```
text -> XLM-R tokenizer -> <s> ... </s>
     -> XLM-RoBERTa-base (12 post-norm layers, 768, 12 heads, exact GELU; positions 2..n+1)
     -> MuLan transformer (8 pre-norm layers, 8 heads x 64; q and k L2-normalized and scaled by learned vectors, similarities x 8; GEGLU feed-forward 768-2048-768)
     -> mean over ALL positions (markers included) -> Linear 768 -> 512 -> L2 normalization
```

The service runs this model in **fp32** and only casts the result to half, so there is a single precision to verify.

## The tokenizer

`AutoTokenizer` gives `XLMRobertaTokenizerFast`, built from `tokenizer.json`: the sentencepiece character map (`Precompiled`), whitespace splitting, the `▁` marker, a **Unigram** model of 250,002 pieces, `<s> A </s>`. It is not what the `sentencepiece.bpe.model` file next to it describes, which is why it is implemented here from `tokenizer.json` and not with a library that reads the `.model`.

- **Normalization** follows the HF Rust normalizer: grapheme clusters shorter than 6 UTF-8 bytes are looked up whole (the **first**, shortest, prefix found in the double-array trie is used), otherwise character by character. 28 strings of the fuzz corpus tell this rule apart from "longest match", so the corpus does pin it down.
- **Unigram**: Viterbi over the word, candidates from the shortest piece to the longest replacing a path only if strictly better; a character no piece covers is an unknown scored `min score - 10`, and consecutive unknowns are fused into one `<unk>`.
- **Special tokens** (`<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`) are recognized in the raw text before normalization.
- U+2581 in the input is treated as a space by the character map, and the word marker splits words before any marker.

## Verification

Fixtures in `tests/fixtures/mulan_text` (`tools/dump_mulan_text_reference.py`): the normalization of **every code point of planes 0 to 2** (4,837 change), 4,062 multi-character strings (combining sequences, jamo, emoji sequences, every kind of space, 4,000 random strings from a pool of interesting code points), the ids of 64 texts in many scripts, and the stage outputs of seven prompts through the real `MuQMuLan`. The reference is bit-for-bit reproducible across processes.

| comparison | measured (relative to the largest magnitude) | gate |
|---|---|---|
| normalization of every code point, 4,062 strings, 64 texts' ids | identical | identical |
| embeddings, encoder layer 0, encoder output | 4.9e-8, 1.3e-7, 3.2e-7 | 1e-5 |
| transformer layer 0, transformer output | 2.0e-7, 4.6e-7 | 1e-5 |
| final embeddings from text (7 prompts) | <= 8.7e-7, cosine 1.000000000 | 1e-5, cosine >= 1 - 1e-7 |
| embedding of a 512-position text | 1.4e-7 | 1e-5 |

Each stage is fed the reference's previous stage, so an error is attributed to the stage that introduced it.

## Performance (CPU, fp32, idle machine)

Loading the tower takes about 1 s and adds about 1.2 GiB to the process; one 13-token embedding takes about 16 ms.
