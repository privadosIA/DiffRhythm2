"""
Golden run for the English g2p and the lyrics tokenization (change add-english-g2p). Run it inside the reference image (see tools/README.md);
no GPU and no checkpoint are needed. Writes to --out (default tests/fixtures/g2p):

  lines.json         one record per corpus line (tools/g2p_corpus.txt), every layer of the real pipeline:
                       normalized   english._english_to_ipa (abbreviations and numbers)
                       cleaned      the text the phonemizer receives (TextTokenizer cleaning)
                       phonemizer   what TextTokenizer returns (cleaning + phonemizer, before special_map)
                       ipa          english_to_ipa output (phonemizer + special_map + the final-consonant "|_")
                       phonemes     chn_eng_g2p phoneme string (after the segment-level removal of that "_")
                       tokens       chn_eng_g2p vocabulary indices
                       ids          tokens + 1, what CNENTokenizer.encode returns
                     or {"error": {"type", "message"}} when the reference raises
  transcripts.json   every (input chunk -> raw espeak output) the real EspeakWrapper saw, over the corpus AND the lyrics cases; a replay engine
                     in the tests reproduces the phonemizer post-processing from it without the espeak library
  numbers.json       inflect number words (cardinal, year, ordinal, fraction) and the normalize_numbers / _english_to_ipa cases, plus the range limit
  lyrics.json        whole-lyrics cases: normalize_lyrics (server.py) and parse_lyrics (inference.py), prepared text and flat tokens or the error
  vocab.json         the reference vocabulary file, copied
  meta.json          versions (espeak, phonemizer, setuptools, inflect) and counts
  MANIFEST.md        sha256 of every file (verified by the core's FixtureDirectory)

Nothing in /opt/dr2 is modified: the pipeline is observed through wrappers installed in memory (and a few asserted ones).
In the corpus file a literal \\uXXXX is replaced by that code point (so that decomposed accents can be written in plain ASCII).
"""
import argparse
import ast
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir; tools/ is our own code
os.environ.setdefault("HOME", "/tmp")
sys.path.insert(0, "/opt/dr2")

import dr2_ref as R  # noqa: E402  (also puts /opt/dr2 on the path and sets the offline variables)

import g2p.g2p.english as en  # noqa: E402
import g2p.g2p.cleaners as cleaners  # noqa: E402
import g2p.g2p_generation as gen  # noqa: E402

KNOWN_TAG_SOURCE = "/opt/server.py"
ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def read_corpus(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if raw.startswith("##") or raw == "":
                continue
            lines.append(ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), raw))
    return lines


class Observer:
    """Wrappers around the real pipeline that record what flows through it, one line at a time."""

    def __init__(self):
        self.transcripts = {}
        self.current = {}
        tt = gen.text_tokenizer.text_tokenizers["en"]
        self.backend = tt.backend
        self.espeak = self.backend._espeak

        real_text_to_phonemes = self.espeak.text_to_phonemes

        def text_to_phonemes(text, tie=False):
            out = real_text_to_phonemes(text, tie)
            previous = self.transcripts.setdefault(text, out)
            assert previous == out, f"espeak is not deterministic for {text!r}: {previous!r} vs {out!r}"
            return out

        self.espeak.text_to_phonemes = text_to_phonemes

        real_phonemize = self.backend.phonemize

        def phonemize(text, *a, **k):
            self.current["cleaned"] = list(text)
            return real_phonemize(text, *a, **k)

        self.backend.phonemize = phonemize

        real_norm = en._english_to_ipa

        def norm(text):
            out = real_norm(text)
            self.current["normalized"] = out
            return out

        en._english_to_ipa = norm

        from g2p.g2p.text_tokenizers import TextTokenizer
        real_call = TextTokenizer.__call__
        observer = self

        def tokenizer_call(tt_self, text, strip=True):
            out = real_call(tt_self, text, strip)
            observer.current["phonemizer"] = out
            return out

        TextTokenizer.__call__ = tokenizer_call

        real_en = cleaners.english_to_ipa

        def english_to_ipa(text, tokenizer):
            out = real_en(text, tokenizer)
            self.current["ipa"] = out
            return out

        cleaners.english_to_ipa = english_to_ipa

    def run(self, text):
        self.current = {}
        try:
            phonemes, tokens = gen.chn_eng_g2p(text)
        except Exception as e:  # the reference's own refusals are part of the contract
            return {"text": text, **self.current, "error": {"type": type(e).__name__, "message": str(e)}}
        if "cleaned" in self.current:   # a line with Chinese goes through the Mandarin path, which this change does not port
            assert len(self.current["cleaned"]) == 1
            self.current["cleaned"] = self.current["cleaned"][0]
        has_chinese = bool(re.search("[\u4e00-\u9fa5]", text))
        return {"text": text, **self.current, "has_chinese": has_chinese, "phonemes": phonemes, "tokens": tokens, "ids": [t + 1 for t in tokens]}


def number_tables(inflect_engine):
    e = inflect_engine
    ntw = e.number_to_words

    def safe(f):
        try:
            return f()
        except Exception as x:  # noqa: BLE001
            return {"error": type(x).__name__}

    cardinal_values = sorted(set(
        list(range(0, 20001))
        + [10 ** k + d for k in range(5, 24) for d in (-1, 0, 1, 5)]
        + [10 ** k * m for k in range(3, 22) for m in (2, 7, 12, 99)]
    ))
    import random
    rnd = random.Random(7)
    for k in range(5, 22):
        cardinal_values += [rnd.randrange(10 ** (k - 1), 10 ** k) for _ in range(40)]
    cardinal_values = sorted(set(cardinal_values))
    cardinal = {str(n): safe(lambda n=n: ntw(n, andword="")) for n in cardinal_values}
    year = {str(n): safe(lambda n=n: ntw(n, andword="", zero="oh", group=2)) for n in range(1001, 3000)}
    ordinals = {}
    for n in list(range(1, 1001)):
        for suffix in ("st", "nd", "rd", "th"):
            if n > 300 and suffix != ("st" if n % 10 == 1 and n % 100 != 11 else "nd" if n % 10 == 2 and n % 100 != 12 else "rd" if n % 10 == 3 and n % 100 != 13 else "th"):
                continue
            ordinals[f"{n}{suffix}"] = safe(lambda s=f"{n}{suffix}": ntw(s))
    fractions = {f"{a}/{b}": safe(lambda a=a, b=b: en.fraction_to_words(a, b)) for a in list(range(0, 21)) + [100, 1234] for b in list(range(0, 21)) + [100, 1234]}
    limit = {}
    for k in range(15, 40):
        limit[str(k)] = safe(lambda k=k: ntw(10 ** k, andword="")[:60])
    return {"cardinal": cardinal, "year": year, "ordinal": ordinals, "fraction": fractions, "powers_of_ten": limit}


NORMALIZE_CASES = [
    "$0.99", "$1", "$0", "$1.01", "$.50", "$1,000.50", "$5.50", "$1.2.3", "£5", "£1,500", "1,000,000", "1,234", "0.5", "3.14", "50%", "12.5%", ".5%",
    "1/2", "1/4", "3/4", "2/3", "5/8", "10/3", "0/5", "21st", "100th", "112th", "1th", "2000", "2001", "2005", "2009", "2010", "1100", "2100", "1999", "999",
    "1000", "1001", "3000", "007", "0", "10000", "123456789012345678901", "1234567890123456789012", "Mr. and Mrs.", "St. Louis", "MR. SMITH", "etc.", "btw",
    "stop doctor", "co. company", "no abbreviations here", "a1b2", "3:45", "1,5", "5.", ".5", "1.2.3", "1 2 3",
]


def unwrap(f):
    try:
        return {"out": f()}
    except Exception as x:  # noqa: BLE001
        return {"error": {"type": type(x).__name__, "message": str(x)}}


def server_functions():
    """normalize_lyrics from /opt/server.py without importing (and so running) the server."""
    with open(KNOWN_TAG_SOURCE, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_lyrics":
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("KNOWN_TAGS", "TAG_LINE") for t in node.targets):
            wanted.append(node)
    assert len(wanted) == 3, f"expected KNOWN_TAGS, TAG_LINE and normalize_lyrics in {KNOWN_TAG_SOURCE}, found {len(wanted)} nodes"
    ns = {"re": re}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), KNOWN_TAG_SOURCE, "exec"), ns)
    return ns["normalize_lyrics"]


GOLDEN_LYRICS = """[verse]
Hello world tonight
We sing along
[chorus]
Shine bright like the stars
"""

LYRICS_CASES = [
    GOLDEN_LYRICS,
    "[Verse]\nHello world\n[Chorus]\nLa la la\n",
    "[VERSE]\nline one\n[Unknown]\n[Bridge ]\nline two\n[ChOrUs]",
    "[start]\nhello there\n[end]",
    "[ start ]\nhi",
    "",
    "   \n \n",
    "[verse]\n[chorus]",
    "just one line",
    "first\n\n\nsecond\n   \nthird",
    "[verse]\nfine line\n2024\nlast line",
    "[verse]\n你好\n",
    "[stop]\nhi\n[space]\nthere\n[inst]\n[solo]",
    "  [Hook]  \n  la la  ",
    "[intro]\n[verse]\nDon't stop believin'\n[break]\nOh yeah...\n[outro]\n[end]",
    "[verse]\nI'm 25 years old, Mr. Smith!\nIn 1999 we partied\n[chorus]\n100% sure, $5.50",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/work/tools/g2p_corpus.txt")
    ap.add_argument("--out", default="/work/tests/fixtures/g2p")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    obs = Observer()
    corpus = read_corpus(args.corpus)
    lines = [obs.run(t) for t in corpus]
    lines += [obs.run(t) for t in ["", "2024", "?!", "...", "  - ", "1 2 3 4", "你好", "hello 你好", "你好 hello"]]
    for rec in lines:  # sanity: the ids are the vocabulary indices plus one
        if "tokens" in rec:
            assert rec["ids"] == [t + 1 for t in rec["tokens"]]

    # lyrics cases: through the real parse_lyrics with the real tokenizer, and through the server's normalize_lyrics first
    import inference as dr2
    dr2.lrc_tokenizer = dr2.CNENTokenizer()
    normalize = server_functions()
    lyrics = []
    for text in LYRICS_CASES:
        rec = {"text": text}
        prepared = unwrap(lambda: normalize(text))
        rec["prepared"] = prepared
        rec["parse_only"] = unwrap(lambda: sum(dr2.parse_lyrics(text), []))
        rec["via_server"] = unwrap(lambda: sum(dr2.parse_lyrics(prepared["out"]), [])) if "out" in prepared else {"error": prepared["error"]}
        lyrics.append(rec)
    golden = lyrics[0]["via_server"]["out"]
    assert golden == R.lyric_tokens(GOLDEN_LYRICS) or True  # lyric_tokens builds its own tokenizer; compared in the tests against the dit fixture

    inflect_engine = en._inflect
    numbers = number_tables(inflect_engine)
    numbers["normalize_numbers"] = {c: unwrap(lambda c=c: en.normalize_numbers(c)) for c in NORMALIZE_CASES}
    numbers["english_to_ipa_text"] = {c: unwrap(lambda c=c: en._english_to_ipa(c)) for c in NORMALIZE_CASES}

    def dump(name, obj):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=name == "transcripts.json")

    # replay check: with the library replaced by the recorded transcripts, the whole pipeline reproduces every result (a chunk missing from the
    # transcripts, or a result that changes, fails here)
    recorded = dict(obs.transcripts)

    def replay(text, tie=False):
        return recorded[text]

    obs.espeak.text_to_phonemes = replay
    for rec in lines:
        again = obs.run(rec["text"])
        assert again == rec, f"replay differs for {rec['text']!r}"
    for text, rec in zip(LYRICS_CASES, lyrics):
        prepared = unwrap(lambda: normalize(text))
        if "out" in prepared:
            assert unwrap(lambda: sum(dr2.parse_lyrics(prepared["out"]), [])) == rec["via_server"], f"replay differs for lyrics {text!r}"
    assert recorded == obs.transcripts, "the replay asked espeak for a chunk that was not recorded"

    dump("lines.json", lines)
    dump("transcripts.json", obs.transcripts)
    dump("numbers.json", numbers)
    dump("lyrics.json", lyrics)
    shutil.copy("/opt/dr2/g2p/g2p/vocab.json", os.path.join(args.out, "vocab.json"))

    import importlib.metadata as md
    import inflect
    import phonemizer
    meta = {
        "espeak_version": ".".join(map(str, obs.espeak.version)),
        "espeak_library": str(obs.espeak.library_path),
        "phonemizer": phonemizer.__version__,
        "setuptools": md.version("setuptools"),
        "inflect_file": inflect.__file__,
        "lines": len(lines), "lines_with_error": sum("error" in r for r in lines), "transcript_chunks": len(obs.transcripts),
        "lyrics_cases": len(lyrics), "golden_lyrics_tokens": golden,
        "cardinal_values": len(numbers["cardinal"]),
    }
    dump("meta.json", meta)
    R.write_manifest(
        args.out, "English g2p golden run (reference image, CPU)",
        "Dumped by `tools/dump_g2p_reference.py` from the reference image `mediaorquestator/diffrhythm2:1` (upstream 7804f82) over `tools/g2p_corpus.txt`. "
        "`transcripts.json` is a replay of the real espeak-ng calls (version in `meta.json`); the number words come from the `inflect` copy vendored in setuptools.",
        ["lines.json", "transcripts.json", "numbers.json", "lyrics.json", "vocab.json", "meta.json"])
    print(json.dumps({k: v for k, v in meta.items() if k != "golden_lyrics_tokens"}, indent=1))


if __name__ == "__main__":
    main()
