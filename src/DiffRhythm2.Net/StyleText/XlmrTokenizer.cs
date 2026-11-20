using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DiffRhythm2.StyleText;

/// <summary>
/// The XLM-RoBERTa tokenizer exactly as the reference runs it (<c>XLMRobertaTokenizerFast</c> built from <c>tokenizer.json</c>): the special tokens
/// are recognized in the raw text, the rest is normalized with the precompiled character map, split at whitespace, every word gets the marker
/// U+2581 in front (and is split again before any marker it contains), each piece is segmented by the Unigram model, and the ids are wrapped as
/// <c>&lt;s&gt; ... &lt;/s&gt;</c>.
/// </summary>
public sealed class XlmrTokenizer
{
    public const char WordMarker = '▁';

    private readonly PrecompiledNormalizer normalizer;
    private readonly UnigramModel unigram;
    private readonly (string Content, int Id)[] specials;

    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }
    public int UnknownId => unigram.UnknownId;
    public int VocabularySize => unigram.Count;

    private XlmrTokenizer(PrecompiledNormalizer normalizer, UnigramModel unigram, (string, int)[] specials, int bos, int eos, int pad)
    {
        this.normalizer = normalizer;
        this.unigram = unigram;
        this.specials = specials;
        BosId = bos;
        EosId = eos;
        PadId = pad;
    }

    /// <summary>Reads a <c>tokenizer.json</c> of an XLM-RoBERTa model (Unigram model, precompiled normalizer, template <c>&lt;s&gt; A &lt;/s&gt;</c>).</summary>
    public static XlmrTokenizer Load(string path)
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = doc.RootElement;

            var norm = root.GetProperty("normalizer");
            if (norm.GetProperty("type").GetString() != "Precompiled") throw new InvalidDataException("the normalizer is not a precompiled character map");
            var normalizer = new PrecompiledNormalizer(Convert.FromBase64String(norm.GetProperty("precompiled_charsmap").GetString()!));

            var model = root.GetProperty("model");
            if (model.TryGetProperty("type", out var type) && type.GetString() != "Unigram") throw new InvalidDataException($"the model is {type.GetString()}, not Unigram");
            var pieces = new List<string>();
            var scores = new List<double>();
            foreach (var entry in model.GetProperty("vocab").EnumerateArray())
            {
                pieces.Add(entry[0].GetString()!);
                scores.Add(entry[1].GetDouble());
            }
            var unigram = new UnigramModel(pieces, scores, model.GetProperty("unk_id").GetInt32());

            var specials = root.GetProperty("added_tokens").EnumerateArray()
                .Where(t => t.GetProperty("special").GetBoolean() && !t.GetProperty("normalized").GetBoolean())
                .Select(t => (t.GetProperty("content").GetString()!, t.GetProperty("id").GetInt32())).ToArray();

            int Id(string content) => specials.FirstOrDefault(s => s.Item1 == content) is { Item1: not null } s ? s.Item2 : throw new InvalidDataException($"the special token {content} is missing");
            return new XlmrTokenizer(normalizer, unigram, specials, Id("<s>"), Id("</s>"), Id("<pad>"));
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or FormatException or InvalidOperationException or IndexOutOfRangeException)
        {
            throw new InvalidDataException($"{path} is not a usable XLM-RoBERTa tokenizer.json: {e.Message}", e);
        }
    }

    /// <summary>The normalization alone, for text that is not a special token.</summary>
    public string Normalize(string text) => normalizer.Normalize(text);

    /// <summary>The ids of <paramref name="text"/> wrapped as <c>&lt;s&gt; ... &lt;/s&gt;</c>.</summary>
    public int[] Encode(string text)
    {
        EnsureValidUnicode(text);
        var ids = new List<int> { BosId };
        foreach (var (segment, special) in SplitSpecials(text))
        {
            if (special is { } id) { ids.Add(id); continue; }
            foreach (var word in Words(normalizer.Normalize(segment)))
                ids.AddRange(unigram.Encode(word));
        }
        ids.Add(EosId);
        return ids.ToArray();
    }

    private static void EnsureValidUnicode(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])) { i++; continue; }
            if (char.IsSurrogate(text[i])) throw new ArgumentException($"the text contains an unpaired surrogate at index {i}: it is not valid Unicode");
        }
    }

    /// <summary>Cuts the raw text at the special tokens (found before any normalization, leftmost first).</summary>
    private IEnumerable<(string Text, int? Special)> SplitSpecials(string text)
    {
        int from = 0;
        while (from <= text.Length)
        {
            int bestAt = -1, bestId = 0, bestLength = 0;
            foreach (var (content, id) in specials)
            {
                int at = text.IndexOf(content, from, StringComparison.Ordinal);
                if (at >= 0 && (bestAt < 0 || at < bestAt || (at == bestAt && content.Length > bestLength))) { bestAt = at; bestId = id; bestLength = content.Length; }
            }
            if (bestAt < 0) { yield return (text[from..], null); yield break; }
            if (bestAt > from) yield return (text[from..bestAt], null);
            yield return ("", bestId);
            from = bestAt + bestLength;
            if (from == text.Length) yield break;
        }
    }

    // Rust's char::is_whitespace (the White_Space property), which is what the reference's whitespace splitter uses.
    private static bool IsWhiteSpace(int c) =>
        c is (>= 0x09 and <= 0x0D) or 0x20 or 0x85 or 0xA0 or 0x1680 or (>= 0x2000 and <= 0x200A) or 0x2028 or 0x2029 or 0x202F or 0x205F or 0x3000;

    /// <summary>Whitespace split, then the word marker: a word that does not start with it gets one, and the word is cut before every marker.</summary>
    private static IEnumerable<string> Words(string normalized)
    {
        var current = new StringBuilder();
        var pieces = new List<string>();

        void Flush()
        {
            if (current.Length == 0) return;
            var word = current.ToString();
            current.Clear();
            if (word[0] != WordMarker) word = WordMarker + word;
            int start = 0;
            for (int i = 1; i < word.Length; i++)
            {
                if (word[i] == WordMarker) { pieces.Add(word[start..i]); start = i; }
            }
            pieces.Add(word[start..]);
        }

        foreach (var rune in normalized.EnumerateRunes())
        {
            if (IsWhiteSpace(rune.Value)) Flush();
            else current.Append(rune.ToString());
        }
        Flush();
        return pieces;
    }
}
