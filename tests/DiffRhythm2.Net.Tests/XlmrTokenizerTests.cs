using System.Text;
using System.Text.Json;
using DiffRhythm2.StyleText;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>The XLM-RoBERTa tokenizer against the reference (HuggingFace fast tokenizer): every code point's normalization, a corpus of multi-character strings and a corpus of texts. Needs the tokenizer.json copied next to the converted checkpoint.</summary>
public class XlmrTokenizerTests
{
    private static readonly Lazy<XlmrTokenizer> Shared = new(() => XlmrTokenizer.Load(MulanTextReference.TokenizerPath!));
    private readonly ITestOutputHelper log;

    public XlmrTokenizerTests(ITestOutputHelper log) => this.log = log;

    private static XlmrTokenizer Tokenizer => Shared.Value;

    [RequiresMulanTokenizer]
    public void Load_ReadsTheVocabularyAndSpecialTokens()
    {
        Assert.Equal(250002, Tokenizer.VocabularySize);
        Assert.Equal((0, 2, 1, 3), (Tokenizer.BosId, Tokenizer.EosId, Tokenizer.PadId, Tokenizer.UnknownId));
    }

    [Fact]
    public void Load_RejectsAFileThatIsNotATokenizer()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "{\"model\": {}}");
            var e = Assert.Throws<InvalidDataException>(() => XlmrTokenizer.Load(path));
            Assert.Contains("not a usable", e.Message);
            File.WriteAllText(path, "not json");
            Assert.Throws<InvalidDataException>(() => XlmrTokenizer.Load(path));
        }
        finally { File.Delete(path); }
    }

    [RequiresMulanTokenizer]
    public void EveryCodePointOfPlanes0To2_IsNormalizedAsTheReferenceDoes()
    {
        var changed = MulanTextReference.Normalization.GetProperty("changed").EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.GetString()!);
        var failures = new List<string>();
        for (int cp = 0; cp < 0x30000; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF) continue;
            var text = char.ConvertFromUtf32(cp);
            var expected = changed.TryGetValue(cp, out var e) ? e : text;
            var got = Tokenizer.Normalize(text);
            if (got != expected) failures.Add($"U+{cp:X4}: expected {Show(expected)}, got {Show(got)}");
        }
        log.WriteLine($"{changed.Count} changed code points of planes 0 to 2 checked, {failures.Count} differences");
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(20)));
    }

    [RequiresMulanTokenizer]
    public void Sequences_AreNormalizedAsTheReferenceDoes()
    {
        var failures = new List<string>();
        int n = 0;
        foreach (var pair in MulanTextReference.Normalization.GetProperty("sequences").EnumerateArray())
        {
            var input = pair.GetProperty("in").GetString()!;
            var expected = pair.GetProperty("out").GetString()!;
            var got = Tokenizer.Normalize(input);
            if (got != expected) failures.Add($"{Show(input)}: expected {Show(expected)}, got {Show(got)}");
            n++;
        }
        log.WriteLine($"{n} strings normalized, {failures.Count} differences");
        Assert.True(failures.Count == 0, $"{failures.Count} of {n} differ:\n" + string.Join("\n", failures.Take(15)));
    }

    [RequiresMulanTokenizer]
    public void InformationalOnly_TheLongestMatchRuleWouldNotBeTheReferences()
    {
        // The reference uses the first (shortest) prefix found in the character map. This reports how many strings of the corpus tell the two rules apart,
        // i.e. how strongly the sequence corpus pins the rule down. It asserts nothing about the number.
        var json = JsonDocument.Parse(File.ReadAllBytes(MulanTextReference.TokenizerPath!)).RootElement;
        var blob = Convert.FromBase64String(json.GetProperty("normalizer").GetProperty("precompiled_charsmap").GetString()!);
        var shortest = new PrecompiledNormalizer(blob);
        var longest = new PrecompiledNormalizer(blob) { PreferLongest = true };
        int differ = MulanTextReference.Normalization.GetProperty("sequences").EnumerateArray()
            .Count(p => shortest.Normalize(p.GetProperty("in").GetString()!) != longest.Normalize(p.GetProperty("in").GetString()!));
        log.WriteLine($"[informational] {differ} of the corpus strings are normalized differently by the longest-match rule");
        Assert.NotEmpty(MulanTextReference.Normalization.GetProperty("sequences").EnumerateArray());   // the number is only reported, but the corpus has to exist
    }

    [RequiresMulanTokenizer]
    public void Corpus_IdsEqualTheReference()
    {
        var failures = new List<string>();
        int n = 0;
        foreach (var item in MulanTextReference.Tokens.EnumerateArray())
        {
            var text = item.GetProperty("text").GetString()!;
            var expected = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var got = Tokenizer.Encode(text);
            if (!expected.SequenceEqual(got)) failures.Add($"{Show(text.Length > 60 ? text[..60] : text)}: expected [{string.Join(",", expected.Take(16))}...], got [{string.Join(",", got.Take(16))}...]");
            n++;
        }
        log.WriteLine($"{n} texts tokenized, {failures.Count} differences");
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(15)));
    }

    [RequiresMulanTokenizer]
    public void SpecScenarios()
    {
        Assert.Equal(new[] { 0, 11584, 4, 115972, 4, 213342, 108171, 7, 4, 32506, 4, 138276, 2 }, Tokenizer.Encode("Pop, Piano, Female vocals, Happy, Bright"));
        Assert.Equal(new[] { 0, 2 }, Tokenizer.Encode(""));
        Assert.Equal(new[] { 0, 2 }, Tokenizer.Encode("  \t\n "));
        Assert.Equal(new[] { 0, 4173, 32628, 7, 2 }, Tokenizer.Encode("  extra   spaces "));
        Assert.Equal(Tokenizer.Encode("fine FULL 12"), Tokenizer.Encode("ﬁne ＦＵＬＬ ①②"));        // ligature, full-width, circled digits
        Assert.Equal(Tokenizer.Encode("a b"), Tokenizer.Encode("a b"));
        Assert.Equal(Tokenizer.Encode("a b"), Tokenizer.Encode("a​b"));
        var fused = Tokenizer.Encode("x" + char.ConvertFromUtf32(0x1FA00) + char.ConvertFromUtf32(0x1FA01) + " y");
        Assert.Equal(new[] { 0, 1022, 3, 113, 2 }, fused);
        Assert.Equal(3, Tokenizer.Encode("a\0b")[2]);
    }

    [RequiresMulanTokenizer]
    public void UnpairedSurrogate_IsRejected()
    {
        var e = Assert.Throws<ArgumentException>(() => Tokenizer.Encode("bad \ud800 text"));
        Assert.Contains("surrogate", e.Message);
    }

    private static string Show(string s) => string.Concat(s.EnumerateRunes().Select(r => r.Value is >= 0x20 and < 0x7F ? r.ToString() : $"<U+{r.Value:X}>"));
}
