using System.Security.Cryptography;
using System.Text.Json;
using DiffRhythm2.Text;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>The English line front end: vocabulary, phoneme clean-up, segmentation, errors, and the whole corpus replaying the reference's espeak answers (no library needed).</summary>
public class EnglishG2pTests
{
    private readonly ITestOutputHelper log;

    public EnglishG2pTests(ITestOutputHelper log) => this.log = log;

    private static EnglishG2p G2p() => new(new ReplayEngine());

    // ---- vocabulary ------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void EmbeddedVocabulary_IsTheReferenceFile()
    {
        var embedded = typeof(PhonemeVocabulary).Assembly.GetManifestResourceStream("DiffRhythm2.Text.vocab.json")!;
        using var ms = new MemoryStream();
        embedded.CopyTo(ms);
        var fixtureBytes = File.ReadAllBytes(G2pReference.Directory.PathOf("vocab.json"));
        Assert.Equal(Convert.ToHexString(SHA256.HashData(fixtureBytes)), Convert.ToHexString(SHA256.HashData(ms.ToArray())));

        var vocab = PhonemeVocabulary.Default;
        Assert.Equal(363, vocab.Count);
        Assert.True(vocab.TryGetId(",", out var comma)); Assert.Equal(0, comma);
        Assert.True(vocab.TryGetId("_", out var space)); Assert.Equal(4, space);
        Assert.True(vocab.TryGetId("…", out var ellipsis)); Assert.Equal(362, ellipsis);
    }

    [Fact]
    public void Vocabulary_SkipsUnknownItems_AndKeepsTheOthersUnchanged()
    {
        var vocab = new PhonemeVocabulary(new Dictionary<string, int> { ["a"] = 3, ["b"] = 9 });
        Assert.Equal(new[] { 3, 9 }, vocab.Tokens("a|zzz|b|"));
        Assert.Empty(vocab.Tokens(""));
    }

    // ---- segmentation ------------------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("abc 123", new[] { "abc 123|en" })]
    [InlineData("123 abc", new[] { "123 abc|en" })]
    [InlineData("hello! 你好", new[] { "hello! |en", "你好|zh" })]
    [InlineData("你好, hello", new[] { "你好, |zh", "hello|en" })]
    [InlineData("2024", new[] { "2024|other" })]
    [InlineData("?!", new[] { "?!|other" })]
    [InlineData("", new[] { "|" })]
    [InlineData("café au lait", new[] { "café au lait|en" })]
    public void Segment_FollowsTheReference(string text, string[] expected)
    {
        var got = EnglishG2p.Segment(text).Select(s => $"{s.Text}|{s.Lang}").ToArray();
        Assert.Equal(expected, got);
    }

    [Fact]
    public void LineWithoutLetters_HasNoSungText()
    {
        foreach (var text in new[] { "2024", "?!", "...", "  - ", "1 2 3 4", "" })
            Assert.Throws<NoSungTextException>(() => G2p().Convert(text));
        Assert.Contains("2024", Assert.Throws<NoSungTextException>(() => G2p().Convert("2024")).Message);
    }

    [Fact]
    public void ChineseText_IsRejectedExplicitly()
    {
        foreach (var text in new[] { "hello 你好", "你好", "你好 hello" })
        {
            var e = Assert.Throws<ChineseNotSupportedException>(() => G2p().Convert(text));
            Assert.Contains("not supported", e.Message);
        }
    }

    // ---- phoneme clean-up ---------------------------------------------------------------------------------------------------------------------

    [Theory]
    [InlineData("a|t|s|b", "a|ts|b")]
    [InlineData("d|z", "dz")]
    [InlineData("t|ɹ|i", "tɹ|i")]
    [InlineData("x", "k")]
    [InlineData("k|x|x|k", "k|k|k|k")]
    [InlineData("ɐ|ɐ", "ɚ|ɚ")]
    [InlineData("w|əl|d", "w|l|d")]
    [InlineData("oː|ɹ", "oːɹ")]
    [InlineData("tx", "tx")]                 // only whole phonemes are rewritten
    [InlineData("t|s|", "ts|")]
    public void SpecialMap_RewritesWholePhonemes(string input, string expected) => Assert.Equal(expected, EnglishG2p.SpecialMap(input));

    // ---- layers of the reference, without espeak ------------------------------------------------------------------------------------------

    [Fact]
    public void ReferencePhonemeStrings_MapToTheReferenceIds_WithoutEspeak()
    {
        int n = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            var tokens = PhonemeVocabulary.Default.Tokens(line.GetProperty("phonemes").GetString()!);
            Assert.Equal(G2pReference.Ints(line.GetProperty("tokens")), tokens.ToArray());
            Assert.Equal(G2pReference.Ints(line.GetProperty("ids")), tokens.Select(t => t + 1).ToArray());
            n++;
        }
        log.WriteLine($"{n} lines: vocabulary lookup equals the reference");
    }

    [Fact]
    public void WholeCorpus_ReplayingEspeak_MatchesEveryLayer()
    {
        var g2p = G2p();
        int n = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            var text = line.GetProperty("text").GetString()!;
            Assert.Equal(line.GetProperty("ipa").GetString(), g2p.EnglishToIpa(text));
            var got = g2p.Convert(text);
            Assert.Equal(line.GetProperty("phonemes").GetString(), got.Phonemes);
            Assert.Equal(G2pReference.Ints(line.GetProperty("tokens")), got.Tokens.ToArray());
            Assert.Equal(G2pReference.Ints(line.GetProperty("ids")), got.Ids);
            n++;
        }
        log.WriteLine($"{n} corpus lines equal the reference at the ipa, phoneme and id layers");
        Assert.True(n > 90);
    }

    [Fact]
    public void ReferenceFailures_AreFailuresHere()
    {
        var g2p = G2p();
        int errors = 0, chinese = 0;
        foreach (var line in G2pReference.Lines.EnumerateArray())
        {
            var text = line.GetProperty("text").GetString()!;
            if (line.TryGetProperty("error", out _)) { Assert.Throws<NoSungTextException>(() => g2p.Convert(text)); errors++; }
            else if (line.GetProperty("has_chinese").GetBoolean()) { Assert.Throws<ChineseNotSupportedException>(() => g2p.Convert(text)); chinese++; }
        }
        log.WriteLine($"{errors} lines with no sung text and {chinese} Chinese lines rejected");
        Assert.True(errors >= 5 && chinese >= 3);
    }

    [Fact]
    public void LineNeverEndsWithTheWordSeparator()
    {
        var g2p = G2p();
        foreach (var line in G2pReference.GoodLines)
        {
            var tokens = g2p.Convert(line.GetProperty("text").GetString()!).Tokens;
            if (tokens.Count > 0) Assert.NotEqual(4, tokens[^1]);   // 4 is the id of "_" in the vocabulary
        }
    }
}
