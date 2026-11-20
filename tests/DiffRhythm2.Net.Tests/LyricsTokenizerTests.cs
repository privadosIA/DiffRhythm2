using System.Text.Json;
using DiffRhythm2.Model;
using DiffRhythm2.Sampling;
using DiffRhythm2.Text;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Lyrics preparation and tokenization against the reference's whole-lyrics cases, with the recorded espeak answers (no library needed).</summary>
public class LyricsTokenizerTests
{
    private readonly ITestOutputHelper log;

    public LyricsTokenizerTests(ITestOutputHelper log) => this.log = log;

    private static LyricsTokenizer Tokenizer() => new(new EnglishG2p(new ReplayEngine()));

    // The reference tokenizes Chinese; this front end refuses it on purpose (design D8), so those cases are checked for the refusal instead.
    private static bool HasChinese(string text) => text.Any(c => c is >= '\u4e00' and <= '\u9fa5');

    private static void AssertChineseRefused(Action tokenize)
    {
        var e = Assert.Throws<LyricsLineException>(tokenize);
        Assert.IsType<ChineseNotSupportedException>(e.InnerException);
    }

    private static long[] Longs(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt64()).ToArray();

    [Fact]
    public void Prepare_MatchesTheServiceOnEveryCase()
    {
        int ok = 0, errors = 0;
        foreach (var c in G2pReference.Lyrics.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var prepared = c.GetProperty("prepared");
            if (prepared.TryGetProperty("error", out _)) { Assert.Throws<G2pException>(() => LyricsTokenizer.Prepare(text)); errors++; }
            else { Assert.Equal(prepared.GetProperty("out").GetString(), LyricsTokenizer.Prepare(text)); ok++; }
        }
        log.WriteLine($"{ok} cases prepared identically, {errors} rejected like the service");
        Assert.True(ok >= 10 && errors >= 3);
    }

    [Fact]
    public void Tokenize_MatchesParseLyrics_OnEveryCase()
    {
        var tokenizer = Tokenizer();
        int ok = 0, errors = 0, chinese = 0;
        foreach (var c in G2pReference.Lyrics.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var parseOnly = c.GetProperty("parse_only");
            if (HasChinese(text)) { AssertChineseRefused(() => tokenizer.Tokenize(text)); chinese++; continue; }
            if (parseOnly.TryGetProperty("error", out _)) { Assert.ThrowsAny<G2pException>(() => tokenizer.Tokenize(text)); errors++; }
            else { Assert.Equal(Longs(parseOnly.GetProperty("out")), tokenizer.Tokenize(text)); ok++; }
        }
        log.WriteLine($"{ok} cases tokenized identically, {errors} rejected like the reference, {chinese} Chinese refused");
        Assert.True(ok >= 10);
    }

    [Fact]
    public void PrepareAndTokenize_MatchesTheServiceFlow_OnEveryCase()
    {
        var tokenizer = Tokenizer();
        foreach (var c in G2pReference.Lyrics.EnumerateArray())
        {
            var text = c.GetProperty("text").GetString()!;
            var viaServer = c.GetProperty("via_server");
            if (HasChinese(text)) { AssertChineseRefused(() => tokenizer.PrepareAndTokenize(text)); continue; }
            if (viaServer.TryGetProperty("error", out _)) Assert.ThrowsAny<G2pException>(() => tokenizer.PrepareAndTokenize(text));
            else Assert.Equal(Longs(viaServer.GetProperty("out")), tokenizer.PrepareAndTokenize(text));
        }
    }

    [Fact]
    public void Structure_AndStartInsertion_FollowTheSpec()
    {
        var t = Tokenizer();
        Assert.Equal(new long[] { 500, 511, 503, 511 }, t.Tokenize("[verse]"));
        Assert.Equal(new long[] { 500, 511 }, t.Tokenize("[start]"));
        Assert.Equal(new long[] { 500, 511, 501, 511 }, t.Tokenize("[start]\n[end]"));
        Assert.Empty(t.Tokenize(""));
        Assert.Equal(new long[] { 500, 511, 503, 511 }, t.Tokenize("[VERSE]"));          // tags are matched ignoring case
        Assert.Equal(new long[] { 500, 511 }, t.Tokenize("[Unknown]\n[start]"));          // an unknown tag line is skipped
        Assert.Equal(13, LyricsTokenizer.TagIds.Count);
    }

    [Fact]
    public void Prepare_NormalizesTags_AndRejectsLyricsWithoutText()
    {
        Assert.Equal("[start]\n[verse]\nline one\n[bridge]\nline two", LyricsTokenizer.Prepare("[VERSE]\nline one\n[Unknown]\n[Bridge ]\nline two"));
        Assert.Equal("[start]\nhello", LyricsTokenizer.Prepare("hello"));
        Assert.Equal("[start]\nhi", LyricsTokenizer.Prepare("[start]\nhi"));
        var e = Assert.Throws<G2pException>(() => LyricsTokenizer.Prepare("[verse]\n[chorus]"));
        Assert.Contains("nothing to sing", e.Message);
    }

    [Fact]
    public void FailingLine_IsNamedWithItsNumberInTheOriginalText()
    {
        var tokenizer = Tokenizer();
        var e = Assert.Throws<LyricsLineException>(() => tokenizer.PrepareAndTokenize("[verse]\nHello world tonight\n\n2024\nWe sing along"));
        Assert.Equal(4, e.Line);                       // the blank line counts: it is the fourth line of what the user wrote
        Assert.Equal("2024", e.Text);
        Assert.Contains("line 4", e.Message);
        Assert.IsType<NoSungTextException>(e.InnerException);
    }

    [Fact]
    public void GoldenRunLyrics_ReproduceTheSamplerInputFixture()
    {
        var lyrics = "[verse]\nHello world tonight\nWe sing along\n[chorus]\nShine bright like the stars\n";
        var tokens = Tokenizer().PrepareAndTokenize(lyrics);
        var expected = Reference.Inputs["tokens"].data<long>().ToArray();
        log.WriteLine($"{tokens.Length} tokens, fixture {expected.Length}");
        Assert.Equal(expected, tokens);
        using var tensor = LyricsTokenizer.ToTensor(tokens);
        Assert.Equal(new long[] { tokens.Length }, tensor.shape);
        Assert.Equal(ScalarType.Int64, tensor.dtype);
    }

    [Fact]
    public void Tokens_FeedTheSamplerAsItsTextInput()
    {
        manual_seed(1);
        using var model = new Dit(new DitConfig(Dim: 32, Depth: 2, Heads: 2, FfMult: 2, MelDim: 8, TextNumEmbeds: 1000, BlockSize: 10));
        using var tokens = LyricsTokenizer.ToTensor(Tokenizer().PrepareAndTokenize("[verse]\nHello world tonight\n[chorus]\nShine bright like the stars"));
        using var style = randn(1, 512);
        var noise = new List<Tensor> { randn(1, 10, 8) };
        using var latent = new BlockSampler(model).Sample(tokens, style, noise, frames: 10, new SamplerOptions(Steps: 2, CfgStrength: 2.0));
        Assert.Equal(new long[] { 1, 10, 8 }, latent.shape);
        Assert.True(latent.isfinite().all().item<bool>());
    }
}
