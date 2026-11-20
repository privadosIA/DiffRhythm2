using DiffRhythm2.Text;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>[Fact] that needs libespeak-ng (see EspeakEngine); with a version it also needs exactly that major.minor, the one the reference output was produced with.</summary>
public sealed class RequiresEspeakAttribute : FactAttribute
{
    public const string ReferenceVersion = "1.51";

    public RequiresEspeakAttribute(bool referenceVersionOnly = false)
    {
        if (!EspeakEngine.TryGetShared(out var engine, out var reason)) Skip = reason;
        else if (referenceVersionOnly && !engine!.VersionText.StartsWith(ReferenceVersion, StringComparison.Ordinal))
            Skip = $"the gating comparison needs espeak-ng {ReferenceVersion}, the version of the reference; the loaded library is {engine.VersionText} (the informational test reports the differences)";
    }
}

/// <summary>The real espeak-ng library: discovery errors, version, concurrency, and the corpus against the reference (gating only for the reference's version).</summary>
public class EspeakTests
{
    private readonly ITestOutputHelper log;

    public EspeakTests(ITestOutputHelper log) => this.log = log;

    [Fact]
    public void MissingLibrary_FailsListingWhatWasTried()
    {
        var e = Assert.Throws<EspeakUnavailableException>(() => EspeakEngine.Load(new EspeakOptions(LibraryPath: "/nonexistent/path/libespeak-ng.so.1")));
        Assert.Contains("/nonexistent/path/libespeak-ng.so.1", e.Message);
        Assert.Contains(EspeakEngine.LibraryVariable, e.Message);
        Assert.Contains("1.51", e.Message);
    }

    [Fact]
    public void LibraryThatIsNotEspeak_IsReportedAsSuch()
    {
        // libm is always there and has none of the espeak functions
        var candidates = new[] { "libm.so.6", "libm.dylib", "msvcrt.dll" };
        var found = candidates.FirstOrDefault(c => System.Runtime.InteropServices.NativeLibrary.TryLoad(c, out _));
        if (found is null) return;
        var e = Assert.Throws<EspeakUnavailableException>(() => EspeakEngine.Load(new EspeakOptions(LibraryPath: found)));
        Assert.Contains("does not look like libespeak-ng", e.Message);
    }

    [RequiresEspeak]
    public void Version_IsReported()
    {
        var engine = EspeakEngine.Shared;
        log.WriteLine($"{engine} (version {engine.Version}, data {engine.DataPath})");
        Assert.NotEmpty(engine.VersionText);
        Assert.True(engine.Version >= new Version(1, 49));
    }

    [RequiresEspeak]
    public void FixedText_ReturnsIpa()
    {
        var raw = EspeakEngine.Shared.TextToPhonemes("Hello world");
        log.WriteLine($"raw: {raw}");
        Assert.False(string.IsNullOrWhiteSpace(raw));
        Assert.Contains("_", raw);       // phonemes are separated by underscores
        var phonemes = new EspeakPhonemizer(EspeakEngine.Shared).Phonemize("Hello world");
        log.WriteLine($"phonemes: {phonemes}");
        Assert.StartsWith("h|", phonemes);
        Assert.Contains("|_|", phonemes);
    }

    [RequiresEspeak]
    public void ConcurrentCalls_EqualSequentialOnes()
    {
        var phonemizer = new EspeakPhonemizer(EspeakEngine.Shared);
        var lines = G2pReference.GoodLines.Select(l => l.GetProperty("normalized").GetString()!).Take(40).ToArray();
        var sequential = lines.Select(phonemizer.Phonemize).ToArray();

        var results = new string[lines.Length * 4];
        Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i => results[i] = phonemizer.Phonemize(lines[i % lines.Length]));
        for (int i = 0; i < results.Length; i++) Assert.Equal(sequential[i % lines.Length], results[i]);
    }

    // ---- the corpus against the reference ------------------------------------------------------------------------------------------------

    [RequiresEspeak(referenceVersionOnly: true)]
    public void Corpus_EqualsTheReference_WithTheReferenceVersion()
    {
        var g2p = new EnglishG2p(EspeakEngine.Shared);
        int n = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            var text = line.GetProperty("text").GetString()!;
            var got = g2p.Convert(text);
            Assert.True(line.GetProperty("phonemes").GetString() == got.Phonemes, $"\"{text}\": expected {line.GetProperty("phonemes").GetString()}, got {got.Phonemes}");
            Assert.Equal(G2pReference.Ints(line.GetProperty("ids")), got.Ids);
            n++;
        }
        log.WriteLine($"{n} corpus lines equal the reference with {EspeakEngine.Shared}");
    }

    [RequiresEspeak]
    public void Corpus_DifferencesFromTheReference_AreReported()
    {
        var g2p = new EnglishG2p(EspeakEngine.Shared);
        var different = new List<string>();
        int n = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            var text = line.GetProperty("text").GetString()!;
            var got = g2p.Convert(text);
            if (got.Phonemes != line.GetProperty("phonemes").GetString())
                different.Add($"\"{text}\"\n    reference: {line.GetProperty("phonemes").GetString()}\n    here:      {got.Phonemes}");
            n++;
        }
        log.WriteLine($"[informational] {EspeakEngine.Shared} vs the reference's espeak-ng {G2pReference.Meta.GetProperty("espeak_version").GetString()}: {n - different.Count}/{n} lines identical, {different.Count} differ");
        foreach (var d in different) log.WriteLine(d);
        Assert.True(n > 0, "the corpus is empty");   // the differences are only reported, but the corpus has to have been read
    }

    [RequiresEspeak(referenceVersionOnly: true)]
    public void GoldenRunLyrics_WithTheRealLibrary_ReproduceTheSamplerInput()
    {
        var tokenizer = new LyricsTokenizer(new EnglishG2p(EspeakEngine.Shared));
        var tokens = tokenizer.PrepareAndTokenize("[verse]\nHello world tonight\nWe sing along\n[chorus]\nShine bright like the stars\n");
        Assert.Equal(Reference.Inputs["tokens"].data<long>().ToArray(), tokens);
    }
}
