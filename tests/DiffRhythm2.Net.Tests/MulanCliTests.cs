using System.Globalization;
using System.Text.Json;
using DiffRhythm2.AudioTower;
using DiffRhythm2.StyleText;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The embed and classify commands that need no model: arguments, vocabulary, the checks made before any work.</summary>
public class MulanCliTests : IDisposable
{
    private readonly string dir = Directory.CreateTempSubdirectory("dr2-mulancli-").FullName;

    public void Dispose()
    {
        Directory.Delete(dir, recursive: true);
        GC.SuppressFinalize(this);
    }

    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Cli.Cli.Run(args, o, e, new StringReader(""));
        return (code, o.ToString(), e.ToString());
    }

    private string Wav(string name, int rate, int samples = 30000)
    {
        var path = Path.Combine(dir, name);
        File.WriteAllBytes(path, WavBuilder.Make(3, 1, rate, 32, WavBuilder.Float32(Enumerable.Range(0, samples).Select(i => 0.1f * (float)Math.Sin(i * 0.05)).ToArray())));
        return path;
    }

    private string Vocab(string json)
    {
        var path = Path.Combine(dir, "vocab.json");
        File.WriteAllText(path, json);
        return path;
    }

    // ---- the vocabulary ----------------------------------------------------------------------------------------------------------

    [Fact]
    public void Vocabulary_APlainListUsesTheTagAsTheText_AndATemplateIsFilled()
    {
        var v = Cli.Cli.ParseVocabulary("""{ "genre": { "template": "This is a {} song.", "tags": ["jazz", "rock"] }, "mood": ["happy", "sad"] }""", "v.json");
        Assert.Equal(new[] { "genre", "mood" }, v.Select(c => c.Name));
        Assert.Equal(new[] { "jazz", "rock" }, v[0].Tags);
        Assert.Equal(new[] { "This is a jazz song.", "This is a rock song." }, v[0].Texts);
        Assert.Equal(new[] { "happy", "sad" }, v[1].Texts);
        Assert.Equal(v[1].Tags, v[1].Texts);
    }

    [Theory]
    [InlineData("""{ "genre": [] }""", "genre")]
    [InlineData("""{ "genre": { "tags": [] } }""", "genre")]
    [InlineData("""{ "genre": ["jazz", ""] }""", "genre")]
    [InlineData("""{ "genre": ["jazz", 3] }""", "genre")]
    [InlineData("""{ "genre": ["jazz", "jazz"] }""", "genre")]
    [InlineData("""{ "genre": "jazz" }""", "genre")]
    [InlineData("""{ "genre": { "template": "no braces", "tags": ["jazz"] } }""", "genre")]
    [InlineData("""{ "genre": { "template": 5, "tags": ["jazz"] } }""", "genre")]
    [InlineData("""{ "genre": { "tags": ["jazz"], "extra": 1 } }""", "extra")]
    [InlineData("""{ "genre": { "template": "{}" } }""", "genre")]
    [InlineData("""{ "genre": ["a"], "genre": ["b"] }""", "genre")]
    [InlineData("""{ "": ["a"] }""", "empty name")]
    [InlineData("""{ }""", "no categories")]
    [InlineData("""["jazz"]""", "JSON object")]
    [InlineData("""{ "genre": [ """, "not valid JSON")]
    public void Vocabulary_BadShapes_AreRefused_NamingTheCategoryOrTheProblem(string json, string mentions)
    {
        var e = Assert.Throws<Cli.Cli.ArgumentError>(() => Cli.Cli.ParseVocabulary(json, "v.json"));
        Assert.Contains("--vocab", e.Message);
        Assert.Contains(mentions, e.Message);
    }

    [Fact]
    public void Vocabulary_TwoTagsThatGiveTheSameText_AreRefused()
    {
        var e = Assert.Throws<Cli.Cli.ArgumentError>(() => Cli.Cli.ParseVocabulary("""{ "g": { "template": "x", "tags": ["a"] } }""", "v.json"));
        Assert.Contains("{}", e.Message);
    }

    // ---- embed: arguments --------------------------------------------------------------------------------------------------------

    [Fact]
    public void Embed_NeitherOrBoth_FailsWithExit2_NamingBothOptions()
    {
        foreach (var args in new[] { new[] { "embed", "--model", dir }, new[] { "embed", "--model", dir, "--audio", "a.wav", "--text", "x" } })
        {
            var (code, _, err) = Run(args);
            Assert.Equal(2, code);
            Assert.Contains("--audio", err);
            Assert.Contains("--text", err);
        }
    }

    [Fact]
    public void Embed_MissingModel_UnknownOption_AndBadDevice_AreUsageErrors()
    {
        Assert.Contains("--model", Run("embed", "--text", "x").Err);
        Assert.Equal(2, Run("embed", "--text", "x").Code);
        Assert.Contains("unknown option '--bogus'", Run("embed", "--model", dir, "--text", "x", "--bogus", "1").Err);
        var (code, _, err) = Run("embed", "--model", dir, "--text", "x", "--device", "tpu");
        Assert.Equal(2, code);
        Assert.Contains("--device", err);
        Assert.Equal(2, Run("embed", "--model", dir, "--text", "  ").Code);
    }

    [Fact]
    public void Embed_AudioWithAnotherSampleRate_FailsBeforeAnyModelIsLoaded()
    {
        var (code, output, err) = Run("embed", "--model", Path.Combine(dir, "no-such-model-dir"), "--audio", Wav("cd.wav", 44100));
        Assert.Equal(1, code);
        Assert.Equal("", output);
        Assert.Contains("cd.wav", err);
        Assert.Contains("44100", err);
        Assert.Contains("24000", err);
        Assert.Contains("ffmpeg", err);
    }

    [Fact]
    public void Embed_AMissingOrUnreadableAudioFile_FailsWithExit1_NamingIt()
    {
        var (code, _, err) = Run("embed", "--model", dir, "--audio", Path.Combine(dir, "gone.wav"));
        Assert.Equal(1, code);
        Assert.Contains("gone.wav", err);
        File.WriteAllText(Path.Combine(dir, "text.wav"), "not a wav");
        var (code2, _, err2) = Run("embed", "--model", dir, "--audio", Path.Combine(dir, "text.wav"));
        Assert.Equal(1, code2);
        Assert.Contains("not a WAV", err2);
        var (code3, _, err3) = Run("embed", "--model", dir, "--audio", Wav("tiny.wav", 24000, samples: 100));
        Assert.Equal(1, code3);
        Assert.Contains("513", err3);
    }

    [Fact]
    public void Embed_Audio_WithoutTheAudioCheckpoint_SaysHowToCreateIt()
    {
        var (code, output, err) = Run("embed", "--model", dir, "--audio", Wav("ok.wav", 24000));
        Assert.Equal(1, code);
        Assert.Equal("", output);
        Assert.Contains("mulan_audio.safetensors", err);
        Assert.Contains("convert", err);
        Assert.Contains("--audio", err);
        Assert.Contains(dir, err);
    }

    [Fact]
    public void Embed_Text_WithoutTheTextCheckpoint_SaysHowToCreateIt_ListingEveryMissingFile()
    {
        var (code, _, err) = Run("embed", "--model", dir, "--text", "Pop");
        Assert.Equal(1, code);
        Assert.Contains("mulan_text.safetensors", err);
        Assert.Contains("mulan_text_tokenizer.json", err);
        Assert.Contains("diffrhythm2 convert", err);
    }

    [Fact]
    public void Embed_AModelDirectoryThatDoesNotExist_IsSaid()
    {
        var (code, _, err) = Run("embed", "--model", Path.Combine(dir, "nowhere"), "--text", "Pop");
        Assert.Equal(1, code);
        Assert.Contains("does not exist", err);
    }

    // ---- classify: arguments -----------------------------------------------------------------------------------------------------

    [Fact]
    public void Classify_MissingOptions_AreUsageErrors()
    {
        Assert.Contains("--vocab", Run("classify", "--model", dir, "--audio", "a.wav").Err);
        Assert.Contains("--audio", Run("classify", "--model", dir, "--vocab", "v.json").Err);
        Assert.Equal(2, Run("classify", "--audio", "a.wav", "--vocab", "v.json").Code);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("two")]
    [InlineData("1.5")]
    public void Classify_BadTop_IsAUsageError_NamingTheOption(string top)
    {
        var (code, _, err) = Run("classify", "--model", dir, "--audio", "a.wav", "--vocab", Vocab("""{"g": ["a"]}"""), "--top", top);
        Assert.Equal(2, code);
        Assert.Contains("--top", err);
    }

    [Fact]
    public void Classify_BadVocabulary_FailsWithExit2_NamingTheCategory_BeforeTheAudioIsRead()
    {
        var (code, output, err) = Run("classify", "--model", dir, "--audio", Path.Combine(dir, "never-read.wav"), "--vocab", Vocab("""{"genre": ["jazz"], "mood": []}"""));
        Assert.Equal(2, code);
        Assert.Equal("", output);
        Assert.Contains("mood", err);
        Assert.DoesNotContain("never-read", err);
    }

    [Fact]
    public void Classify_AVocabularyFileThatDoesNotExist_IsAUsageError()
    {
        var (code, _, err) = Run("classify", "--model", dir, "--audio", "a.wav", "--vocab", Path.Combine(dir, "nope.json"));
        Assert.Equal(2, code);
        Assert.Contains("--vocab", err);
        Assert.Contains("nope.json", err);
    }

    [Fact]
    public void Classify_WrongSampleRate_FailsWithExit1_BeforeLoadingAnyModel()
    {
        var (code, _, err) = Run("classify", "--model", Path.Combine(dir, "no-such-model-dir"), "--audio", Wav("cd.wav", 44100), "--vocab", Vocab("""{"g": ["a"]}"""));
        Assert.Equal(1, code);
        Assert.Contains("cd.wav", err);
        Assert.Contains("44100", err);
        Assert.Contains("24000", err);
    }

    [Fact]
    public void Classify_WithoutTheAudioCheckpoint_SaysHowToCreateIt()
    {
        // the text files exist, the audio one does not
        File.WriteAllText(Path.Combine(dir, "mulan_text.safetensors"), "x");
        File.WriteAllText(Path.Combine(dir, "mulan_text_tokenizer.json"), "{}");
        var (code, _, err) = Run("classify", "--model", dir, "--audio", Wav("ok.wav", 24000), "--vocab", Vocab("""{"g": ["a"]}"""));
        Assert.Equal(1, code);
        Assert.Contains("mulan_audio.safetensors", err);
        Assert.Contains("convert", err);
        Assert.Contains("--audio", err);
        Assert.DoesNotContain("mulan_text", err);
    }

    [Fact]
    public void Usage_ListsTheNewCommandsAndTheirOptions()
    {
        var (code, help, _) = Run("--help");
        Assert.Equal(0, code);
        foreach (var word in new[] { "embed", "classify", "--vocab", "--top", "--json", "--text", "mulan_audio.safetensors", "24 kHz" })
            Assert.Contains(word, help);
        Assert.Contains("embed, classify", Run("bogus").Err);
    }
}

/// <summary>embed and classify against the real models: an audio file in, the reference's embeddings out.</summary>
public class MulanCliRealTests : IDisposable
{
    private readonly string dir = Directory.CreateTempSubdirectory("dr2-mulancli-real-").FullName;
    private readonly ITestOutputHelper log;

    public MulanCliRealTests(ITestOutputHelper log) => this.log = log;

    public void Dispose()
    {
        Directory.Delete(dir, recursive: true);
        GC.SuppressFinalize(this);
    }

    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Cli.Cli.Run(args, o, e, new StringReader(""));
        return (code, o.ToString(), e.ToString());
    }

    private string WriteSignal(string name, string signal)
    {
        var path = Path.Combine(dir, name);
        File.WriteAllBytes(path, WavBuilder.Make(3, 1, 24000, 32, WavBuilder.Float32(MulanAudioReference.Signal(signal))));
        return path;
    }

    private static bool HasText => MulanTextReference.WeightsPath is not null && MulanTextReference.TokenizerPath is not null;

    [RequiresMulanAudio]
    public void Embed_Audio_PrintsTheReferencesEmbeddingOfThe25SecondSignal()
    {
        var (code, output, err) = Run("embed", "--model", Checkpoint.Dir!, "--audio", WriteSignal("s25.wav", "s25"));
        Assert.Equal(0, code);
        Assert.Contains("clip 3/3", err);
        using var doc = JsonDocument.Parse(output);
        var root = doc.RootElement;
        Assert.Equal("audio", root.GetProperty("kind").GetString());
        Assert.Equal(512, root.GetProperty("dim").GetInt32());
        Assert.Equal(3, root.GetProperty("clips").GetInt32());
        Assert.Equal(25.0, root.GetProperty("seconds").GetDouble(), 3);
        Assert.Equal(0.997341, root.GetProperty("length").GetDouble(), 5);
        var values = root.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        Assert.Equal(512, values.Length);
        using var got = tensor(values);
        Reference.AssertClose(log, "embed --audio (25 s)", got, MulanAudioReference.Embeddings["s25"], Tolerances.AudioEmbedding);
    }

    [RequiresMulanAudio]
    public void Embed_Audio_WithOut_WritesTheFile_AndPrintsNothing_AndLeavesNoTemporaryFile()
    {
        var target = Path.Combine(dir, "emb.json");
        var (code, output, _) = Run("embed", "--model", Checkpoint.Dir!, "--audio", WriteSignal("n.wav", "noise10"), "--out", target);
        Assert.Equal(0, code);
        Assert.Equal("", output);
        Assert.True(File.Exists(target));
        Assert.False(File.Exists(target + ".tmp"));
        using var doc = JsonDocument.Parse(File.ReadAllText(target));
        Assert.Equal(1, doc.RootElement.GetProperty("clips").GetInt32());
        Assert.Equal(1.0, doc.RootElement.GetProperty("length").GetDouble(), 5);
    }

    [Fact]
    public void Embed_Text_PrintsTheReferencesUnitEmbedding()
    {
        if (!HasText || MulanTextReference.TowerMissingReason is not null) { log.WriteLine("skipped: the converted text tower is needed"); return; }
        var (code, output, _) = Run("embed", "--model", Checkpoint.Dir!, "--text", MulanTextReference.Prompts[0]);
        Assert.Equal(0, code);
        using var doc = JsonDocument.Parse(output);
        Assert.Equal("text", doc.RootElement.GetProperty("kind").GetString());
        Assert.Equal(512, doc.RootElement.GetProperty("dim").GetInt32());
        Assert.False(doc.RootElement.TryGetProperty("clips", out _));
        var values = doc.RootElement.GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray();
        using var got = tensor(values);
        Assert.Equal(1.0, got.norm().item<float>(), 5);
        Reference.AssertClose(log, "embed --text", got, MulanTextReference.Tower["p0_emb"], Tolerances.Embedding);
    }

    [RequiresMulanAudio]
    public void Classify_RanksTheTagsOfEachCategory_AsTheLibraryDoes()
    {
        if (!HasText) { log.WriteLine("skipped: the converted text tower is needed"); return; }
        var vocab = Path.Combine(dir, "vocab.json");
        File.WriteAllText(vocab, """
            { "genre": { "template": "This is a {} song.", "tags": ["jazz", "rock", "classical", "pop"] },
              "mood": ["happy", "sad", "calm"] }
            """);
        var wav = WriteSignal("s25.wav", "s25");

        var (code, output, _) = Run("classify", "--model", Checkpoint.Dir!, "--audio", wav, "--vocab", vocab, "--top", "2");
        Assert.Equal(0, code);
        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.TrimEnd('\r')).ToList();
        Assert.Equal(2, lines.Count);
        Assert.StartsWith("genre: ", lines[0]);
        Assert.StartsWith("mood: ", lines[1]);
        foreach (var line in lines) log.WriteLine(line);
        var genre = lines[0]["genre: ".Length..].Split(", ");
        Assert.Equal(2, genre.Length);
        var scores = genre.Select(g => double.Parse(g[(g.LastIndexOf(' ') + 1)..], CultureInfo.InvariantCulture)).ToArray();
        Assert.True(scores[0] >= scores[1], "scores come in descending order");

        // the same ranking through the library
        using var encoder = MuLanAudioEncoder.LoadDirectory(Checkpoint.Dir!, CPU);
        var embedding = encoder.Embed(MulanAudioReference.Signal("s25"));
        using var style = StyleEncoder.LoadDirectory(Checkpoint.Dir!, CPU);
        var expected = TagRanker.Rank(embedding.Values, new[] { "This is a jazz song.", "This is a rock song.", "This is a classical song.", "This is a pop song." }, style, 2);
        Assert.Equal(expected.Select(r => r.Text["This is a ".Length..^" song.".Length]), genre.Select(g => g[..g.LastIndexOf(' ')]));
        Assert.True(Math.Abs(expected[0].Score - scores[0]) < 6e-5, $"the printed score {scores[0]} should be the library's {expected[0].Score} to the 4 decimals printed");

        // JSON
        var (code2, json, _) = Run("classify", "--model", Checkpoint.Dir!, "--audio", wav, "--vocab", vocab, "--top", "1", "--json");
        Assert.Equal(0, code2);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(3, doc.RootElement.GetProperty("clips").GetInt32());
        Assert.Equal(25.0, doc.RootElement.GetProperty("seconds").GetDouble(), 3);
        var best = doc.RootElement.GetProperty("categories").GetProperty("genre")[0];
        Assert.Equal(expected[0].Text, best.GetProperty("text").GetString());
        Assert.Equal(expected[0].Text["This is a ".Length..^" song.".Length], best.GetProperty("tag").GetString());
        Assert.Equal(1, doc.RootElement.GetProperty("categories").GetProperty("mood").GetArrayLength());
    }
}
