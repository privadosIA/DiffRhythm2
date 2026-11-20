using DiffRhythm2.Cli;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Text;
using MusicGen.Core.Testing;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>The command line run in-process with captured output: arguments, exit codes, messages. The tests that make a song need the models and the phonemizer library.</summary>
public class CliTests : IDisposable
{
    private readonly ITestOutputHelper log;
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-cli-").FullName;

    public CliTests(ITestOutputHelper log) => this.log = log;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    private static (int Code, string Out, string Err) Run(string[] args, string stdin = "")
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Cli.Cli.Run(args, o, e, new StringReader(stdin));
        return (code, o.ToString(), e.ToString());
    }

    private string Lyrics(string text = "[verse]\nHello world tonight\nWe sing along")
    {
        var path = Path.Combine(tmp, "lyrics.txt");
        File.WriteAllText(path, text);
        return path;
    }

    // ---- usage -------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Help_ListsTheCommandsAndEveryOption()
    {
        var (code, output, _) = Run(new[] { "--help" });
        Assert.Equal(0, code);
        foreach (var word in new[] { "generate", "check", "--model", "--lyrics", "--style", "--out", "--seconds", "--seed", "--cfg", "--steps", "--device", "--dtype", "--mono", "210", "0.5 to 10", "Exit codes" })
            Assert.Contains(word, output);
    }

    [Fact]
    public void NoArguments_PrintUsage_AndFail()
    {
        var (code, _, err) = Run(Array.Empty<string>());
        Assert.Equal(2, code);
        Assert.Contains("Usage", err);
    }

    [Fact]
    public void UnknownCommand_IsNamed_AndTheKnownOnesListed()
    {
        var (code, _, err) = Run(new[] { "frobnicate" });
        Assert.Equal(2, code);
        Assert.Contains("frobnicate", err);
        Assert.Contains("generate, check", err);
    }

    // ---- errors before any work -----------------------------------------------------------------------------------------------------------

    private (int Code, string Out, string Err) Generate(params string[] extra)
    {
        var args = new List<string> { "generate", "--model", Path.Combine(tmp, "no-models-here"), "--lyrics", Lyrics(), "--style", "Pop, Piano", "--out", Path.Combine(tmp, "out.wav") };
        args.AddRange(extra);
        return Run(args.ToArray());
    }

    [Theory]
    [InlineData("--seed", "abc", "--seed")]
    [InlineData("--seed", "-1", "--seed")]
    [InlineData("--seed", "9223372036854775808", "--seed")]
    [InlineData("--seconds", "5", "--seconds:")]
    [InlineData("--seconds", "many", "--seconds")]
    [InlineData("--cfg", "11", "--cfg:")]
    [InlineData("--steps", "2", "--steps:")]
    [InlineData("--steps", "2.5", "--steps")]
    [InlineData("--device", "tpu", "--device")]
    [InlineData("--dtype", "bf16", "--dtype")]
    [InlineData("--nonsense", "1", "--nonsense")]
    public void BadArguments_FailWithUsageCode_NamingTheOption_BeforeLoadingAnything(string option, string value, string expectedInMessage)
    {
        var (code, output, err) = Generate(option, value);
        Assert.Equal(2, code);
        Assert.Contains(expectedInMessage, err);
        Assert.DoesNotContain("seed=", output);                           // nothing started
        Assert.False(File.Exists(Path.Combine(tmp, "out.wav")));
    }

    [Fact]
    public void MissingOptions_AndMissingLyricsFile_AreNamed()
    {
        var (code, _, err) = Run(new[] { "generate", "--model", tmp });
        Assert.Equal(2, code);
        Assert.Contains("--lyrics is required", err);

        (code, _, err) = Run(new[] { "generate", "--model", tmp, "--lyrics", "/no/such/file.txt", "--style", "Pop", "--out", "x.wav" });
        Assert.Equal(2, code);
        Assert.Contains("--lyrics", err);
        Assert.Contains("does not exist", err);
    }

    [Fact]
    public void StyleAndLyricsRules_AreReportedWithTheirOptions()
    {
        var (code, _, err) = Run(new[] { "generate", "--model", tmp, "--lyrics", Lyrics(), "--style", " , ", "--out", "x.wav" });
        Assert.Equal(2, code);
        Assert.Contains("--style: style must name at least one style", err);

        (code, _, err) = Run(new[] { "generate", "--model", tmp, "--lyrics", "-", "--style", "Pop", "--out", "x.wav" }, stdin: "[verse]\n[chorus]");
        Assert.Equal(2, code);
        Assert.Contains("--lyrics", err);
    }

    [Fact]
    public void LyricsFromStandardInput_AreUsed_AndAMissingModelDirectoryFailsWithCode1_ListingAllFiles()
    {
        var (code, output, err) = Run(new[] { "generate", "--model", Path.Combine(tmp, "nope"), "--lyrics", "-", "--style", "Pop", "--out", Path.Combine(tmp, "o.wav"), "--seed", "42" }, stdin: "hello there");
        Assert.Equal(1, code);
        Assert.Contains("seed=42", output);
        foreach (var name in ModelDirectory.RequiredFiles) Assert.Contains(name, err);
        Assert.False(File.Exists(Path.Combine(tmp, "o.wav")));
    }

    [Fact]
    public void SeedOmitted_IsPickedAndPrinted()
    {
        var (_, output, _) = Generate();
        var line = output.Split('\n').First(l => l.StartsWith("seed="));
        Assert.True(long.TryParse(line["seed=".Length..].Trim(), out var seed) && seed >= 0);
    }

    // ---- check -------------------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Check_OnAnIncompleteDirectory_ListsFilesAndProblems_WithCode1()
    {
        File.WriteAllText(Path.Combine(tmp, "config.json"), "{}");
        var (code, output, _) = Run(new[] { "check", "--model", tmp });
        Assert.Equal(1, code);
        Assert.Contains("[ok]      config.json", output);
        Assert.Contains("[missing] decoder.safetensors", output);
        Assert.Contains("phonemizer:", output);
        Assert.Contains("devices:", output);
        Assert.Contains("not ready:", output);
        Assert.Contains("decoder.safetensors is missing", output);
    }

    [Fact]
    public void Check_NeedsAModelDirectory_AndAValidDevice()
    {
        Assert.Equal(2, Run(new[] { "check" }).Code);
        Assert.Equal(2, Run(new[] { "check", "--model", tmp, "--device", "tpu" }).Code);
    }

    [RequiresSongModels]
    public void Check_OnTheRealDirectory_IsReadyWhenThePhonemizerIsThere()
    {
        var (code, output, _) = Run(new[] { "check", "--model", Checkpoint.Dir! });
        log.WriteLine(output);
        Assert.Contains("[ok]      decoder.safetensors", output);
        if (EspeakEngine.TryGetShared(out _, out _)) { Assert.Equal(0, code); Assert.Contains("ready: a song can be made on cpu", output); }
        else { Assert.Equal(1, code); Assert.Contains("phonemizer: NOT FOUND", output); }
    }

    // ---- making songs ------------------------------------------------------------------------------------------------------------------

    [RequiresSongModels]
    public void Generate_WritesAStereoWav_AndTheSameSeedGivesTheSameFile()
    {
        if (!EspeakEngine.TryGetShared(out _, out var reason)) { log.WriteLine($"skipped: {reason}"); return; }
        var first = Path.Combine(tmp, "a.wav");
        var second = Path.Combine(tmp, "b.wav");
        var lyrics = Lyrics();
        foreach (var path in new[] { first, second })
        {
            var (code, output, err) = Run(new[] { "generate", "--model", Checkpoint.Dir!, "--lyrics", lyrics, "--style", "Pop, Piano, Female vocals", "--out", path, "--seconds", "10", "--seed", "7" });
            log.WriteLine(output);
            Assert.True(code == 0, err);
            Assert.Contains("seed=7", output);
            Assert.Contains("block 5/5", err);
        }
        var wav = WavFile.Read(first);
        Assert.Equal((2, 48000u, 16), ((int)wav.Channels, wav.SampleRate, (int)wav.Bits));
        Assert.True(wav.Shorts().Length / 2 % 9600 == 0 && wav.Shorts().Length / 2 >= 9600 * 10);
        Assert.Equal(File.ReadAllBytes(first), File.ReadAllBytes(second));
        Assert.Empty(Directory.GetFiles(tmp, "*.tmp"));
    }

    [RequiresSongModels]
    public void Generate_ThatFailsAfterSampling_LeavesNoFile()
    {
        if (!EspeakEngine.TryGetShared(out _, out var reason)) { log.WriteLine($"skipped: {reason}"); return; }
        // the same models, but a decoder configuration the port refuses: it fails when the decoder is built, after the network has sampled
        var models = Path.Combine(tmp, "models");
        Directory.CreateDirectory(models);
        foreach (var name in ModelDirectory.RequiredFiles.Where(n => n != "decoder.json"))
            File.CreateSymbolicLink(Path.Combine(models, name), Path.Combine(Checkpoint.Dir!, name));
        File.WriteAllText(Path.Combine(models, "decoder.json"), File.ReadAllText(Path.Combine(Checkpoint.Dir!, "decoder.json")).Replace("\"snakebeta\"", "\"snake\""));
        var outPath = Path.Combine(tmp, "never.wav");
        var (code, _, err) = Run(new[] { "generate", "--model", models, "--lyrics", Lyrics(), "--style", "Pop", "--out", outPath, "--seconds", "10", "--seed", "3" });
        Assert.Equal(1, code);
        Assert.Contains("snake", err);
        Assert.False(File.Exists(outPath));
        Assert.Empty(Directory.GetFiles(tmp, "*.tmp"));
    }
}
