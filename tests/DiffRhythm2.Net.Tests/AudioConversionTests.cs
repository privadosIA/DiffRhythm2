using DiffRhythm2.Weights;

namespace DiffRhythm2.Tests;

/// <summary>
/// The conversion of the audio tower (change add-muq-mulan-audio) on the small archives of fixtures/torch_archive: the key filter and rename of the real profile,
/// the dtype rule (float32 except the int64 batch-norm counters), the counts, and the wiring of the <c>Audio</c> option. The real checkpoint is in RealWeightsTests.
/// </summary>
public class AudioConversionTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "dr2-audio-weights-" + Guid.NewGuid().ToString("N"));

    public AudioConversionTests() => Directory.CreateDirectory(dir);

    public void Dispose()
    {
        try { Directory.Delete(dir, true); } catch (IOException) { }
        GC.SuppressFinalize(this);
    }

    private void Place(string fixture, string asName)
    {
        File.Copy(TorchArchiveTests.Of(fixture), Path.Combine(dir, asName));
    }

    // plain.pt holds a.weight (F32), b.bias (F32), ids (int64) among others: ids plays the part of a batch-norm counter
    private static readonly Func<string, string> CounterRule = n => n == "ids" ? "I64" : "F32";
    private static WeightConverter.Profile Tiny(int expected = 3, int int64 = 1, Func<string, string>? rule = null) =>
        new("audio.safetensors", "decoder.bin", null, n => n is "a.weight" or "b.bias" or "ids" ? n : null, expected, "the audio file", rule ?? CounterRule, int64);

    private ConversionResult Run(WeightConverter.Profile p, ConversionOptions? o = null) =>
        WeightConverter.ConvertTensors(dir, p, o ?? new ConversionOptions(), new List<string>(), default);

    [Fact]
    public void TheAudioProfile_KeepsTheAudioTensors_WithoutTheLeadingMulan_AndNothingElse()
    {
        var map = WeightConverter.Audio.Map;
        Assert.Equal("audio.proj.weight", map("mulan.audio.proj.weight"));
        Assert.Equal("audio.model.model.conformer.layers.3.self_attn.linear_q.bias", map("mulan.audio.model.model.conformer.layers.3.self_attn.linear_q.bias"));
        Assert.Equal("audio.aggregator.bias", map("mulan.audio.aggregator.bias"));
        Assert.Equal("audio_to_latents.weight", map("mulan.audio_to_latents.weight"));
        Assert.Null(map("mulan.text.model.embeddings.word_embeddings.weight"));
        Assert.Null(map("mulan.text_to_latents.weight"));
        Assert.Null(map("mulan.contrast.temperatures"));
        Assert.Null(map("audio.proj.weight"));
    }

    [Fact]
    public void TheAudioProfile_Expects531Tensors_18OfThemInt64Counters()
    {
        var p = WeightConverter.Audio;
        Assert.Equal(531, p.Expected);
        Assert.Equal(18, p.ExpectedInt64);
        Assert.Equal("mulan_audio.safetensors", p.Output);
        Assert.Equal(WeightConverter.AudioFile, p.Output);
        Assert.Equal("pytorch_model.bin", p.Source);
        Assert.Equal("I64", p.DTypeOf("audio.model.model.conformer.layers.0.conv_module.batch_norm.num_batches_tracked"));
        Assert.Equal("I64", p.DTypeOf("audio.model.model.conv.conv.0.bn1.num_batches_tracked"));
        Assert.Equal("F32", p.DTypeOf("audio.model.model.conformer.layers.0.conv_module.batch_norm.running_mean"));
        Assert.Equal("F32", p.DTypeOf("audio.proj.weight"));
    }

    [Fact]
    public void TheCounters_AreKeptAsInt64_AndTheValuesCopiedExactly()
    {
        Place("plain.pt", "decoder.bin");
        var r = Run(Tiny());
        Assert.Equal(ConversionStatus.Written, r.Status);
        var path = Path.Combine(dir, "audio.safetensors");
        var h = SafeTensorsHeader.Read(path);
        // the Python library puts the wider type first (int64 before float32), each group sorted by name
        Assert.Equal(new[] { "ids", "a.weight", "b.bias" }, h.Entries.Select(e => e.Name));
        Assert.Equal(new[] { "I64", "F32", "F32" }, h.Entries.Select(e => e.DType));
        using var archive = new TorchArchive(Path.Combine(dir, "decoder.bin"), null);
        var ids = archive.Tensors.Single(t => t.Name == "ids");
        var expected = new MemoryStream();
        archive.CopyTo(ids, expected);
        var entry = h.Entries.Single(e => e.Name == "ids");
        using var f = File.OpenRead(path);
        f.Position = h.DataStart + entry.Start;
        var got = new byte[entry.End - entry.Start];
        f.ReadExactly(got);
        Assert.Equal(expected.ToArray(), got);
    }

    [Fact]
    public void AWrongNumberOfCounters_SaysWhatWasExpectedAndFound_AndWritesNothing()
    {
        Place("plain.pt", "decoder.bin");
        var e = Assert.Throws<ConversionException>(() => Run(Tiny(int64: 2)));
        Assert.Contains("1 int64 counters", e.Message);
        Assert.Contains("2 were expected", e.Message);
        Assert.Empty(Directory.GetFiles(dir, "*.safetensors*"));
    }

    [Fact]
    public void ATensorOfTheWrongType_IsRefusedNamingIt_BothWays()
    {
        Place("plain.pt", "decoder.bin");
        // an int64 where float32 is required
        var e = Assert.Throws<ConversionException>(() => Run(Tiny(rule: _ => "F32", int64: 0)));
        Assert.Contains("'ids'", e.Message);
        Assert.Contains("I64", e.Message);
        Assert.Contains("expected F32", e.Message);
        // a float32 where the counter rule asks for int64
        var e2 = Assert.Throws<ConversionException>(() => Run(Tiny(rule: n => n == "a.weight" ? "I64" : "F32", int64: 1)));
        Assert.Contains("'a.weight'", e2.Message);
        Assert.Contains("expected I64", e2.Message);
        Assert.Empty(Directory.GetFiles(dir, "*.safetensors*"));
    }

    [Fact]
    public void AWrongTensorCount_SaysWhatWasExpectedAndFound()
    {
        Place("plain.pt", "decoder.bin");
        var e = Assert.Throws<ConversionException>(() => Run(Tiny(expected: 531)));
        Assert.Contains("3 tensors", e.Message);
        Assert.Contains("531", e.Message);
    }

    [Fact]
    public void AValidExistingOutput_IsUpToDate_AndAnotherDTypeIsNotValid()
    {
        Place("plain.pt", "decoder.bin");
        Run(Tiny());
        Assert.Equal(ConversionStatus.UpToDate, Run(Tiny()).Status);
        // the same file judged by a rule that wants every tensor in float32 is not a valid output: it is rewritten
        var all32 = Tiny(rule: _ => "F32", int64: 0);
        Assert.Throws<ConversionException>(() => Run(all32));   // the source's int64 is refused, so the existing file was not accepted as up to date
    }

    [Fact]
    public void WithoutTheAudioOption_NoAudioFileIsRequiredOrWritten_AndWithItTheAudioStepRuns()
    {
        // valid-looking outputs for the three usual files, so Convert goes straight to what is asked for
        Dummy("decoder.safetensors", 783);
        Dummy("mulan_text.safetensors", 273);
        File.WriteAllText(Path.Combine(dir, "mulan_text_tokenizer.json"), "{}");

        var plain = WeightConverter.Convert(dir);
        Assert.Equal(new[] { "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json" }, plain.Select(r => r.Output));
        Assert.All(plain, r => Assert.Equal(ConversionStatus.UpToDate, r.Status));
        Assert.False(File.Exists(Path.Combine(dir, WeightConverter.AudioFile)));

        var steps = new List<string>();
        var e = Assert.Throws<ConversionException>(() => WeightConverter.Convert(dir, new ConversionOptions(Audio: true), steps.Add));
        Assert.Equal(new[] { "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json", "mulan_audio.safetensors" }, steps);
        Assert.Contains("pytorch_model.bin was not found", e.Message);
        Assert.False(File.Exists(Path.Combine(dir, WeightConverter.AudioFile)));
    }

    [Fact]
    public void AValidAudioOutput_NeedsNoSource_AndIsLeftAlone()
    {
        Dummy("decoder.safetensors", 783);
        Dummy("mulan_text.safetensors", 273);
        File.WriteAllText(Path.Combine(dir, "mulan_text_tokenizer.json"), "{}");
        Dummy(WeightConverter.AudioFile, 531, counters: 18);
        var results = WeightConverter.Convert(dir, new ConversionOptions(Audio: true));
        Assert.Equal(4, results.Count);
        Assert.All(results, r => Assert.Equal(ConversionStatus.UpToDate, r.Status));
    }

    [Fact]
    public void TheManifest_AsksForTheMuLanCheckpoint_ForTheAudioTowerOnlyWhenAudioIsRequested()
    {
        Dummy("decoder.safetensors", 783);
        Dummy("mulan_text.safetensors", 273);
        File.WriteAllText(Path.Combine(dir, "mulan_text_tokenizer.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "config.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "decoder.json"), "{}");
        File.WriteAllText(Path.Combine(dir, "model.safetensors"), "x");
        Assert.DoesNotContain(ModelManifest.Needed(dir), f => f.Target == "sources/pytorch_model.bin");
        Assert.Contains(ModelManifest.Needed(dir, audio: true), f => f.Target == "sources/pytorch_model.bin");
        Dummy(WeightConverter.AudioFile, 531, counters: 18);
        Assert.DoesNotContain(ModelManifest.Needed(dir, audio: true), f => f.Target == "sources/pytorch_model.bin");
    }

    /// <summary>A safetensors file with <paramref name="count"/> tiny tensors (the first <paramref name="counters"/> of them int64 counters named like the real ones).</summary>
    private void Dummy(string name, int count, int counters = 0)
    {
        var entries = new List<SafeTensorsEntry>();
        for (int i = 0; i < count; i++)
        {
            bool counter = i < counters;
            string tensor = counter ? $"audio.model.model.conformer.layers.{i}.conv_module.batch_norm.num_batches_tracked" : $"t{i:D4}";
            entries.Add(new SafeTensorsEntry(tensor, counter ? TorchDType.Int64 : TorchDType.Float32, new long[] { 1 }, counter ? 8 : 4, s => s.Write(new byte[counter ? 8 : 4])));
        }
        SafeTensorsStreamWriter.Write(Path.Combine(dir, name), entries);
    }

    // ---- the commands ------------------------------------------------------------------------------------------------------------

    private static (int Code, string Out, string Err) RunCli(params string[] args)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Cli.Cli.Run(args, o, e, new StringReader(""));
        return (code, o.ToString(), e.ToString());
    }

    private void CompleteDirectory()
    {
        Dummy("decoder.safetensors", 783);
        Dummy("mulan_text.safetensors", 273);
        foreach (var f in new[] { "mulan_text_tokenizer.json", "config.json", "decoder.json", "model.safetensors" }) File.WriteAllText(Path.Combine(dir, f), "{}");
    }

    [Fact]
    public void Convert_WithAudio_PrintsAStepForEachOutput_AndLeavesTheAudioFileAlone_WhenItIsValid()
    {
        CompleteDirectory();
        Dummy(WeightConverter.AudioFile, 531, counters: 18);
        var (code, output, _) = RunCli("convert", "--model", dir, "--audio");
        Assert.Equal(0, code);
        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.TrimEnd('\r')).ToList();
        Assert.Contains("step: mulan_audio.safetensors", lines);
        Assert.Contains("  mulan_audio.safetensors: up to date", lines);
        Assert.Equal("done", lines[^1]);
    }

    [Fact]
    public void Convert_WithoutAudio_NeverMentionsOrWritesTheAudioFile()
    {
        CompleteDirectory();
        var (code, output, _) = RunCli("convert", "--model", dir);
        Assert.Equal(0, code);
        Assert.DoesNotContain("mulan_audio", output);
        Assert.False(File.Exists(Path.Combine(dir, WeightConverter.AudioFile)));
    }

    [Fact]
    public void Convert_WithAudio_ButNoCheckpoint_FailsListingThePlaces_AndWritesNoAudioFile()
    {
        CompleteDirectory();
        var (code, _, err) = RunCli("convert", "--model", dir, "--audio");
        Assert.Equal(1, code);
        Assert.Contains("pytorch_model.bin was not found", err);
        Assert.False(File.Exists(Path.Combine(dir, WeightConverter.AudioFile)));
    }

    [Fact]
    public void Install_WithAudio_EndsReadyOnlyWhenTheAudioFileIsThere()
    {
        Cli.Cli.ManifestOverride = Array.Empty<ManifestFile>();
        try
        {
            CompleteDirectory();
            Dummy(WeightConverter.AudioFile, 531, counters: 18);
            var (code, output, _) = RunCli("install", "--model", dir, "--audio", "--accept-licenses");
            Assert.Equal(0, code);
            Assert.EndsWith("ready", output.TrimEnd());

            // without the option the audio file is not even looked at
            File.Delete(Path.Combine(dir, WeightConverter.AudioFile));
            var (code2, output2, _) = RunCli("install", "--model", dir, "--accept-licenses");
            Assert.Equal(0, code2);
            Assert.EndsWith("ready", output2.TrimEnd());
        }
        finally { Cli.Cli.ManifestOverride = null; }
    }

    [Fact]
    public void Fetch_DoesNotAcceptTheAudioOption_AndTheUsageDescribesIt()
    {
        var (code, _, err) = RunCli("fetch", "--model", dir, "--audio", "--accept-licenses");
        Assert.Equal(2, code);
        Assert.Contains("--audio", err);
        var (_, help, _) = RunCli("--help");
        Assert.Contains("--audio", help);
        Assert.Contains("mulan_audio.safetensors", help);
    }
}
