using DiffRhythm2.AudioTower;
using DiffRhythm2.StyleText;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The embedding of clips and signals, and the ranking, on a tiny random tower. No checkpoint needed.</summary>
public class MuLanAudioEmbeddingModelTests : IDisposable
{
    // 128 bands, as the front end produces; everything else tiny
    private static readonly MuQConfig TinyConfig = new(Bands: 128, ConvDim: 4, Hidden: 16, Layers: 1, Heads: 2, Intermediate: 32, ConvKernel: 5);
    private readonly MuLanAudioEncoder encoder;

    public MuLanAudioEmbeddingModelTests()
    {
        manual_seed(21);
        var tower = new MuLanAudioTower(TinyConfig, dim: 12, latentDim: 6);
        using (no_grad())
        {
            foreach (var p in tower.parameters()) p.normal_(0, 0.5);
            foreach (var (name, b) in tower.named_buffers())
                if (name.EndsWith("running_var", StringComparison.Ordinal)) b.fill_(1.0);
                else if (name.EndsWith("running_mean", StringComparison.Ordinal)) b.zero_();
        }
        encoder = new MuLanAudioEncoder(tower, CPU);
    }

    public void Dispose()
    {
        encoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private static float[] Noise(int samples, int seed)
    {
        var rnd = new Random(seed);
        return Enumerable.Range(0, samples).Select(_ => (float)(rnd.NextDouble() * 0.4 - 0.2)).ToArray();
    }

    [Fact]
    public void AClipEmbedding_HasUnitLength_AndTheLatentWidth()
    {
        using var e = encoder.EmbedClip(Noise(AudioClips.ClipSamples, 1));
        Assert.Equal(new long[] { 1, 6 }, e.shape);
        Assert.Equal(1.0, e.norm().item<float>(), 5);
    }

    [Fact]
    public void ASignalEmbedding_IsThePlainMeanOfItsClipEmbeddings_NotRenormalized()
    {
        // three very different clips: loud noise, silence, a loud tone
        var signal = Noise(10 * 24000, 2).Concat(new float[10 * 24000]).Concat(Enumerable.Range(0, 5 * 24000).Select(i => 0.5f * (float)Math.Sin(2 * Math.PI * 440 * i / 24000.0))).ToArray();
        var clips = AudioClips.Split(signal);
        Assert.Equal(3, clips.Count);
        var expected = new float[6];
        foreach (var clip in clips)
        {
            using var e = encoder.EmbedClip(clip);
            var v = e.data<float>().ToArray();
            for (int i = 0; i < 6; i++) expected[i] += v[i] / 3f;
        }
        var got = encoder.Embed(signal);
        Assert.Equal(3, got.Clips);
        Assert.Equal(25.0, got.Seconds, 9);
        for (int i = 0; i < 6; i++) Assert.Equal(expected[i], got.Values[i], 5);
        // not renormalized: its length is the plain mean's, and below 1 because the clip embeddings differ
        Assert.Equal(Math.Sqrt(expected.Sum(v => (double)v * v)), got.Length, 6);
        Assert.True(got.Length < 1.0 - 1e-6, $"the mean of different unit vectors is shorter than 1, got {got.Length}");
    }

    [Fact]
    public void ASingleClipSignal_KeepsUnitLength()
    {
        var got = encoder.Embed(Noise(AudioClips.ClipSamples, 3));
        Assert.Equal(1, got.Clips);
        Assert.Equal(1.0, got.Length, 5);
    }

    [Fact]
    public void ARepeatedSignal_GivesTheSameEmbedding()
    {
        var signal = Noise(12 * 24000, 4);
        Assert.Equal(encoder.Embed(signal).Values, encoder.Embed(signal).Values);
    }

    [Fact]
    public void AShortSignal_IsEmbeddedFromItsEightSecondClip_AndATooShortOneRefused()
    {
        var four = encoder.Embed(Noise(4 * 24000, 5));
        Assert.Equal(1, four.Clips);
        Assert.Equal(4.0, four.Seconds, 9);
        Assert.Equal(1.0, four.Length, 5);
        Assert.Throws<ArgumentException>(() => encoder.Embed(Noise(500, 6)));
    }

    [Fact]
    public void Progress_IsReportedAfterEachClip_AndCancellationStopsBetweenClips()
    {
        var signal = Noise(25 * 24000, 7);
        var seen = new List<(int Done, int Total)>();
        encoder.Embed(signal, (done, total) => seen.Add((done, total)));
        Assert.Equal(new[] { (1, 3), (2, 3), (3, 3) }, seen);

        using var cts = new CancellationTokenSource();
        int calls = 0;
        Assert.Throws<OperationCanceledException>(() => encoder.Embed(signal, (_, _) => { if (++calls == 1) cts.Cancel(); }, cts.Token));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void DifferentAudio_GivesDifferentEmbeddings()
    {
        var a = encoder.Embed(Noise(10 * 24000, 8)).Values;
        var b = encoder.Embed(Noise(10 * 24000, 9)).Values;
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void LoadDirectory_WithoutTheAudioFile_SaysHowToCreateIt()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-noaudio-").FullName;
        try
        {
            var e = Assert.Throws<FileNotFoundException>(() => MuLanAudioEncoder.LoadDirectory(dir, CPU));
            Assert.Contains("mulan_audio.safetensors", e.Message);
            Assert.Contains("convert", e.Message);
            Assert.Contains("--audio", e.Message);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // ---- similarity and ranking ---------------------------------------------------------------------------------------------------

    [Fact]
    public void Dot_IsTheSumOfProducts_AndRefusesDifferentLengths()
    {
        Assert.Equal(11f, TagRanker.Dot(new[] { 1f, 2f, 3f }, new[] { 2f, 3f, 1f }) + 0f - 0f, 5);
        Assert.Equal(1f, TagRanker.Dot(new[] { 0.6f, 0.8f }, new[] { 0.6f, 0.8f }), 6);
        Assert.Throws<ArgumentException>(() => TagRanker.Dot(new[] { 1f, 2f }, new[] { 1f }));
    }

    private static float[] Fake(string text) => text switch
    {
        "jazz" => new[] { 1f, 0f },
        "rock" => new[] { 0f, 1f },
        "blues" => new[] { 0.6f, 0.8f },
        "swing" => new[] { 1f, 0f },          // ties with jazz
        "far" => new[] { -1f, 0f },
        "toolong" => throw new ArgumentException("the text has 600 positions but the encoder supports at most 512"),
        _ => throw new ArgumentException("unknown"),
    };

    [Fact]
    public void Rank_OrdersByDotProduct_HighestFirst()
    {
        var ranked = TagRanker.Rank(new[] { 0.6f, 0.8f }, new[] { "jazz", "rock", "blues", "far" }, Fake);
        Assert.Equal(new[] { "blues", "rock", "jazz", "far" }, ranked.Select(r => r.Text));
        Assert.Equal(1f, ranked[0].Score, 5);
        Assert.Equal(0.8f, ranked[1].Score, 5);
        Assert.Equal(0.6f, ranked[2].Score, 5);
        Assert.Equal(-0.6f, ranked[3].Score, 5);
    }

    [Fact]
    public void Rank_Top_KeepsOnlyTheBest_InOrder()
    {
        var ranked = TagRanker.Rank(new[] { 0.6f, 0.8f }, new[] { "jazz", "rock", "blues", "far" }, Fake, top: 2);
        Assert.Equal(new[] { "blues", "rock" }, ranked.Select(r => r.Text));
        Assert.Equal(4, TagRanker.Rank(new[] { 0.6f, 0.8f }, new[] { "jazz", "rock", "blues", "far" }, Fake, top: 10).Count);
        Assert.Throws<ArgumentOutOfRangeException>(() => TagRanker.Rank(new[] { 1f, 0f }, new[] { "jazz" }, Fake, top: 0));
    }

    [Fact]
    public void Rank_EqualScores_KeepTheOrderOfTheList()
    {
        Assert.Equal(new[] { "jazz", "swing" }, TagRanker.Rank(new[] { 1f, 0f }, new[] { "jazz", "swing" }, Fake).Select(r => r.Text));
        Assert.Equal(new[] { "swing", "jazz" }, TagRanker.Rank(new[] { 1f, 0f }, new[] { "swing", "jazz" }, Fake).Select(r => r.Text));
    }

    [Fact]
    public void Rank_AnUnembeddableText_FailsNamingIt()
    {
        var e = Assert.Throws<ArgumentException>(() => TagRanker.Rank(new[] { 1f, 0f }, new[] { "jazz", "toolong" }, Fake));
        Assert.Contains("'toolong'", e.Message);
        Assert.Contains("600 positions", e.Message);
        var longText = new string('x', 200);
        var e2 = Assert.Throws<ArgumentException>(() => TagRanker.Rank(new[] { 1f, 0f }, new[] { longText }, _ => throw new ArgumentException("nope")));
        Assert.Contains(new string('x', 60) + "...", e2.Message);
        Assert.DoesNotContain(longText, e2.Message);
    }

    [Fact]
    public void Rank_AnEmptyList_GivesNothing()
    {
        Assert.Empty(TagRanker.Rank(new[] { 1f }, Array.Empty<string>(), _ => new[] { 1f }));
    }
}

[CollectionDefinition(Name)]
public sealed class RealMulanBothCollection :
    ICollectionFixture<RealMulanFixture>,
    ICollectionFixture<RealMulanAudioFixture>
{
    public const string Name = "RealMulanBoth";
}

/// <summary>The audio embedding and the similarity with text against the reference, stage by stage and end to end.</summary>
[Collection(RealMulanBothCollection.Name)]
public class MuLanAudioEmbeddingParityTests
{
    private readonly RealMulanFixture text;
    private readonly RealMulanAudioFixture audio;
    private readonly ITestOutputHelper log;

    public MuLanAudioEmbeddingParityTests(RealMulanFixture text, RealMulanAudioFixture audio, ITestOutputHelper log)
    {
        this.text = text;
        this.audio = audio;
        this.log = log;
    }

    private static Tensor Stage(string name) => MulanAudioReference.Stages[name];

    private static Tensor Emb(string name) => MulanAudioReference.Embeddings[name];

    private MuLanAudioTower Tower => audio.Tower!;

    private MuLanAudioEncoder Encoder() => new(audio.Tower!, CPU, ownsTower: false);

    [RequiresMulanAudio]
    public void Stages_ProjectionMeanAndLatent_MatchTheReference_EachFedTheReferencesPreviousStage()
    {
        using var _ = no_grad();
        using var last = Stage("last").unsqueeze(0);
        using var proj = Tower.Audio.Project(last);
        Reference.AssertClose(log, "projection 1024 -> 768", proj[0], Stage("proj"), Tolerances.AudioEmbedding);
        using var refProj = Stage("proj").unsqueeze(0);
        using var mean = MuLanAudioTower.MeanOverTime(refProj);
        Reference.AssertClose(log, "mean over time", mean[0], Stage("mean"), Tolerances.AudioEmbedding);
        using var refMean = Stage("mean").unsqueeze(0);
        using var latent = Tower.ToLatent(refMean);
        Reference.AssertClose(log, "latent 768 -> 512", latent[0], Stage("latent"), Tolerances.AudioEmbedding);
        using var unit = nn.functional.normalize(Stage("latent").unsqueeze(0), 2.0, -1);
        Reference.AssertClose(log, "L2 normalization", unit[0], Stage("emb"), Tolerances.AudioEmbedding);
    }

    [RequiresMulanAudio]
    public void TheFirstClipOfS25_MatchesTheReference_FromTheNormalizedSpectrogram()
    {
        using var _ = no_grad();
        using var features = MulanAudioReference.Frontend["s25_norm"].unsqueeze(0);
        using var got = Tower.Embed(features);
        Reference.AssertClose(log, "s25 clip 0 embedding", got[0], Stage("emb"), Tolerances.AudioEmbedding);
        Assert.Equal(1.0, got.norm().item<float>(), 5);
    }

    [RequiresMulanAudio]
    public void ClipEmbeddings_FromTheRawSamples_MatchTheReference()
    {
        using var enc = Encoder();
        foreach (var (name, count) in new[] { ("s4", 1), ("s25", 3), ("noise10", 1) })
        {
            var clips = AudioClips.Split(MulanAudioReference.Signal(name));
            Assert.Equal(count, clips.Count);
            for (int i = 0; i < count; i++)
            {
                using var got = enc.EmbedClip(clips[i]);
                Reference.AssertClose(log, $"{name} clip {i}", got[0], Emb($"{name}_clip{i}"), Tolerances.AudioEmbedding);
            }
        }
    }

    [RequiresMulanAudio]
    public void SignalEmbeddings_MatchTheReference_AndAMultiClipMeanIsShorterThanOne()
    {
        using var enc = Encoder();
        foreach (var name in new[] { "s4", "s25", "noise10" })
        {
            var got = enc.Embed(MulanAudioReference.Signal(name));
            using var gotT = tensor(got.Values);
            Reference.AssertClose(log, $"{name} signal embedding", gotT, Emb(name), Tolerances.AudioEmbedding);
            double refLength = Emb($"{name}_length").item<float>();
            log.WriteLine($"{name}: length {got.Length:F6} (reference {refLength:F6}), {got.Clips} clip(s), {got.Seconds:F1} s");
            Assert.Equal(refLength, got.Length, 5);
        }
        var s25 = enc.Embed(MulanAudioReference.Signal("s25"));
        Assert.Equal(3, s25.Clips);
        Assert.True(s25.Length < 1.0 - 1e-4, "the mean of three different clip embeddings is shorter than 1");
    }

    [RequiresMulanAudio]
    public void AudioAgainstAudio_IsTheDotProduct_OfTheReferencesSimilarity()
    {
        using var enc = Encoder();
        var names = new[] { "s4", "s25", "noise10" };
        var embeddings = names.Select(n => enc.Embed(MulanAudioReference.Signal(n)).Values).ToArray();
        var reference = Emb("audio_audio");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                float got = TagRanker.Dot(embeddings[i], embeddings[j]);
                float expected = reference[i, j].item<float>();
                log.WriteLine($"{names[i]} . {names[j]}: {got:F7} (reference {expected:F7})");
                Assert.True(Math.Abs(got - expected) < Tolerances.Similarity, $"{names[i]}.{names[j]}: {got} vs {expected}");
            }
    }

    [Fact]
    public void RankingTheReferencesPrompts_GivesTheReferencesSimilarities_AndOrder()
    {
        if (audio.Tower is null || text.Encoder is null) { log.WriteLine("skipped: the converted audio and text towers are needed"); return; }
        using var enc = Encoder();
        var prompts = MulanAudioReference.Meta.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();
        var names = new[] { "s4", "s25", "noise10" };
        var simOne = Emb("sim_one");
        for (int a = 0; a < names.Length; a++)
        {
            var embedding = enc.Embed(MulanAudioReference.Signal(names[a])).Values;
            var ranked = TagRanker.Rank(embedding, prompts, text.Encoder);
            var expected = Enumerable.Range(0, prompts.Length).Select(i => (Text: prompts[i], Score: simOne[a, i].item<float>())).OrderByDescending(x => x.Score).ToList();
            log.WriteLine($"{names[a]}: best '{ranked[0].Text}' {ranked[0].Score:F6} (reference '{expected[0].Text}' {expected[0].Score:F6})");
            Assert.Equal(expected.Select(x => x.Text), ranked.Select(r => r.Text));
            for (int i = 0; i < ranked.Count; i++)
                Assert.True(Math.Abs(ranked[i].Score - expected[i].Score) < Tolerances.Similarity, $"{names[a]} rank {i}: {ranked[i].Score} vs {expected[i].Score}");
        }
    }

    [RequiresMulanFixturesFact]
    public void TheDifferenceBetweenOneByOneAndPaddedBatchTextEmbeddings_IsMeasured()
    {
        // informational (design D7): the owner's earlier study embedded all tags of a category as one padded batch
        var one = Emb("text_one");
        var batch = Emb("text_batch");
        var worst = new List<string>();
        for (int i = 0; i < one.shape[0]; i++)
        {
            double d = (one[i] - batch[i]).abs().max().item<float>();
            double cos = (one[i] * batch[i]).sum().item<float>();
            worst.Add($"prompt {i}: max abs difference {d:E2}, cosine {cos:F6}");
        }
        foreach (var line in worst) log.WriteLine(line);
        var simDelta = (Emb("sim_one") - Emb("sim_batch")).abs().max().item<float>();
        log.WriteLine($"largest change of a similarity score: {simDelta:E2}");
        Assert.True(simDelta >= 0f);
    }

    [RequiresMulanAudio]
    public void Timing_AndMemory_OfEmbeddingA25SecondSignal_AreMeasured()
    {
        // informational (docs/mulan-audio.md): run on an idle machine with
        //   DIFFRHYTHM2_CKPT=DIR dotnet test -c Release --filter "FullyQualifiedName~Timing_AndMemory" --logger "console;verbosity=detailed"
        using var proc = System.Diagnostics.Process.GetCurrentProcess();
        proc.Refresh();
        long afterLoad = proc.WorkingSet64;
        var signal = MulanAudioReference.Signal("s25");
        long peak = afterLoad;
        using var stop = new CancellationTokenSource();
        var monitor = Task.Run(() => { while (!stop.IsCancellationRequested) { proc.Refresh(); peak = Math.Max(peak, proc.WorkingSet64); Thread.Sleep(20); } });
        using var enc = Encoder();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var first = enc.Embed(signal);
        double firstSeconds = sw.Elapsed.TotalSeconds;
        sw.Restart();
        var second = enc.Embed(signal);
        double secondSeconds = sw.Elapsed.TotalSeconds;
        sw.Restart();
        using var front = new MelFrontEnd();
        using var features = front.Features(AudioClips.Split(signal)[0]);
        double frontSeconds = sw.Elapsed.TotalSeconds;
        stop.Cancel();
        monitor.Wait();
        log.WriteLine($"threads: {torch.get_num_threads()}");
        log.WriteLine($"working set with the tower loaded: {afterLoad / 1048576.0:F0} MiB; peak while embedding 25 s: {peak / 1048576.0:F0} MiB (+{(peak - afterLoad) / 1048576.0:F0} MiB)");
        log.WriteLine($"25 s (3 clips): first call {firstSeconds:F2} s, second call {secondSeconds:F2} s; the front end of one clip alone {frontSeconds * 1000:F0} ms");
        Assert.Equal(first.Values, second.Values);
    }
}
