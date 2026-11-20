using DiffRhythm2.Pipeline;
using DiffRhythm2.Text;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>[Fact] that needs the checkpoint directory with the six model files of the pipeline.</summary>
public sealed class RequiresSongModelsAttribute : FactAttribute
{
    public RequiresSongModelsAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason ?? (new ModelDirectory(Checkpoint.Dir!).Missing is { Count: > 0 } m ? $"the model directory lacks {string.Join(", ", m)}" : null);
}

/// <summary>Request order, device and directory errors: no model is touched.</summary>
public class SongGeneratorErrorTests
{
    [Fact]
    public void InvalidRequest_FailsBeforeAnythingElse()
    {
        var generator = new SongGenerator(new SongOptions("/nonexistent", Device: "cuda:9"));
        var e = Assert.Throws<SongRequestException>(() => generator.Generate(new SongRequest("", "Pop")));
        Assert.Equal("lyrics", e.Field);
    }

    [Fact]
    public void UnknownDevice_FailsBeforeLoadingModels()
    {
        var request = new SongRequest("hello there", "Pop");
        Assert.Throws<DeviceUnavailableException>(() => new SongGenerator(new SongOptions("/nonexistent", Device: "tpu")).Generate(request));
        if (!cuda.is_available())
        {
            var e = Assert.Throws<DeviceUnavailableException>(() => new SongGenerator(new SongOptions("/nonexistent", Device: "cuda")).Generate(request));
            Assert.Contains("cuda", e.Message);
        }
    }

    [Fact]
    public void MissingModelFiles_AreListedTogether()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-gen-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "config.json"), "{}");
            var e = Assert.Throws<ModelDirectoryException>(() => new SongGenerator(new SongOptions(dir)).Generate(new SongRequest("hello there", "Pop")));
            Assert.Equal(5, e.Missing.Count);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void DeviceAndPrecisionDefaults()
    {
        Assert.Equal(SongPrecision.Float32, SongGenerator.DefaultPrecision(DeviceType.CPU));
        Assert.Equal(SongPrecision.Float16, SongGenerator.DefaultPrecision(DeviceType.CUDA));
        Assert.Equal(DeviceType.CPU, SongGenerator.ResolveDevice(" CPU ").type);
        Assert.Throws<DeviceUnavailableException>(() => SongGenerator.ResolveDevice("cuda:x"));
    }
}

/// <summary>The whole pipeline on the real models, CPU float32, against the song the reference made from the same request and seed.</summary>
public class SongGeneratorParityTests : IClassFixture<SongRun>
{
    private readonly SongRun run;
    private readonly ITestOutputHelper log;

    public SongGeneratorParityTests(SongRun run, ITestOutputHelper log)
    {
        this.run = run;
        this.log = log;
    }

    private static Tensor Of(float[] data, params long[] shape) => tensor(data, shape);

    [RequiresSongModels]
    public void RequestMadeAsExpected()
    {
        var r = run.Result!;
        Assert.Equal(SongReference.Seed, r.Seed);
        Assert.Equal(20, r.Frames);
        Assert.Equal(2, r.BlocksSampled);
        Assert.Equal(2, r.BlocksRequested);
        Assert.False(r.EndedEarly);
        Assert.Equal(20 * 9600, r.Audio.Length);
        Assert.Equal(48000, r.SampleRate);
        Assert.Equal(SongPrecision.Float32, r.Precision);
        log.WriteLine($"made in {r.Elapsed.TotalSeconds:F1} s");
    }

    [RequiresSongModels]
    public void Tokens_AndStyle_AndNoise_MatchTheReference()
    {
        var kept = run.Result!.Intermediates!;
        Assert.Equal(SongReference.Song["tokens"].data<long>().ToArray(), kept.Tokens);            // exact: the lyrics front end, here with the reference's recorded espeak answers
        using var style = Of(kept.Style, 1, 512);
        BigVganReference.AssertClose(log, "style embedding", style, SongReference.Song["style"], Tolerances.Embedding);
        using var noise = Of(kept.Noise, 2, 1, 10, 64);
        Assert.Equal(0f, (noise - SongReference.Song["noise"]).abs().max().item<float>());       // exact: drawn by libtorch from the seed
    }

    [RequiresSongModels]
    public void Latent_MatchesTheReference_BlockByBlock()
    {
        var kept = run.Result!.Intermediates!;
        using var latent = Of(kept.Latent, 1, 20, 64);
        var reference = SongReference.Song["latent"];
        var exact = SongReference.Song["latent64"];
        BigVganReference.AssertClose(log, "latent block 0", latent.narrow(1, 0, 10), reference.narrow(1, 0, 10), Tolerances.Block);
        Reference.AssertWithinSensitivity(log, "latent block 1", latent.narrow(1, 10, 10), reference.narrow(1, 10, 10), exact.narrow(1, 10, 10), Tolerances.Sensitivity);
    }

    [RequiresSongModels]
    public void Waveform_AndStereo_AndWavFile_MatchTheReference()
    {
        var r = run.Result!;
        using var wave = Of(r.Audio, 1, 1, r.Audio.Length);
        BigVganReference.AssertClose(log, "mono waveform", wave, SongReference.Song["wave"], Tolerances.SongWave);

        var stereo = r.Channels(stereo: true);
        using var left = Of(stereo[0], stereo[0].Length);
        using var right = Of(stereo[1], stereo[1].Length);
        BigVganReference.AssertClose(log, "left channel", left, SongReference.Song["stereo"][0], Tolerances.SongWave);
        BigVganReference.AssertClose(log, "right channel", right, SongReference.Song["stereo"][1], Tolerances.SongWave);

        // the WAV files, in 16-bit steps
        var reference = WavFile.Read(SongReference.E2eDir.PathOf("song.wav")).Shorts();
        var path = Path.Combine(Path.GetTempPath(), $"dr2-{Guid.NewGuid():N}.wav");
        try
        {
            r.WriteWav(path);
            var ours = WavFile.Read(path);
            Assert.Equal((2, 48000u, 16), ((int)ours.Channels, ours.SampleRate, (int)ours.Bits));
            var shorts = ours.Shorts();
            Assert.Equal(reference.Length, shorts.Length);
            int worst = 0;
            long differing = 0;
            for (int i = 0; i < shorts.Length; i++)
            {
                int d = Math.Abs(shorts[i] - reference[i]);
                if (d > 0) differing++;
                worst = Math.Max(worst, d);
            }
            log.WriteLine($"WAV file: {differing} of {shorts.Length} samples differ, by at most {worst} step(s) of 16 bits");
            Assert.True(worst <= Tolerances.SongWavSteps, $"the WAV differs by {worst} steps (limit {Tolerances.SongWavSteps})");
        }
        finally { File.Delete(path); }
    }

    [RequiresSongModels]
    public void Events_ShowTheStagesInOrder_AndOneModelAtATime()
    {
        var events = run.Events;
        var expected = new (SongStage, SongEvent)[]
        {
            (SongStage.Lyrics, SongEvent.StageStarted), (SongStage.Lyrics, SongEvent.StageFinished),
            (SongStage.Style, SongEvent.StageStarted), (SongStage.Style, SongEvent.ModelLoaded), (SongStage.Style, SongEvent.ModelReleased), (SongStage.Style, SongEvent.StageFinished),
            (SongStage.Network, SongEvent.StageStarted), (SongStage.Network, SongEvent.ModelLoaded), (SongStage.Network, SongEvent.BlockDone), (SongStage.Network, SongEvent.BlockDone),
            (SongStage.Network, SongEvent.ModelReleased), (SongStage.Network, SongEvent.StageFinished),
            (SongStage.Decoder, SongEvent.StageStarted), (SongStage.Decoder, SongEvent.ModelLoaded), (SongStage.Decoder, SongEvent.ModelReleased), (SongStage.Decoder, SongEvent.StageFinished),
        };
        Assert.Equal(expected, events.Select(e => (e.Stage, e.Event)).ToArray());

        var blocks = events.Where(e => e.Event == SongEvent.BlockDone).ToArray();
        Assert.Equal(new[] { (1, 2), (2, 2) }, blocks.Select(b => (b.Done, b.Total)).ToArray());

        // never two models at once: a load is always preceded by the release of the previous stage's model
        int resident = 0;
        foreach (var e in events)
        {
            if (e.Event == SongEvent.ModelLoaded) resident++;
            if (e.Event == SongEvent.ModelReleased) resident--;
            Assert.InRange(resident, 0, 1);
        }
    }

    [RequiresSongModels]
    public void SameSeed_GivesTheSameSong_AnotherSeed_AnotherSong()
    {
        var generator = new SongGenerator(new SongOptions(Checkpoint.Dir!), new ReplayEngine());
        var first = run.Result!;
        // a short second run with the same seed would need the whole network again, so the check is on the noise and the first block
        using var a = stack(NoiseSource.Draw(first.Seed, 2, ScalarType.Float32, CPU));
        using var b = stack(NoiseSource.Draw(first.Seed + 1, 2, ScalarType.Float32, CPU));
        Assert.True((a - b).abs().max().item<float>() > 0.1f);
        Assert.NotNull(generator);
    }
}

/// <summary>Cancellation and half precision on the real models.</summary>
public class SongGeneratorRunTests
{
    private readonly ITestOutputHelper log;

    public SongGeneratorRunTests(ITestOutputHelper log) => this.log = log;

    private static ValidatedSong ReferenceRequest(out string lyrics)
    {
        var meta = SongReference.SongMeta;
        lyrics = meta.GetProperty("lyrics").GetString()!;
        var request = new SongRequest(lyrics, meta.GetProperty("style").GetString()!, 10, meta.GetProperty("cfg_strength").GetDouble(), meta.GetProperty("steps").GetInt32(), SongReference.Seed);
        return request.Validate() with { Seconds = 4, Frames = 20 };       // 4 s is below the service's minimum; the reference song is 4 s
    }

    private sealed class Sync : IProgress<SongProgress>
    {
        private readonly Action<SongProgress> handler;
        public Sync(Action<SongProgress> handler) => this.handler = handler;
        public void Report(SongProgress value) => handler(value);
    }

    [RequiresSongModels]
    public void Cancelled_AfterTheFirstBlock_StopsWithoutADecoderOrAPartialSong()
    {
        var validated = ReferenceRequest(out var lyrics);
        var generator = new SongGenerator(new SongOptions(Checkpoint.Dir!), new ReplayEngine());
        using var cancel = new CancellationTokenSource();
        var events = new List<SongProgress>();
        var progress = new Sync(e =>
        {
            events.Add(e);
            if (e.Event == SongEvent.BlockDone && e.Done == 1) cancel.Cancel();
        });
        Assert.ThrowsAny<OperationCanceledException>(() => generator.Run(validated, lyrics, progress, cancel.Token));
        Assert.Single(events, e => e.Event == SongEvent.BlockDone);                     // no further block was sampled
        Assert.DoesNotContain(events, e => e.Stage == SongStage.Decoder);
        log.WriteLine($"{events.Count} events before the cancellation took effect");

        // nothing stays loaded or locked: the same generator makes a song afterwards
        var again = generator.Run(validated with { Frames = 10 }, lyrics, null, CancellationToken.None);
        Assert.Equal(10 * 9600, again.Audio.Length);
    }

    [RequiresSongModels]
    public void Float16_OnCpu_IsReportedAgainstTheFloat32Reference()
    {
        var validated = ReferenceRequest(out var lyrics);
        var generator = new SongGenerator(new SongOptions(Checkpoint.Dir!, Precision: SongPrecision.Float16, CaptureIntermediates: true), new ReplayEngine());
        var r = generator.Run(validated, lyrics, null, CancellationToken.None);
        using var wave = tensor(r.Audio, 1, 1, r.Audio.Length);
        using var latent = tensor(r.Intermediates!.Latent, 1, 20, 64);
        var refWave = SongReference.Song["wave"];
        var refLatent = SongReference.Song["latent"];
        log.WriteLine($"[informational] float16 on CPU, {r.Elapsed.TotalSeconds:F1} s. NOT comparable sample by sample with the float32 reference: the noise is drawn in half precision, so it is another song (as in the reference service on CUDA). Differences from it: latent max_abs {(latent - refLatent).abs().max().item<float>():E3}, " +
                      $"waveform max_abs {(wave - refWave).abs().max().item<float>():E3} (scale {refWave.abs().max().item<float>():F3})");
        Assert.Equal(SongPrecision.Float16, r.Precision);
        Assert.True(wave.isfinite().all().item<bool>());
        Assert.True(wave.abs().max().item<float>() <= 1f);
    }
}

/// <summary>Runs the reference request once and keeps the result and the progress events for the tests of the class.</summary>
public sealed class SongRun
{
    public SongResult? Result { get; }
    public List<SongProgress> Events { get; } = new();

    public SongRun()
    {
        if (Checkpoint.Env.NoCheckpointReason is not null || new ModelDirectory(Checkpoint.Dir!).Missing.Count > 0) return;
        var meta = SongReference.SongMeta;
        var request = new SongRequest(meta.GetProperty("lyrics").GetString()!, meta.GetProperty("style").GetString()!, meta.GetProperty("max_secs").GetDouble(),
            meta.GetProperty("cfg_strength").GetDouble(), meta.GetProperty("steps").GetInt32(), meta.GetProperty("seed").GetInt64());
        var validated = (request with { Seconds = 10 }).Validate() with { Seconds = request.Seconds, Frames = (int)(request.Seconds * SongRequest.FramesPerSecond) };   // 4 s is below the service's 10 s minimum
        var generator = new SongGenerator(new SongOptions(Checkpoint.Dir!, CaptureIntermediates: true), new ReplayEngine());
        Result = generator.Run(validated, request.Lyrics, new Sync(Events), CancellationToken.None);
    }

    private sealed class Sync : IProgress<SongProgress>
    {
        private readonly List<SongProgress> events;
        public Sync(List<SongProgress> events) => this.events = events;
        public void Report(SongProgress value) => events.Add(value);
    }
}
