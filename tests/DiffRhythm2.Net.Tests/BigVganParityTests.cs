using System.Diagnostics;
using DiffRhythm2.Decoder;
using MusicGen.Core.Audio;
using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The full decoder against the fp32 golden run: strict loading, weight-norm folding, every stage, whole chunks, and chunked decoding of the 40-frame latent.</summary>
[Collection(RealDecoderCollection.Name)]
public class BigVganParityTests
{
    private const int Spf = 9600;
    private readonly RealDecoderFixture fx;
    private readonly ITestOutputHelper log;

    public BigVganParityTests(RealDecoderFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    private BigVganGenerator Model => fx.Model!;

    // ---- loading ------------------------------------------------------------------------------------------------------------------

    [RequiresDecoder]
    public void Load_PublishedCheckpoint_ReadsEveryTensorAndFoldsEveryPair()
    {
        var r = fx.Report!;
        log.WriteLine($"read {r.TensorsRead} tensors, folded {r.PairsFolded} pairs, model tensors {r.ModelTensors}, {r.BytesRead / 1e6:F0} MB");
        Assert.Equal(783, r.TensorsRead);         // 116 weight-norm pairs (2 tensors each) + biases + alpha/beta + filters
        Assert.Equal(116, r.PairsFolded);         // conv_pre, 6 ups, 18 resblocks x 6 convs, conv_post
        Assert.Equal(783 - 116, r.ModelTensors);  // every pair became one plain weight; nothing is left over
        Assert.Equal(0, r.OutsidePrefix);
        Assert.DoesNotContain(Model.state_dict().Keys, k => k.Contains("original") || k.EndsWith("weight_g") || k.EndsWith("weight_v"));
        Assert.DoesNotContain("conv_post.bias", Model.state_dict().Keys);
    }

    [RequiresDecoder]
    public void FoldedWeights_MatchTheReferenceFold()
    {
        var state = Model.state_dict();
        foreach (var (name, expected) in BigVganReference.Load("weights"))
            BigVganReference.AssertClose(log, $"folded {name}", state[name], expected, DecoderTolerances.Wave);
    }

    // ---- stage by stage and whole chunks -------------------------------------------------------------------------------------------------

    private void CompareStages(string fixture)
    {
        var fix = BigVganReference.Load(fixture);
        using var _ = no_grad();
        var seen = new List<string>();
        Tensor wave;
        using (var x = fix["input"].alias())
        {
            wave = Model.Forward(x, (name, t) =>
            {
                seen.Add(name);
                if (fix.TryGetValue($"s_{name}", out var raw))
                    BigVganReference.AssertClose(log, $"{fixture} {name}", t.narrow(1, 0, raw.shape[1]), raw, DecoderTolerances.Stage);
                using var rms = t.to(ScalarType.Float64).pow(2).mean(new long[] { 2 }).sqrt()[0].to(ScalarType.Float32);
                var expectedRms = fix[$"s_{name}_rms"];
                double worst = ((rms.to(ScalarType.Float64) - expectedRms.to(ScalarType.Float64)).abs() / expectedRms.to(ScalarType.Float64).clamp_min(1e-12)).max().item<double>();
                log.WriteLine($"{fixture} {name} per-channel RMS: worst relative error {worst:E3} over {rms.shape[0]} channels");
                Assert.True(worst <= DecoderTolerances.Rms, $"{fixture} {name}: per-channel RMS differs by {worst:E3}");
            });
        }
        using (wave)
        {
            Assert.Equal(new long[] { 1, 1, 20 * Spf }, wave.shape);
            BigVganReference.AssertClose(log, $"{fixture} waveform", wave, fix["wave"], DecoderTolerances.Wave);
        }
        Assert.Equal(new[] { "conv_pre" }.Concat(Enumerable.Range(0, 6).SelectMany(i => new[] { $"up{i}", $"stage{i}" })).Concat(new[] { "act_post", "conv_post" }), seen);
    }

    [RequiresDecoder]
    public void RealChunk_MatchesTheReference_StageByStage() => CompareStages("chunk_real");

    [RequiresDecoder]
    public void RandomChunk_MatchesTheReference_StageByStage() => CompareStages("chunk_random");

    [RequiresDecoder]
    public void LoudChunk_IsClippedToTheUnitInterval_NotSquashedByATanh()
    {
        var fix = BigVganReference.Load("chunk_loud");
        using var _ = no_grad();
        using var wave = Model.forward(fix["input"]);
        float peak = wave.abs().max().item<float>();
        long clipped = (wave.abs() >= 1f).sum().item<long>();
        log.WriteLine($"loud chunk: peak {peak}, samples at the bound {clipped}");
        Assert.Equal(1f, peak);                    // a clamp reaches the bound exactly
        Assert.True(clipped > 0);
        BigVganReference.AssertClose(log, "loud waveform", wave, fix["wave"], DecoderTolerances.Wave);
    }

    // ---- chunked decoding of the golden-run latent ---------------------------------------------------------------------------------------

    [RequiresDecoder]
    public void GoldenLatent_DecodesToTheReferenceWaveform_IncludingEverySeam()
    {
        using var latent = BigVganReference.Latent40;
        var decoder = new LatentDecoder(Model);
        var sw = Stopwatch.StartNew();
        using var wave = decoder.Decode(latent);
        log.WriteLine($"decoded 40 frames in {sw.Elapsed.TotalSeconds:F1} s");

        Assert.Equal(new long[] { 1, 1, 40L * Spf }, wave.shape);
        Assert.True(wave.abs().max().item<float>() <= 1f);
        var expected = BigVganReference.Load("decode_40")["wave"];
        BigVganReference.AssertClose(log, "40-frame waveform", wave, expected, DecoderTolerances.Wave);

        // seams of T=40: chunk 1->2 at frame 17 and 2->3 at frame 22; each is judged on a window of two frames either side
        foreach (var seam in new[] { 17, 22 })
        {
            long from = (seam - 2L) * Spf, len = 4L * Spf;
            BigVganReference.AssertClose(log, $"seam at frame {seam}", wave.narrow(2, from, len), expected.narrow(2, from, len), DecoderTolerances.Wave);
        }
    }

    [RequiresDecoder]
    public void ShortSequence_EqualsDirectDecoding_AndEmptyIsEmpty()
    {
        using var latent = BigVganReference.Latent40.narrow(2, 0, 7).contiguous();
        using var _ = no_grad();
        using var viaDecoder = new LatentDecoder(Model).Decode(latent);
        using var direct = Model.forward(latent);
        Assert.Equal(new long[] { 1, 1, 7L * Spf }, viaDecoder.shape);
        Assert.Equal(0f, (viaDecoder - direct).abs().max().item<float>());

        using var empty = new LatentDecoder(Model).Decode(zeros(1, 64, 0));
        Assert.Equal(new long[] { 1, 1, 0 }, empty.shape);
    }

    [RequiresDecoder]
    public void WrongChannelCount_IsRejectedByTheRealDecoder()
    {
        var e = Assert.Throws<ArgumentException>(() => new LatentDecoder(Model).Decode(zeros(1, 32, 20)));
        Assert.Contains("64", e.Message);
        Assert.Contains("32", e.Message);
    }

    // ---- memory ----------------------------------------------------------------------------------------------------------------------

    /// <summary>Peak resident set of the process while <paramref name="work"/> runs, sampled every 20 ms (the OS high-water mark cannot be reset, so it is sampled).</summary>
    private static long PeakWorkingSet(Action work)
    {
        using var process = Process.GetCurrentProcess();
        long peak = 0;
        using var stop = new CancellationTokenSource();
        var sampler = Task.Run(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                process.Refresh();
                peak = Math.Max(peak, process.WorkingSet64);
                Thread.Sleep(20);
            }
        });
        try { work(); }
        finally { stop.Cancel(); sampler.Wait(); }
        return peak;
    }

    [RequiresCpuDecoder]
    public void LongSequence_DoesNotNeedMoreThanAChunkPlusTheOutputBuffer()
    {
        var decoder = new LatentDecoder(Model);
        Tensor Latents(int frames)
        {
            var g = BigVganReference.Latent40;
            return cat(Enumerable.Range(0, (frames + 39) / 40).Select(_ => g).ToArray(), 2).narrow(2, 0, frames).contiguous();
        }

        double baselineMb;
        using (var warm = decoder.Decode(Latents(60))) { }      // first-use allocations (thread pools, oneDNN primitives, allocator arenas) are not what is measured
        using (var self = Process.GetCurrentProcess()) baselineMb = self.WorkingSet64 / 1e6;
        long peak40 = 0, peak400 = 0, wave40 = 0, wave400 = 0;
        using (var l40 = Latents(40)) peak40 = PeakWorkingSet(() => { using var w = decoder.Decode(l40); wave40 = w.shape[2] * 4; });
        using (var l400 = Latents(400)) peak400 = PeakWorkingSet(() => { using var w = decoder.Decode(l400); wave400 = w.shape[2] * 4; });

        double growthMb = (peak400 - peak40) / 1e6, bufferMb = (wave400 - wave40) / 1e6;
        log.WriteLine($"peak working set: T=40 {peak40 / 1e6:F0} MB, T=400 {peak400 / 1e6:F0} MB; growth {growthMb:F0} MB, output buffer growth {bufferMb:F0} MB, resident before: {baselineMb:F0} MB");
        // the growth may not exceed the extra output buffer plus a margin for allocator noise; keeping a second chunk alive (and the activations of one chunk are several hundred MB) would blow through it
        Assert.True(growthMb <= bufferMb + 64, $"peak grew by {growthMb:F0} MB from 40 to 400 frames; the output buffer accounts for {bufferMb:F0} MB");
    }

    // ---- output file ---------------------------------------------------------------------------------------------------------------------

    [RequiresDecoder]
    public void Wav_IsMono48kHz_WithTheDecodedNumberOfSamples()
    {
        using var latent = BigVganReference.Latent40;
        using var wave = new LatentDecoder(Model).Decode(latent);
        var path = Path.Combine(Path.GetTempPath(), $"dr2-{Guid.NewGuid():N}.wav");
        try
        {
            LatentDecoder.WriteWav(path, wave, Model.Config.SamplingRate);
            var wav = WavFile.Read(path);
            Assert.Equal(1, wav.Channels);
            Assert.Equal(48000u, wav.SampleRate);
            Assert.Equal(40 * Spf, wav.Shorts().Length);
        }
        finally { File.Delete(path); }
    }

    [RequiresAudioDir]
    public void ListeningFile_IsWrittenToTheAudioDirectory()
    {
        using var latent = BigVganReference.Latent40;
        using var wave = new LatentDecoder(Model).Decode(latent);
        Directory.CreateDirectory(Checkpoint.AudioDir!);
        var path = Path.Combine(Checkpoint.AudioDir!, "diffrhythm2-golden-run.wav");
        LatentDecoder.WriteWav(path, wave, Model.Config.SamplingRate);
        log.WriteLine($"written: {path} ({new FileInfo(path).Length / 1e6:F1} MB, {wave.shape[2] / (double)Model.Config.SamplingRate:F1} s)");
        Assert.True(File.Exists(path));
    }
}
