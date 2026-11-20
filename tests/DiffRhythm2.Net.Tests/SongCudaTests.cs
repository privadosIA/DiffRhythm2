using DiffRhythm2.Pipeline;
using DiffRhythm2.Text;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>[Fact] that needs a CUDA device (a build with -p:TorchBackend=cuda-linux or cuda-windows) and, for the song, the models.</summary>
public sealed class RequiresCudaSongAttribute : FactAttribute
{
    public RequiresCudaSongAttribute(bool needsModels = false) =>
        Skip = !cuda.is_available() ? "no CUDA device: build and run with -p:TorchBackend=cuda-linux"
            : needsModels ? (Checkpoint.Env.NoCheckpointReason ?? (new ModelDirectory(Checkpoint.Dir!).Missing is { Count: > 0 } m ? $"the model directory lacks {string.Join(", ", m)}" : null))
            : null;
}

/// <summary>
/// CUDA, measured and never gating (half precision on CUDA is informational throughout this project): does the noise equal the reference's CUDA stream, and how close
/// is the song to the one the service itself makes on the GPU for the same request?
/// </summary>
public class SongCudaTests
{
    private readonly ITestOutputHelper log;

    public SongCudaTests(ITestOutputHelper log) => this.log = log;

    private static Dictionary<string, Tensor> Cuda => SongReference.Cuda;

    [RequiresCudaSong]
    public void Noise_DrawnOnEveryCudaDevice_EqualsTheReferencesCudaStream()
    {
        for (int i = 0; i < cuda.device_count(); i++)
        {
            var device = torch.device(DeviceType.CUDA, i);
            using var half = stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float16, device)).to(ScalarType.Float32).cpu();
            using var full = stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float32, device)).cpu();
            double dh = (half - Cuda["noise_cuda_f16"]).abs().max().item<float>(), df = (full - Cuda["noise_cuda_f32"]).abs().max().item<float>();
            log.WriteLine($"cuda:{i}: float16 draw differs from the reference's CUDA stream by {dh}, float32 by {df}");
            Assert.Equal(0.0, dh);
            Assert.Equal(0.0, df);
        }
    }

    [RequiresCudaSong(needsModels: true)]
    public void Song_OnCuda_InHalfPrecision_IsReportedAgainstTheServicesOwnSong()
    {
        var meta = SongReference.SongMeta;
        var request = new SongRequest(meta.GetProperty("lyrics").GetString()!, meta.GetProperty("style").GetString()!, 10, 2.0, 16, SongReference.Seed);
        var validated = request.Validate() with { Seconds = 4, Frames = 20 };
        var generator = new SongGenerator(new SongOptions(Checkpoint.Dir!, Device: "cuda:0", CaptureIntermediates: true), new ReplayEngine());
        var r = generator.Run(validated, request.Lyrics, null, CancellationToken.None);
        Assert.Equal(SongPrecision.Float16, r.Precision);

        using var noise = tensor(r.Intermediates!.Noise, 2, 1, 10, 64);
        double noiseDiff = (noise - Cuda["noise_cuda_f16"].narrow(0, 0, 2)).abs().max().item<float>();
        using var latent = tensor(r.Intermediates.Latent, 1, 20, 64);
        using var wave = tensor(r.Audio, 1, 1, r.Audio.Length);
        var refLatent = Cuda["latent"];
        var refWave = Cuda["wave"];
        double SnrDb(Tensor got, Tensor want) => 20 * Math.Log10(want.to(ScalarType.Float64).pow(2).sum().sqrt().item<double>() / (got.to(ScalarType.Float64) - want.to(ScalarType.Float64)).pow(2).sum().sqrt().item<double>());
        log.WriteLine($"[informational] cuda:0 float16, {r.Elapsed.TotalSeconds:F1} s. Noise vs the reference's CUDA stream: {noiseDiff}. " +
                      $"Latent vs the service's: block 0 max_abs {(latent.narrow(1, 0, 10) - refLatent.narrow(1, 0, 10)).abs().max().item<float>():E3}, block 1 {(latent.narrow(1, 10, 10) - refLatent.narrow(1, 10, 10)).abs().max().item<float>():E3}. " +
                      $"Waveform vs the service's: max_abs {(wave - refWave).abs().max().item<float>():E3}, SNR {SnrDb(wave, refWave):F1} dB (peak {refWave.abs().max().item<float>():F3})");
        Assert.True(wave.isfinite().all().item<bool>());
        Assert.True(wave.abs().max().item<float>() <= 1f);
        Assert.Equal(0.0, noiseDiff);
    }

    [RequiresCudaSong(needsModels: true)]
    public void ListeningFile_OnCuda_IsWrittenToTheAudioDirectory()
    {
        if (Checkpoint.AudioDir is not { } audio) { log.WriteLine("DIFFRHYTHM2_AUDIO_DIR not set: nothing written"); return; }
        if (!EspeakEngine.TryGetShared(out _, out var reason)) { log.WriteLine($"skipped: {reason}"); return; }
        Directory.CreateDirectory(audio);
        var lyrics = "[verse]\nWalking down the road tonight\nStars are shining way up high\nEvery step feels so alright\nDreaming under the open sky\n[chorus]\nSing along, sing along\nFeel the music all night long\nHold my hand and don't let go\nLet the rhythm take us home";
        var result = new SongGenerator(new SongOptions(Checkpoint.Dir!, Device: "cuda:0")).Generate(new SongRequest(lyrics, "Pop, Piano, Female vocals, Happy, Bright", 60, 2.0, 16, SongReference.Seed));
        var path = Path.Combine(audio, "pipeline-cuda-fp16-seed1234.wav");
        result.WriteWav(path);
        log.WriteLine($"written {path}: {result.Audio.Length / 48000.0:F1} s of audio in {result.Elapsed.TotalSeconds:F1} s, {result.Precision} on cuda:0");
        Assert.True(File.Exists(path));
    }
}
