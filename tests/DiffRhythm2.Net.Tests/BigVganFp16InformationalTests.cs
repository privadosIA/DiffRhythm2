using DiffRhythm2.Decoder;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>
/// INFORMATIONAL: the production precision (half) against the half-precision reference and against the fp32 reference. It prints the deviations and
/// never fails on their magnitude; only that the output is finite and in range is asserted. Nothing here gates a change (design D6).
/// </summary>
public class BigVganFp16InformationalTests
{
    private readonly ITestOutputHelper log;

    public BigVganFp16InformationalTests(ITestOutputHelper log) => this.log = log;

    [RequiresCpuDecoder]
    public void Fp16_IsReportedAgainstTheHalfAndTheFp32References()
    {
        using var model = BigVganWeights.LoadDirectory(Checkpoint.Dir!, CPU, ScalarType.Float16);
        using var latent = BigVganReference.Latent40.to(ScalarType.Float16);
        using var wave = new LatentDecoder(model).Decode(latent);

        var half = BigVganReference.Fp16.Load("decode_40.safetensors")["wave"];
        var full = BigVganReference.Load("decode_40")["wave"];
        void Report(string what, Tensor reference)
        {
            using var d = (wave.to(ScalarType.Float64) - reference.to(ScalarType.Float64)).abs();
            log.WriteLine($"[informational] C# fp16 vs {what}: max_abs={d.max().item<double>():E3} mean_abs={d.mean().item<double>():E3}");
        }
        Report("fp16 reference (CPU half)", half);
        Report("fp32 reference", full);
        log.WriteLine($"[informational] fp16 reference vs fp32 reference: max_abs={(half - full).abs().max().item<float>():E3}");

        Assert.Equal(new long[] { 1, 1, 40L * 9600 }, wave.shape);
        Assert.True(wave.isfinite().all().item<bool>(), "the half-precision output must be finite");
        Assert.True(wave.abs().max().item<float>() <= 1f);
    }
}
