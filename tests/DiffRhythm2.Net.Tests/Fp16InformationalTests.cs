using DiffRhythm2.Model;
using DiffRhythm2.Sampling;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>
/// INFORMATIONAL. Production runs the DiT in half precision; this reports how far the C# fp16 sampler is from the reference's fp16 output
/// (tests/fixtures/dit_fp16, dumped on CPU) and from the fp32 reference. It never fails on magnitude: the fp32 gates decide whether the port is right
/// (design D2), and the sampler amplifies rounding so much (design D6) that a half-precision comparison cannot tell a bug from noise.
/// Note that the rotary table here is the core's float32 table cast back, whereas HuggingFace multiplies in the activation dtype: a known, accepted
/// difference of this path.
/// </summary>
[Collection(RealModelCollection.Name)]
public class Fp16InformationalTests
{
    private readonly ITestOutputHelper log;

    public Fp16InformationalTests(RealDitFixture _, ITestOutputHelper log) => this.log = log;

    [RequiresCheckpoint]
    public void Fp16Sampler_ReportsItsDeviationFromTheReferences()
    {
        var reference16 = new MusicGen.Core.Testing.FixtureDirectory(Path.Combine(Reference.Dit.Dir, "..", "dit_fp16")).Load("run.safetensors");
        var (model, _) = DitWeights.LoadDirectory(Checkpoint.Dir!, CPU, ScalarType.Float16);
        using var _m = model;
        using var _ = no_grad();

        var noise = Enumerable.Range(0, 5).Select(i => Reference.Inputs["noise"][i]).ToList();
        var traces = new List<BlockTrace>();
        var started = DateTime.UtcNow;
        var latent = new BlockSampler(model).Sample(Reference.Inputs["tokens"], Reference.Inputs["style"].to(ScalarType.Float16), noise, 40, new SamplerOptions(16, 2.0), traces.Add);
        log.WriteLine($"fp16 run: {latent.shape[1]} frames in {(DateTime.UtcNow - started).TotalSeconds:F1} s, dtype {latent.dtype}");

        Assert.Equal(ScalarType.Float16, latent.dtype);
        Assert.Equal(4, traces.Count);
        foreach (var t in traces)
        {
            Assert.True(t.Sampled.isfinite().all().item<bool>(), $"block {t.Index} has non-finite values");
            double Max(Tensor a, Tensor b) => (a.to(ScalarType.Float64) - b.to(ScalarType.Float64)).abs().max().item<double>();
            double Mean(Tensor a, Tensor b) => (a.to(ScalarType.Float64) - b.to(ScalarType.Float64)).abs().mean().item<double>();
            var r16 = reference16["block_states"][t.Index][15];
            var r32 = Reference.Run["block_states"][t.Index][15];
            log.WriteLine($"block {t.Index}: vs fp16 reference max={Max(t.Sampled, r16):E3} mean={Mean(t.Sampled, r16):E3} | vs fp32 reference max={Max(t.Sampled, r32):E3} mean={Mean(t.Sampled, r32):E3} | (fp16 reference vs fp32 reference max={Max(r16, r32):E3})");
        }
    }
}

/// <summary>INFORMATIONAL. Peak memory and wall time of the 4-block fp32 run; run it alone on a free machine (<c>--filter PeakMemory</c>) for figures worth recording.</summary>
[Collection(RealModelCollection.Name)]
public class PerformanceReportTests
{
    private readonly RealDitFixture fx;
    private readonly ITestOutputHelper log;

    public PerformanceReportTests(RealDitFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    [RequiresCpuCheckpoint]
    public void PeakMemory_AndWallTime_OfTheFourBlockRun()
    {
        using var _ = no_grad();
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        proc.Refresh();
        double afterLoadGiB = proc.WorkingSet64 / 1073741824.0;
        var noise = Enumerable.Range(0, 5).Select(i => Reference.Inputs["noise"][i]).ToList();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var latent = new BlockSampler(fx.Model!).Sample(Reference.Inputs["tokens"], Reference.Inputs["style"], noise, 40, new SamplerOptions(16, 2.0));
        sw.Stop();
        proc.Refresh();
        log.WriteLine($"working set after loading the model: {afterLoadGiB:F2} GiB");
        log.WriteLine($"peak working set of the process (loading + run): {proc.PeakWorkingSet64 / 1073741824.0:F2} GiB");
        log.WriteLine($"4-block fp32 run on CPU ({Environment.ProcessorCount} logical cores): {sw.Elapsed.TotalSeconds:F1} s, {latent.shape[1]} frames");
        Assert.Equal(40, latent.shape[1]);
    }
}
