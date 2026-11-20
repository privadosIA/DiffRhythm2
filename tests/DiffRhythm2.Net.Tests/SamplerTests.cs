using System.Text.Json;
using DiffRhythm2.Model;
using DiffRhythm2.Sampling;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

public class EndOfSongTests
{
    /// <summary>Scripted blocks that went through the real sample_block_cache (tools/dump_eos_reference.py): same stop block, same length.</summary>
    [Fact]
    public void EveryReferenceCase_StopsAtTheSameBlockWithTheSameLength()
    {
        var tensors = Reference.Eos.Load("eos.safetensors");
        using var doc = JsonDocument.Parse(Reference.Eos.ReadAllText("eos.json"));
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToList();
        Assert.True(cases.Count >= 8);
        foreach (var c in cases)
        {
            int i = c.GetProperty("index").GetInt32();
            string name = c.GetProperty("name").GetString()!;
            int given = c.GetProperty("blocks_given").GetInt32();
            long frames = c.GetProperty("frames").GetInt64();
            int sampled = c.GetProperty("blocks_sampled").GetInt32();

            var stream = zeros(1, 0, 64);
            long keptFrames = -1;
            int blocksUsed = given;
            for (int j = 0; j < given; j++)
            {
                stream = cat(new[] { stream, tensors[$"case{i}_block{j}"] }, dim: 1);
                if (EndOfSong.EndPosition(stream) is { } end) { keptFrames = end; blocksUsed = j + 1; break; }
            }
            if (keptFrames < 0) keptFrames = stream.shape[1];
            Assert.True(frames == keptFrames, $"{name}: expected {frames} frames, got {keptFrames}");
            Assert.True(sampled == blocksUsed, $"{name}: expected to stop after {sampled} blocks, stopped after {blocksUsed}");
        }
    }

    [Fact]
    public void Threshold_IsInclusive_AndMeasuredOnTheMeanSquaredErrorToOnes()
    {
        Tensor Frames(float last) => cat(new[] { zeros(1, 9, 64), full(new long[] { 1, 1, 64 }, last) }, dim: 1);
        Assert.NotNull(EndOfSong.EndPosition(Frames(1f + MathF.Sqrt(0.049f))));
        Assert.Null(EndOfSong.EndPosition(Frames(1f + MathF.Sqrt(0.051f))));
    }
}

public class EulerSolverTests
{
    [Fact]
    public void TimeGrid_IsTheReferenceLinspace()
    {
        var grid = EulerSolver.TimeGrid(16);
        Assert.Equal(Reference.Run["time_grid"].shape, grid.shape);
        Assert.Equal(0f, (grid - Reference.Run["time_grid"]).abs().max().item<float>());
    }

    [Fact]
    public void Integrate_TakesGridPointsMinusOneSteps_EachFromThePreviousState()
    {
        var times = new List<float>();
        Tensor F(Tensor t, Tensor y) { times.Add(t.item<float>()); return y.clone().fill_(1f); }   // dy/dt = 1
        var states = new List<Tensor>();
        var end = EulerSolver.Integrate(F, zeros(2, 3), EulerSolver.TimeGrid(16), states);
        Assert.Equal(15, times.Count);
        Assert.Equal(16, states.Count);
        Assert.Equal(0f, times[0]);
        Assert.Equal(1f, end[0, 0].item<float>(), 5);
    }

    [Fact]
    public void Integrate_IsExplicitEuler_NotAHigherOrderMethod()
    {
        // dy/dt = t on [0, 1] with 3 points: Euler gives 0.5*0 + 0.5*0.5 = 0.25 (the exact integral is 0.5)
        var end = EulerSolver.Integrate((t, y) => y.clone().fill_(t.item<float>()), zeros(1), EulerSolver.TimeGrid(3));
        Assert.Equal(0.25f, end.item<float>(), 6);
    }

    [Fact]
    public void Grid_NeedsAtLeastTwoPoints() => Assert.Throws<ArgumentException>(() => EulerSolver.TimeGrid(1));
}

/// <summary>Sampler behaviour that needs no checkpoint: a tiny random DiT.</summary>
public class BlockSamplerStructureTests
{
    private static readonly DitConfig Cfg = new(32, 2, 2, 2, 8, 10, 10);

    private static BlockSampler Sampler()
    {
        manual_seed(11);
        return new BlockSampler(new Dit(Cfg));
    }

    private static List<Tensor> Noise(int blocks, long seed)
    {
        manual_seed(seed);
        return Enumerable.Range(0, blocks).Select(_ => randn(1, 10, 8)).ToList();
    }

    private static Tensor Style() { manual_seed(2); return randn(1, 512); }
    private static Tensor Tokens() => tensor(new long[] { 3, 1, 4, 1, 5 });
    private static readonly SamplerOptions Fast = new(Steps: 4, CfgStrength: 2.0);

    [Fact]
    public void SameInputs_GiveIdenticalLatents()
    {
        var s = Sampler();
        var noise = Noise(2, 5);
        var a = s.Sample(Tokens(), Style(), noise, 20, Fast);
        var b = s.Sample(Tokens(), Style(), noise, 20, Fast);
        Assert.Equal(new long[] { 1, 20, 8 }, a.shape);
        Assert.Equal(0f, (a - b).abs().max().item<float>());
    }

    [Fact]
    public void ALaterBlock_DependsOnTheEarlierBlocks()
    {
        var s = Sampler();
        var n1 = Noise(2, 5);
        var n2 = new List<Tensor> { Noise(1, 99)[0], n1[1] };            // block 0 differs, block 1 has the same noise
        var a = s.Sample(Tokens(), Style(), n1, 20, Fast);
        var b = s.Sample(Tokens(), Style(), n2, 20, Fast);
        Assert.True((a.narrow(1, 10, 10) - b.narrow(1, 10, 10)).abs().max().item<float>() > 1e-4f, "block 1 did not react to a different block 0");
    }

    [Fact]
    public void MissingNoise_FailsBeforeStarting_NamingTheShortfall()
    {
        var e = Assert.Throws<ArgumentException>(() => Sampler().Sample(Tokens(), Style(), Noise(3, 5), 40, Fast));
        Assert.Contains("4 noise blocks", e.Message);
        Assert.Contains("short by 1", e.Message);
    }

    [Fact]
    public void FramesAreRoundedUpToWholeBlocks_AndNeverPadded()
    {
        var traces = new List<BlockTrace>();
        var latent = Sampler().Sample(Tokens(), Style(), Noise(5, 5), 45, Fast, traces.Add);
        Assert.Equal(5, traces.Count);
        Assert.Equal(new long[] { 1, 50, 8 }, latent.shape);
    }

    [Fact]
    public void EmptyLyrics_SkipTheTextPrefill_AndStillGenerate()
    {
        var latent = Sampler().Sample(zeros(0, dtype: ScalarType.Int64), Style(), Noise(1, 5), 10, Fast);
        Assert.Equal(new long[] { 1, 10, 8 }, latent.shape);
    }

    [Fact]
    public void Guidance_UsesTwoForwardsPerStep_AndNoneWhenDisabled()
    {
        var s = Sampler();
        var guided = new List<BlockTrace>();
        s.Sample(Tokens(), Style(), Noise(1, 5), 10, new SamplerOptions(16, 2.0), guided.Add);
        Assert.Equal(15, guided[0].VelocityCalls);
        Assert.Equal(30, guided[0].ModelForwards);

        var plain = new List<BlockTrace>();
        s.Sample(Tokens(), Style(), Noise(1, 5), 10, new SamplerOptions(16, 0.0), plain.Add);
        Assert.Equal(15, plain[0].VelocityCalls);
        Assert.Equal(15, plain[0].ModelForwards);
    }

    [Fact]
    public void InvalidInputs_AreRejected()
    {
        var s = Sampler();
        Assert.Throws<ArgumentException>(() => s.Sample(Tokens(), Style(), Noise(1, 5), 0, Fast));
        Assert.Throws<ArgumentException>(() => s.Sample(tensor(new[] { 1f }), Style(), Noise(1, 5), 10, Fast));
        Assert.Throws<ArgumentException>(() => s.Sample(Tokens(), randn(1, 7), Noise(1, 5), 10, Fast));
        Assert.Throws<ArgumentException>(() => s.Sample(Tokens(), Style(), new List<Tensor> { randn(1, 9, 8) }, 10, Fast));
    }
}

/// <summary>The sampler against the fp32 golden run (inputs: tokens, style and the per-block noise of the fixture).</summary>
[Collection(RealModelCollection.Name)]
public class BlockSamplerParityTests
{
    private readonly RealDitFixture fx;
    private readonly ITestOutputHelper log;

    public BlockSamplerParityTests(RealDitFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    private static List<Tensor> Noise(int blocks) => Enumerable.Range(0, blocks).Select(i => Reference.Inputs["noise"][i]).ToList();

    private Tensor Run(Tensor tokens, int frames, double cfg, List<BlockTrace> traces)
    {
        using var _ = no_grad();
        return new BlockSampler(fx.Model!).Sample(tokens, Reference.Inputs["style"], Noise(5), frames, new SamplerOptions(16, cfg), traces.Add);
    }

    /// <summary>Every integration state of every block against the reference (block_states[block][step]).</summary>
    private void CompareStates(Dictionary<string, Tensor> run, List<BlockTrace> traces, double relTol)
    {
        foreach (var t in traces)
            for (int step = 0; step < t.States.Count; step++)
                Reference.AssertClose(log, $"block {t.Index} step {step}", t.States[step], run["block_states"][t.Index][step], relTol);
    }

    [RequiresCheckpoint]
    public void OneBlock_MatchesTheReference_StepByStep()
    {
        var traces = new List<BlockTrace>();
        var latent = Run(Reference.Inputs["tokens"], 10, 2.0, traces);
        Assert.Equal(15, traces[0].VelocityCalls);
        Assert.Equal(30, traces[0].ModelForwards);
        Assert.Equal(16, traces[0].States.Count);
        CompareStates(Reference.Run, traces, Tolerances.Block);
        Reference.AssertClose(log, "latent (1 block)", latent, Reference.Run["block_states"][0][15], Tolerances.Block);
    }

    [RequiresCheckpoint]
    public void FourBlocks_MatchTheReference_AsWellAsTheReferenceMatchesTheExactAnswer()
    {
        var traces = new List<BlockTrace>();
        var latent = Run(Reference.Inputs["tokens"], 40, 2.0, traces);
        Assert.Equal(4, traces.Count);
        Assert.Equal(new long[] { 1, 40, 64 }, latent.shape);
        foreach (var t in traces)
            Reference.AssertWithinSensitivity(log, $"block {t.Index}", t.Sampled, Reference.Run["block_states"][t.Index][15], Reference.Fp64["block_states"][t.Index][15], Tolerances.Sensitivity);
        Reference.AssertWithinSensitivity(log, "latent (4 blocks)", latent, Reference.Run["latent"], Reference.Fp64["latent"].narrow(1, 0, 40), Tolerances.Sensitivity);
    }

    [RequiresCheckpoint]
    public void GuidanceDisabled_MatchesTheReference_WithOneForwardPerStep()
    {
        var traces = new List<BlockTrace>();
        var latent = Run(Reference.Inputs["tokens"], 10, 0.0, traces);
        var reference = Reference.Dit.Load("run_cfg0.safetensors");
        Assert.Equal(15, traces[0].ModelForwards);
        Reference.AssertClose(log, "latent (cfg 0)", latent, reference["latent"], Tolerances.Block);
    }

    [RequiresCheckpoint]
    public void EmptyLyrics_MatchTheReference()
    {
        var traces = new List<BlockTrace>();
        var latent = Run(zeros(0, dtype: ScalarType.Int64), 10, 2.0, traces);
        var reference = Reference.Dit.Load("run_empty.safetensors");
        Reference.AssertClose(log, "latent (no lyrics)", latent, reference["latent"], Tolerances.Block);
    }

    [RequiresCheckpoint]
    public void FortyFiveFrames_GiveFiftyLikeTheReference()
    {
        var traces = new List<BlockTrace>();
        var latent = Run(Reference.Inputs["tokens"], 45, 2.0, traces);
        var reference = Reference.Dit.Load("run_45.safetensors");
        Assert.Equal(5, traces.Count);
        Assert.Equal(reference["latent"].shape, latent.shape);
        Assert.Equal(new long[] { 1, 50, 64 }, latent.shape);
        foreach (var t in traces)
            Reference.AssertWithinSensitivity(log, $"block {t.Index}", t.Sampled, reference["block_states"][t.Index][15], Reference.Fp64["block_states"][t.Index][15], Tolerances.Sensitivity);
        Reference.AssertWithinSensitivity(log, "latent (45 frames)", latent, reference["latent"], Reference.Fp64["latent"], Tolerances.Sensitivity);
    }
}
