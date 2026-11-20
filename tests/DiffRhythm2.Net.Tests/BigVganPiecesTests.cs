using DiffRhythm2.Decoder;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The decoder's building blocks against the reference (no checkpoint needed: the activation fixtures carry their own parameters), the configuration, and strict loading on a tiny synthetic checkpoint.</summary>
public class BigVganPiecesTests : IDisposable
{
    private readonly ITestOutputHelper log;
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-bv-").FullName;

    public BigVganPiecesTests(ITestOutputHelper log) => this.log = log;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    private const string Published = """
        { "resblock": "1", "upsample_rates": [10,10,8,3,2,2], "upsample_kernel_sizes": [20,20,16,7,4,4], "upsample_initial_channel": 1536,
          "resblock_kernel_sizes": [3,7,11], "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
          "use_tanh_at_final": false, "use_bias_at_final": false, "activation": "snakebeta", "snake_logscale": true,
          "fps": 5, "num_mels": 128, "in_channels": 64, "sampling_rate": 48000 }
        """;

    // ---- configuration ----------------------------------------------------------------------------------------------------------------

    [Fact]
    public void Config_Published_TakesTheDocumentedGeometry()
    {
        var c = BigVganConfig.Parse(Published);
        Assert.Equal(64, c.InChannels);
        Assert.Equal(1536, c.UpsampleInitialChannel);
        Assert.Equal(new[] { 10, 10, 8, 3, 2, 2 }, c.UpsampleRates);
        Assert.Equal(9600, c.SamplesPerFrame);
        Assert.Equal(48000, c.SamplingRate);
        Assert.False(c.UseTanhAtFinal);
        Assert.False(c.UseBiasAtFinal);
    }

    [Fact]
    public void Config_PlainSnake_IsRejectedByName()
    {
        var e = Assert.Throws<NotSupportedException>(() => BigVganConfig.Parse(Published.Replace("\"snakebeta\"", "\"snake\"")));
        Assert.Contains("snake", e.Message);
        Assert.Contains("activation", e.Message);
    }

    [Fact]
    public void Config_ResidualBlockKind2_IsRejectedByName()
    {
        var e = Assert.Throws<NotSupportedException>(() => BigVganConfig.Parse(Published.Replace("\"resblock\": \"1\"", "\"resblock\": \"2\"")));
        Assert.Contains("\"2\"", e.Message);
    }

    // ---- activation pieces vs the reference ---------------------------------------------------------------------------------------------

    private static Tensor Fx(string key) => BigVganReference.Load("activation")[key];

    [Fact]
    public void SnakeBeta_MatchesTheReference()
    {
        using var _ = no_grad();
        var snake = new SnakeBeta(24, logscale: true);
        snake.state_dict()["alpha"].copy_(Fx("post_alpha"));
        snake.state_dict()["beta"].copy_(Fx("post_beta"));
        BigVganReference.AssertClose(log, "snakebeta", snake.forward(Fx("snake_in")), Fx("snake_out"), Tolerances.Piece);
    }

    [Theory]
    [InlineData("post")]
    [InlineData("res17_5")]
    public void KaiserSincFilters_AreWhatTheFormulaGives(string which)
    {
        // 12 taps, cutoff 0.25, half width 0.3 for both the up- and the down-sampling filter
        var taps = KaiserSinc.Filter(0.25, 0.3, 12);
        foreach (var key in new[] { $"{which}_up_filter", $"{which}_down_filter" })
        {
            var stored = Fx(key).reshape(-1).to(ScalarType.Float64).data<double>().ToArray();
            double worst = stored.Zip(taps, (a, b) => Math.Abs(a - b)).Max();
            log.WriteLine($"{key}: max_abs difference to the computed filter {worst:E3}");
            Assert.True(worst < 1e-6, $"{key}: computed filter differs from the checkpoint's by {worst:E3}");
        }
        Assert.Equal(1.0, taps.Sum(), 12);
    }

    [Theory]
    [InlineData("post")]
    [InlineData("res17_5")]
    public void Activation1d_MatchesTheReference_IncludingTheEdgeSamples(string which)
    {
        using var _ = no_grad();
        var act = new Activation1d(24, logscale: true);
        var state = act.state_dict();
        state["act.alpha"].copy_(Fx($"{which}_alpha"));
        state["act.beta"].copy_(Fx($"{which}_beta"));
        state["upsample.filter"].copy_(Fx($"{which}_up_filter"));
        state["downsample.lowpass.filter"].copy_(Fx($"{which}_down_filter"));

        var got = act.forward(Fx($"{which}_in"));
        var expected = Fx($"{which}_out");
        BigVganReference.AssertClose(log, $"{which} activation", got, expected, Tolerances.Piece);
        // The first and last samples are produced by the replicate padding of both resamplers: judged on their own so that a wrong edge cannot hide in the whole-signal scale.
        BigVganReference.AssertClose(log, $"{which} left edge", got.narrow(2, 0, 6), expected.narrow(2, 0, 6), Tolerances.Piece);
        BigVganReference.AssertClose(log, $"{which} right edge", got.narrow(2, got.shape[2] - 6, 6), expected.narrow(2, expected.shape[2] - 6, 6), Tolerances.Piece);
    }

    [Fact]
    public void UpSample1d_DoublesTheLength_AndDownSample1d_HalvesIt()
    {
        using var _ = no_grad();
        var x = randn(1, 3, 50);
        var up = new UpSample1d(2, 12).forward(x);
        Assert.Equal(new long[] { 1, 3, 100 }, up.shape);
        Assert.Equal(new long[] { 1, 3, 50 }, new DownSample1d(2, 12).forward(up).shape);
    }

    // ---- tiny synthetic generator ---------------------------------------------------------------------------------------------------------

    private static readonly BigVganConfig Tiny = new(
        InChannels: 4, UpsampleInitialChannel: 16, UpsampleRates: new[] { 2, 3 }, UpsampleKernelSizes: new[] { 4, 7 },
        ResblockKernelSizes: new[] { 3 }, ResblockDilationSizes: new[] { new[] { 1, 3 } },
        SnakeLogscale: true, UseBiasAtFinal: false, UseTanhAtFinal: false, SamplingRate: 1000);

    /// <summary>A checkpoint for <see cref="Tiny"/> with random weights, in the converted-file spelling (weight-norm pairs), plus an optional edit. Returns the effective (folded) tensors.</summary>
    private Dictionary<string, Tensor> WriteCheckpoint(Action<Dictionary<string, Tensor>>? edit = null)
    {
        using var _ = no_grad();
        manual_seed(11);
        var source = new BigVganGenerator(Tiny);
        var effective = source.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var file = new Dictionary<string, Tensor>();
        foreach (var (name, t) in effective)
        {
            if (name.EndsWith(".filter")) { file[name] = t; continue; }           // the anti-alias buffers keep their computed values
            t.normal_();
            if (name.EndsWith(".weight"))
            {
                // v random, g random positive: the effective weight is v * g / ||v|| (norm over every dim but 0), computed here without the loader
                var v = t.clone();
                var g = rand(v.shape[0], 1, 1) + 0.5;
                var norm = v.pow(2).sum(new long[] { 1, 2 }, keepdim: true).sqrt();
                var stem = name[..^".weight".Length];
                file[stem + ".parametrizations.weight.original0"] = g;
                file[stem + ".parametrizations.weight.original1"] = v;
                effective[name] = v * (g / norm);
            }
            else file[name] = t;
        }
        edit?.Invoke(file);
        SafeTensorsWriter.Save(Path.Combine(tmp, "decoder.safetensors"), file);
        return effective;
    }

    private Exception LoadFails(Action<Dictionary<string, Tensor>> edit)
    {
        WriteCheckpoint(edit);
        return Assert.Throws<InvalidDataException>(() => BigVganWeights.Load(new BigVganGenerator(Tiny), CheckpointIndex.Open(Path.Combine(tmp, "decoder.safetensors")), CPU));
    }

    [Fact]
    public void Load_MatchingCheckpoint_FoldsEveryPairAndAssignsEveryTensor()
    {
        var expected = WriteCheckpoint();
        var model = new BigVganGenerator(Tiny);
        var result = BigVganWeights.Load(model, CheckpointIndex.Open(Path.Combine(tmp, "decoder.safetensors")), CPU);

        Assert.Equal(expected.Count, result.ModelTensors);
        int pairs = expected.Keys.Count(k => k.EndsWith(".weight"));
        Assert.Equal(pairs, result.PairsFolded);
        foreach (var (name, value) in model.state_dict())
        {
            double diff = (value - expected[name]).abs().max().item<float>();
            Assert.True(diff < 1e-5, $"{name} differs from the folded expectation by {diff:E3}");
        }
        Assert.DoesNotContain(model.state_dict().Keys, k => k.Contains("weight_g") || k.Contains("original"));
        Assert.DoesNotContain("conv_post.bias", model.state_dict().Keys);
    }

    [Fact]
    public void Load_MissingTensor_NamesIt()
    {
        var e = LoadFails(f => f.Remove("resblocks.1.convs2.1.bias"));
        Assert.Contains("resblocks.1.convs2.1.bias", e.Message);
        Assert.Contains("missing", e.Message);
    }

    [Fact]
    public void Load_MissingAntiAliasFilter_NamesIt()
    {
        var e = LoadFails(f => f.Remove("resblocks.0.activations.2.upsample.filter"));
        Assert.Contains("resblocks.0.activations.2.upsample.filter", e.Message);
    }

    [Fact]
    public void Load_ShapeMismatch_NamesTheTensorWithBothShapes()
    {
        var e = LoadFails(f => f["conv_pre.bias"] = zeros(3));
        Assert.Contains("conv_pre.bias", e.Message);
        Assert.Contains("[16]", e.Message);
        Assert.Contains("[3]", e.Message);
    }

    [Fact]
    public void Load_BiasOnTheBiaslessFinalConvolution_IsNotSilentlyDropped()
    {
        var e = LoadFails(f => f["conv_post.bias"] = zeros(1));
        Assert.Contains("conv_post.bias", e.Message);
    }

    [Fact]
    public void Load_UnpairedWeightNorm_IsRejected()
    {
        var e = LoadFails(f => f.Remove("conv_pre.parametrizations.weight.original0"));
        Assert.Contains("conv_pre", e.Message);
    }

    [Fact]
    public void Generator_OutputsExactlySamplesPerFrame_InsideTheUnitInterval()
    {
        WriteCheckpoint();
        using var model = new BigVganGenerator(Tiny);
        BigVganWeights.Load(model, CheckpointIndex.Open(Path.Combine(tmp, "decoder.safetensors")), CPU);
        using var _ = no_grad();
        var y = model.forward(randn(1, 4, 7) * 50);
        Assert.Equal(new long[] { 1, 1, 7 * 6 }, y.shape);
        Assert.Equal(6, Tiny.SamplesPerFrame);
        Assert.True(y.abs().max().item<float>() <= 1f);
    }

    [Fact]
    public void Generator_TanhAtFinal_IsHonouredWhenConfigured()
    {
        // Same weights, small input so that nothing is clipped: the tanh model must give tanh of what the clamp model gives.
        using var clamp = new BigVganGenerator(Tiny);
        using var tanh = new BigVganGenerator(Tiny with { UseTanhAtFinal = true });
        using var _ = no_grad();
        manual_seed(3);
        foreach (var (name, p) in clamp.named_parameters()) { if (name.EndsWith(".weight")) p.normal_(0, 0.15); else p.zero_(); }   // alpha = beta = 0 in log scale
        tanh.load_state_dict(clamp.state_dict());
        var x = randn(1, 4, 5) * 0.2;
        var a = clamp.forward(x);
        var b = tanh.forward(x);
        Assert.True(a.abs().max().item<float>() < 1f, "the premise: nothing is clipped");
        Assert.True(a.abs().max().item<float>() > 0.05f);
        Assert.True((b - a.tanh()).abs().max().item<float>() < 1e-6f);
        Assert.True((b - a).abs().max().item<float>() > 1e-4f, "tanh and clamp must differ here");
    }
}
