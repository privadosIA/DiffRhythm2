using DiffRhythm2.AudioTower;
using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The audio tower on a tiny synthetic checkpoint: names, strict loading, evaluation mode, shapes. No checkpoint needed.</summary>
public class MuQEncoderModelTests : IDisposable
{
    private static readonly MuQConfig Tiny = new(Bands: 8, ConvDim: 4, Hidden: 16, Layers: 2, Heads: 2, Intermediate: 32, ConvKernel: 5);
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-muq-").FullName;

    public void Dispose()
    {
        Directory.Delete(tmp, recursive: true);
        GC.SuppressFinalize(this);
    }

    private static MuLanAudioTower TinyTower() => new(Tiny, dim: 12, latentDim: 6);

    /// <summary>The tensors a real checkpoint has and the reference never evaluates, with small shapes.</summary>
    private static Dictionary<string, Tensor> Unused() => new()
    {
        ["audio.model.model.rvq.quantizers.0.codebook.weight"] = randn(4, 4),
        ["audio.model.model.rvq.quantizers.0.in_proj.weight"] = randn(3, 3),
        ["audio.model.model.cls_token"] = randn(16),
        ["audio.model.model.linear.weight"] = randn(20, 16),
        ["audio.model.model.linear.bias"] = randn(20),
        ["audio.model.model.conformer.pos_conv_embed.conv.weight_g"] = randn(1, 1, 5),
        ["audio.model.model.conformer.pos_conv_embed.conv.weight_v"] = randn(16, 8, 5),
        ["audio.model.model.conformer.pos_conv_embed.conv.bias"] = randn(16),
        ["audio.aggregator.weight"] = randn(1, 25, 1),
        ["audio.aggregator.bias"] = randn(1),
    };

    private string WriteCheckpoint(Action<Dictionary<string, Tensor>>? edit = null)
    {
        using var _ = no_grad();
        manual_seed(9);
        var source = TinyTower();
        var file = source.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        foreach (var (name, t) in file.ToList())
        {
            if (t.dtype == ScalarType.Float32) t.normal_(0, 0.3);
            if (name.EndsWith("running_var", StringComparison.Ordinal)) t.fill_(1.0);
        }
        foreach (var (k, v) in Unused()) file[k] = v;
        edit?.Invoke(file);
        var path = Path.Combine(tmp, MuLanAudioWeights.CheckpointFile);
        SafeTensorsWriter.Save(path, file);
        return path;
    }

    private InvalidDataException LoadFails(Action<Dictionary<string, Tensor>> edit) =>
        Assert.Throws<InvalidDataException>(() => MuLanAudioWeights.Load(TinyTower(), CheckpointIndex.Open(WriteCheckpoint(edit)), CPU));

    [Fact]
    public void ModelKeys_AreTheCheckpointKeys()
    {
        var keys = TinyTower().state_dict().Keys.ToHashSet();
        foreach (var expected in new[]
        {
            "audio.model.model.conv.conv.0.conv1.weight", "audio.model.model.conv.conv.0.conv1.bias", "audio.model.model.conv.conv.0.bn1.weight",
            "audio.model.model.conv.conv.0.bn1.running_mean", "audio.model.model.conv.conv.0.bn1.running_var", "audio.model.model.conv.conv.0.bn1.num_batches_tracked",
            "audio.model.model.conv.conv.0.conv2.weight", "audio.model.model.conv.conv.0.bn2.bias", "audio.model.model.conv.conv.0.conv3.weight", "audio.model.model.conv.conv.0.bn3.running_var",
            "audio.model.model.conv.conv.1.conv1.weight", "audio.model.model.conv.conv.1.conv3.bias", "audio.model.model.conv.linear.weight", "audio.model.model.conv.linear.bias",
            "audio.model.model.conformer.embed_positions.inv_freq", "audio.model.model.conformer.layer_norm.weight", "audio.model.model.conformer.layer_norm.bias",
            "audio.model.model.conformer.layers.1.ffn1_layer_norm.weight", "audio.model.model.conformer.layers.1.ffn1.intermediate_dense.weight", "audio.model.model.conformer.layers.0.ffn1.output_dense.bias",
            "audio.model.model.conformer.layers.0.self_attn_layer_norm.bias", "audio.model.model.conformer.layers.0.self_attn.linear_q.weight", "audio.model.model.conformer.layers.0.self_attn.linear_k.bias",
            "audio.model.model.conformer.layers.0.self_attn.linear_v.weight", "audio.model.model.conformer.layers.0.self_attn.linear_out.weight",
            "audio.model.model.conformer.layers.0.conv_module.layer_norm.weight", "audio.model.model.conformer.layers.0.conv_module.pointwise_conv1.weight",
            "audio.model.model.conformer.layers.0.conv_module.depthwise_conv.weight", "audio.model.model.conformer.layers.0.conv_module.batch_norm.running_mean",
            "audio.model.model.conformer.layers.0.conv_module.batch_norm.num_batches_tracked", "audio.model.model.conformer.layers.0.conv_module.pointwise_conv2.weight",
            "audio.model.model.conformer.layers.1.ffn2_layer_norm.bias", "audio.model.model.conformer.layers.1.ffn2.intermediate_dense.bias", "audio.model.model.conformer.layers.1.final_layer_norm.weight",
            "audio.proj.weight", "audio.proj.bias", "audio_to_latents.weight", "audio_to_latents.bias",
        })
            Assert.Contains(expected, keys);
        // nothing the reference does not evaluate exists in the model
        Assert.DoesNotContain(keys, MuLanAudioWeights.IsUnused);
        Assert.DoesNotContain(keys, k => k.Contains("pos_conv") || k.Contains("rvq") || k.Contains("cls_token") || k.Contains("aggregator"));
    }

    [Fact]
    public void TheRealSizes_GiveTheCheckpointTensorCount()
    {
        // 525 tensors of MuQ-large: 72 unused by the reference + what the model holds; the model also has audio.proj (2) and audio_to_latents (2)
        var model = new MuQSubsampling(new MuQConfig());
        var keys = model.state_dict();
        Assert.Equal(44, keys.Count);                                   // 2 blocks x (3 convs x 2 + 3 batch norms x 5) + linear 2
        Assert.Equal(new long[] { 1024, 16384 }, keys["linear.weight"].shape);
        Assert.Equal(new long[] { 512, 1, 3, 3 }, keys["conv.0.conv1.weight"].shape);
        Assert.Equal(new long[] { 512, 512, 3, 3 }, keys["conv.1.conv3.weight"].shape);
    }

    [Fact]
    public void Load_MatchingCheckpoint_AssignsEveryTensor_AndReportsTheUnusedOnes()
    {
        var path = WriteCheckpoint();
        var model = TinyTower();
        var report = MuLanAudioWeights.Load(model, CheckpointIndex.Open(path), CPU);
        var unused = Unused().Keys.Order(StringComparer.Ordinal).ToList();
        Assert.Equal(unused, report.IgnoredKeys);
        Assert.Equal(model.state_dict().Count, report.Result.TensorsRead);
        var file = SafeTensors.LoadFile(path);
        foreach (var (name, value) in model.state_dict())
            Assert.True((value.to(ScalarType.Float32) - file[name].to(ScalarType.Float32)).abs().max().item<float>() == 0f, $"{name} was not loaded");
        Assert.False(model.training);
    }

    [Fact]
    public void Load_MissingTensor_NamesIt()
    {
        var e = LoadFails(f => f.Remove("audio.model.model.conformer.layers.1.conv_module.batch_norm.running_var"));
        Assert.Contains("audio.model.model.conformer.layers.1.conv_module.batch_norm.running_var", e.Message);
        Assert.Contains("missing", e.Message);
    }

    [Fact]
    public void Load_MissingRotaryBuffer_NamesIt()
    {
        var e = LoadFails(f => f.Remove("audio.model.model.conformer.embed_positions.inv_freq"));
        Assert.Contains("embed_positions.inv_freq", e.Message);
    }

    [Fact]
    public void Load_ShapeMismatch_NamesTheTensorWithBothShapes()
    {
        var e = LoadFails(f => f["audio_to_latents.weight"] = zeros(3, 5));
        Assert.Contains("audio_to_latents.weight", e.Message);
        Assert.Contains("[6, 12]", e.Message);
        Assert.Contains("[3, 5]", e.Message);
    }

    [Fact]
    public void Load_UnexpectedTensor_IsRejected()
    {
        var e = LoadFails(f => f["audio.model.model.surprise.weight"] = zeros(2));
        Assert.Contains("surprise.weight", e.Message);
    }

    [Fact]
    public void Load_NothingIsAssigned_WhenTheCheckpointDoesNotFit()
    {
        var model = TinyTower();
        var before = model.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        var path = WriteCheckpoint(f => f.Remove("audio.proj.bias"));
        Assert.Throws<InvalidDataException>(() => MuLanAudioWeights.Load(model, CheckpointIndex.Open(path), CPU));
        foreach (var (name, value) in model.state_dict())
            Assert.True((value.to(ScalarType.Float32) - before[name].to(ScalarType.Float32)).abs().max().item<float>() == 0f, $"{name} changed although loading failed");
    }

    [Fact]
    public void Unused_CoversExactlyTheTensorsTheReferenceNeverEvaluates()
    {
        Assert.True(MuLanAudioWeights.IsUnused("audio.model.model.rvq.quantizers.7.codebook.weight"));
        Assert.True(MuLanAudioWeights.IsUnused("audio.model.model.cls_token"));
        Assert.True(MuLanAudioWeights.IsUnused("audio.model.model.linear.bias"));
        Assert.True(MuLanAudioWeights.IsUnused("audio.model.model.conformer.pos_conv_embed.conv.weight_v"));
        Assert.True(MuLanAudioWeights.IsUnused("audio.aggregator.weight"));
        Assert.False(MuLanAudioWeights.IsUnused("audio.model.model.conv.linear.weight"));       // the subsampling's linear is NOT the prediction layer
        Assert.False(MuLanAudioWeights.IsUnused("audio.model.model.conformer.embed_positions.inv_freq"));
        Assert.False(MuLanAudioWeights.IsUnused("audio.proj.weight"));
        Assert.False(MuLanAudioWeights.IsUnused("audio_to_latents.weight"));
    }

    [Fact]
    public void Evaluation_UsesTheStoredBatchStatistics_AndChangesNothing()
    {
        using var _ = no_grad();
        var model = TinyTower();
        MuLanAudioWeights.Load(model, CheckpointIndex.Open(WriteCheckpoint()), CPU);
        manual_seed(3);
        using var x = randn(1, 8, 40);
        var meanBefore = model.state_dict()["audio.model.model.conformer.layers.0.conv_module.batch_norm.running_mean"].detach().clone();
        var countBefore = model.state_dict()["audio.model.model.conv.conv.0.bn1.num_batches_tracked"].item<long>();
        using var a = model.Encode(x);
        using var b = model.Encode(x);
        Assert.True((a - b).abs().max().item<float>() == 0f, "the same input should give the same output in evaluation mode");
        Assert.Equal(0f, (model.state_dict()["audio.model.model.conformer.layers.0.conv_module.batch_norm.running_mean"] - meanBefore).abs().max().item<float>());
        Assert.Equal(countBefore, model.state_dict()["audio.model.model.conv.conv.0.bn1.num_batches_tracked"].item<long>());
        Assert.All(model.parameters(), p => Assert.False(p.requires_grad));
    }

    [Theory]
    [InlineData(40, 10)]
    [InlineData(41, 11)]       // ceil(ceil(41 / 2) / 2)
    [InlineData(8, 2)]
    public void Encode_GivesOneStepPerFourFrames(int frames, int steps)
    {
        using var _ = no_grad();
        var model = TinyTower();
        using var x = randn(2, 8, frames);
        using var y = model.Encode(x);
        Assert.Equal(new long[] { 2, steps, 16 }, y.shape);
        using var z = model.forward(x);
        Assert.Equal(new long[] { 2, steps, 12 }, z.shape);
    }

    [Fact]
    public void TheRealSubsampling_TakesAClipOf1000FramesTo250Steps()
    {
        using var _ = no_grad();
        var sub = new MuQSubsampling(new MuQConfig());
        sub.eval();
        using var x = randn(1, 128, 1000);
        using var y = sub.forward(x);
        Assert.Equal(new long[] { 1, 250, 1024 }, y.shape);
    }

    [Fact]
    public void Rotary_GivesTheSamePositionsTheSameRotation_WhateverTheLength()
    {
        using var _ = no_grad();
        var enc = new ConformerEncoder(new MuQConfig(Hidden: 16, Heads: 2, Layers: 1, Intermediate: 8, ConvKernel: 3));
        using var short5 = enc.Rotary(5);
        using var long12 = enc.Rotary(12);
        Assert.Equal(new long[] { 2, 5, 1, 1, 8 }, short5.shape);
        Assert.Equal(0f, (short5 - long12[.., ..5]).abs().max().item<float>());
        // position 0 rotates nothing: cosine 1, sine 0
        Assert.Equal(1f, long12[0, 0].min().item<float>());
        Assert.Equal(0f, long12[1, 0].abs().max().item<float>());
        // the 32 frequencies of the real configuration are the reference's: 1 / 10000^(2i / 64)
        var real = new ConformerEncoder(new MuQConfig(Layers: 0));
        var inv = real.state_dict()["embed_positions.inv_freq"];
        Assert.Equal(32, inv.shape[0]);
        Assert.Equal(1f, inv[0].item<float>());
        Assert.Equal((float)(1.0 / Math.Pow(10000.0, 2.0 / 64.0)), inv[1].item<float>(), 6);
    }
}

/// <summary>The full fp32 audio tower, loaded once for every real-tower test (1.3 GB; null when the converted file is missing, and those tests are then skipped).</summary>
public sealed class RealMulanAudioFixture : IDisposable
{
    public MuLanAudioTower? Tower { get; }

    public MuLanAudioLoadReport? Report { get; }

    public RealMulanAudioFixture()
    {
        if (MulanAudioReference.WeightsPath is not { } path) return;
        Tower = new MuLanAudioTower();
        Report = MuLanAudioWeights.Load(Tower, CheckpointIndex.Open(path), CPU);
    }

    public void Dispose() => Tower?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class RealMulanAudioCollection : ICollectionFixture<RealMulanAudioFixture>
{
    public const string Name = "RealMulanAudio";
}

/// <summary>The real audio encoder against the reference, stage by stage: each stage is fed the reference's previous stage, so an error is attributed to the module that introduced it.</summary>
[Collection(RealMulanAudioCollection.Name)]
public class MuQEncoderParityTests
{
    private readonly RealMulanAudioFixture fx;
    private readonly ITestOutputHelper log;

    public MuQEncoderParityTests(RealMulanAudioFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    private MuLanAudioTower Tower => fx.Tower!;

    private static Tensor Stage(string name) => MulanAudioReference.Stages[name];

    [RequiresMulanAudio]
    public void Load_ReadsEveryTensorTheModelHolds_AndIgnoresExactly72()
    {
        var report = fx.Report!;
        log.WriteLine($"read {report.Result.TensorsRead} tensors, ignored {report.IgnoredKeys.Count}");
        Assert.Equal(531 - 72, report.Result.TensorsRead);
        Assert.Equal(72, report.IgnoredKeys.Count);
        Assert.Equal(64, report.IgnoredKeys.Count(k => k.StartsWith("audio.model.model.rvq.")));
        Assert.Equal(3, report.IgnoredKeys.Count(k => k.Contains("pos_conv_embed")));
        Assert.Equal(2, report.IgnoredKeys.Count(k => k.StartsWith("audio.aggregator.")));
        Assert.Contains("audio.model.model.cls_token", report.IgnoredKeys);
        Assert.False(Tower.training);
    }

    [RequiresMulanAudio]
    public void Subsampling_MatchesTheReference()
    {
        using var _ = no_grad();
        using var input = MulanAudioReference.Frontend["s25_norm"].unsqueeze(0);
        using var got = Tower.Audio.Muq.Encoder.Subsampling.forward(input);
        Assert.Equal(new long[] { 1, 250, 1024 }, got.shape);
        Reference.AssertClose(log, "subsampling", got[0], Stage("conv"), Tolerances.MuqStage);
    }

    [RequiresMulanAudio]
    public void ConformerLayers_MatchTheReference_EachFedTheReferencesPreviousStage()
    {
        using var _ = no_grad();
        var conformer = Tower.Audio.Muq.Encoder.Conformer;
        using var rotary = conformer.Rotary(250);
        foreach (var (layer, input) in new[] { (1, "conv"), (6, "layer5"), (12, "layer11") })
        {
            using var x = Stage(input).unsqueeze(0);
            using var got = conformer.Layer(layer - 1, x, rotary);
            Reference.AssertClose(log, $"conformer layer {layer}", got[0], Stage($"layer{layer}"), Tolerances.MuqStage);
        }
        using var l12 = Stage("layer12").unsqueeze(0);
        using var final = conformer.FinalNorm(l12);
        Reference.AssertClose(log, "final layer norm", final[0], Stage("last"), Tolerances.MuqStage);
    }

    [RequiresMulanAudio]
    public void TheWholeEncoder_MatchesTheReference_FromTheSpectrogramToTheLastHiddenState()
    {
        using var _ = no_grad();
        using var input = MulanAudioReference.Frontend["s25_norm"].unsqueeze(0);
        using var got = Tower.Encode(input);
        Assert.Equal(new long[] { 1, 250, 1024 }, got.shape);
        Reference.AssertClose(log, "encoder (12 layers chained)", got[0], Stage("last"), Tolerances.MuqChain);
    }

    [RequiresMulanAudio]
    public void Encoding_IsRepeatable()
    {
        using var _ = no_grad();
        using var input = MulanAudioReference.Frontend["noise10_norm"].unsqueeze(0);
        using var a = Tower.Encode(input);
        using var b = Tower.Encode(input);
        Assert.Equal(0f, (a - b).abs().max().item<float>());
    }

    [RequiresMulanAudio]
    public void AClipOf8Seconds_GivesTwoHundredSteps()
    {
        using var _ = no_grad();
        using var input = MulanAudioReference.Frontend["s4_norm"].unsqueeze(0);
        using var got = Tower.Encode(input);
        Assert.Equal(new long[] { 1, 200, 1024 }, got.shape);
    }
}
