using DiffRhythm2.Model;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Configuration, strict loading (on a tiny synthetic checkpoint and on the real one) and the names/shapes of the model.</summary>
public class DitModelTests : IDisposable
{
    private static readonly DitConfig Tiny = new(Dim: 32, Depth: 2, Heads: 2, FfMult: 2, MelDim: 8, TextNumEmbeds: 10, BlockSize: 10);
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-dit-").FullName;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    /// <summary>Writes a checkpoint for <see cref="Tiny"/> with random weights, the training-only projector tensors, and an optional edit; returns the source model's state.</summary>
    private Dictionary<string, Tensor> WriteCheckpoint(Action<Dictionary<string, Tensor>>? edit = null)
    {
        using var _ = no_grad();
        manual_seed(5);
        var source = new Dit(Tiny);
        var state = source.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        foreach (var t in state.Values) t.normal_();
        var file = state.ToDictionary(kv => "transformer." + kv.Key, kv => kv.Value);
        file["transformer.projectors.0.0.weight"] = randn(4, 4);
        file["transformer.projectors.0.0.bias"] = randn(4);
        edit?.Invoke(file);
        SafeTensorsWriter.Save(Path.Combine(tmp, "model.safetensors"), file);
        return state;
    }

    private Exception LoadFails(Action<Dictionary<string, Tensor>> edit)
    {
        WriteCheckpoint(edit);
        return Assert.Throws<InvalidDataException>(() => DitWeights.Load(new Dit(Tiny), CheckpointIndex.Open(tmp), CPU));
    }

    [Fact]
    public void Load_MatchingCheckpoint_AssignsEveryTensorAndReportsTheIgnoredOnes()
    {
        var expected = WriteCheckpoint();
        var model = new Dit(Tiny);
        var report = DitWeights.Load(model, CheckpointIndex.Open(tmp), CPU);

        Assert.Equal(expected.Count, report.Result.TensorsRead);
        Assert.Equal(new[] { "transformer.projectors.0.0.bias", "transformer.projectors.0.0.weight" }, report.IgnoredKeys);
        foreach (var (name, value) in model.state_dict())
            Assert.True((value - expected[name]).abs().max().item<float>() == 0f, $"{name} was not loaded");
    }

    [Fact]
    public void Load_MissingTensor_NamesIt()
    {
        var e = LoadFails(f => f.Remove("transformer.transformer_blocks.1.mlp.up_proj.weight"));
        Assert.Contains("transformer_blocks.1.mlp.up_proj.weight", e.Message);
        Assert.Contains("missing", e.Message);
    }

    [Fact]
    public void Load_ShapeMismatch_NamesTheTensorWithBothShapes()
    {
        var e = LoadFails(f => f["transformer.proj_out.weight"] = zeros(3, 5));
        Assert.Contains("proj_out.weight", e.Message);
        Assert.Contains("[8, 32]", e.Message);
        Assert.Contains("[3, 5]", e.Message);
    }

    [Fact]
    public void Load_UnexpectedTensor_IsRejected()
    {
        var e = LoadFails(f => f["transformer.surprise.weight"] = zeros(2));
        Assert.Contains("surprise.weight", e.Message);
    }

    [Fact]
    public void Config_RejectsMissingKeys()
    {
        var path = Path.Combine(tmp, "config.json");
        File.WriteAllText(path, "{\"dim\": 32}");
        Assert.Contains("depth", Assert.Throws<InvalidDataException>(() => DitConfig.Load(path)).Message);
    }

    [RequiresCheckpoint]
    public void Config_OfThePublishedCheckpoint()
    {
        var c = DitConfig.Load(Path.Combine(Checkpoint.Dir!, "config.json"));
        Assert.Equal(new DitConfig(2048, 16, 16, 4, 64, 1000, 10), c);
        Assert.Equal(128, c.HeadDim);
    }

    [RequiresCheckpoint]
    public void Model_HasTheCheckpointParameterNamesAndShapes()
    {
        var ck = CheckpointIndex.Open(Checkpoint.Dir!);
        var model = new Dit(DitConfig.Load(Path.Combine(Checkpoint.Dir!, "config.json")));
        var fromModel = model.named_parameters().ToDictionary(p => p.name, p => p.parameter.shape);
        var fromFile = ck.Tensors.Where(kv => !DitWeights.IsTrainingOnly(kv.Key["transformer.".Length..]))
            .ToDictionary(kv => kv.Key["transformer.".Length..], kv => kv.Value.Shape);
        Assert.Equal(fromFile.Keys.Order(), fromModel.Keys.Order());
        foreach (var (k, shape) in fromFile) Assert.Equal(shape, fromModel[k]);
    }
}

[Collection(RealModelCollection.Name)]
public class DitFullCheckpointTests
{
    private readonly RealDitFixture fx;
    public DitFullCheckpointTests(RealDitFixture fx) => this.fx = fx;

    [RequiresCheckpoint]
    public void Load_PublishedCheckpoint_ReadsEveryInferenceTensorAndIgnoresOnlyTheTrainingHeads()
    {
        var report = fx.Report!;
        Assert.Equal(193, report.Result.TensorsRead);               // 205 tensors in the file minus the 12 of the two REPA projectors
        Assert.Equal(12, report.IgnoredKeys.Count);
        Assert.All(report.IgnoredKeys, k => Assert.StartsWith("transformer.projectors.", k));
        Assert.Equal(report.Result.ModelTensors, report.Result.TensorsRead);
    }
}
