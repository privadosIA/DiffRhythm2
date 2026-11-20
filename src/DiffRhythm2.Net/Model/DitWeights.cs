using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Model;

/// <summary>What <see cref="DitWeights.Load"/> did: the core's counts plus the checkpoint tensors it deliberately did not read.</summary>
public sealed record DitLoadReport(StrictLoadResult Result, IReadOnlyList<string> IgnoredKeys);

public static class DitWeights
{
    private const string Prefix = "transformer.";

    /// <summary>Checkpoint tensors that exist only for training (the REPA representation-alignment heads); the reference never evaluates them.</summary>
    public static bool IsTrainingOnly(string modelKey) => modelKey.StartsWith("projectors.", StringComparison.Ordinal);

    /// <summary>
    /// Loads <c>model.safetensors</c> into <paramref name="model"/>. If the checkpoint and the model disagree on any name or shape, nothing is read and the
    /// exception lists every discrepancy. The tensors are read as float32; <paramref name="modelDtype"/> converts the model while it is loaded (fp16 is only an informational path). The training-only tensors are the only allowed extras and are returned by name.
    /// </summary>
    public static DitLoadReport Load(Dit model, CheckpointIndex checkpoint, Device device, ScalarType? modelDtype = null)
    {
        var result = StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The DiffRhythm2 checkpoint does not match the DiT:")
        {
            Prefix = Prefix,
            Skip = IsTrainingOnly,
            ModelDtype = modelDtype,
            UnexpectedLabel = "checkpoint keys",
        });
        var ignored = checkpoint.Tensors.Keys
            .Where(k => k.StartsWith(Prefix, StringComparison.Ordinal) && IsTrainingOnly(k[Prefix.Length..]))
            .Order(StringComparer.Ordinal).ToList();
        return new DitLoadReport(result, ignored);
    }

    /// <summary>Builds the model from <c>config.json</c> and loads <c>model.safetensors</c> from a checkpoint directory.</summary>
    public static (Dit Model, DitLoadReport Report) LoadDirectory(string dir, Device device, ScalarType? modelDtype = null)
    {
        var model = new Dit(DitConfig.Load(Path.Combine(dir, "config.json")));
        var report = Load(model, CheckpointIndex.Open(dir), device, modelDtype);
        return (model, report);
    }
}
