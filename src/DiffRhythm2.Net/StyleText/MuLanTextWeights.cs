using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.StyleText;

/// <summary>What <see cref="MuLanTextWeights.Load"/> did: the core's counts plus the checkpoint tensors it deliberately did not read.</summary>
public sealed record MuLanLoadReport(StrictLoadResult Result, IReadOnlyList<string> IgnoredKeys);

/// <summary>Loads the converted text tower (<c>tools/convert_mulan_text.py</c>).</summary>
public static class MuLanTextWeights
{
    public const string CheckpointFile = "mulan_text.safetensors";
    public const string TokenizerFile = "mulan_text_tokenizer.json";

    /// <summary>The pooler of the XLM-RoBERTa checkpoint is never evaluated (only the last hidden state is used).</summary>
    public static bool IsUnused(string modelKey) => modelKey.StartsWith("text.model.pooler.", StringComparison.Ordinal);

    /// <summary>
    /// Reads the checkpoint into <paramref name="model"/>. If it and the model disagree on any name or shape nothing is read and the exception lists
    /// every discrepancy; the unused pooler tensors are the only allowed extras and are returned by name.
    /// </summary>
    public static MuLanLoadReport Load(MuLanTextTower model, CheckpointIndex checkpoint, Device device)
    {
        var result = StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The MuLan text checkpoint does not match the tower:")
        {
            Skip = IsUnused,
            UnexpectedLabel = "checkpoint keys",
        });
        var ignored = checkpoint.Tensors.Keys.Where(IsUnused).Order(StringComparer.Ordinal).ToList();
        return new MuLanLoadReport(result, ignored);
    }
}
