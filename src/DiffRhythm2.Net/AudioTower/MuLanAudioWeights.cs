using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.AudioTower;

/// <summary>What <see cref="MuLanAudioWeights.Load"/> did: the core's counts plus the checkpoint tensors it deliberately did not read.</summary>
public sealed record MuLanAudioLoadReport(StrictLoadResult Result, IReadOnlyList<string> IgnoredKeys);

/// <summary>Loads the converted audio tower (<c>diffrhythm2 convert --audio</c>, or <c>tools/convert_mulan_audio.py</c>).</summary>
public static class MuLanAudioWeights
{
    /// <summary>Name of the converted audio tower inside the model directory.</summary>
    public const string CheckpointFile = "mulan_audio.safetensors";

    /// <summary>
    /// The tensors the reference never evaluates when it extracts features: MuQ-large's random-projection quantizer (64), its classification token (1) and its
    /// prediction layer to codebook logits (2), the convolutional positional embedding that the Conformer constructs but never calls (3), and MuLan's layer
    /// aggregator (2), which is not used when the last layer is taken.
    /// </summary>
    public static bool IsUnused(string modelKey) =>
        modelKey.StartsWith("audio.model.model.rvq.", StringComparison.Ordinal)
        || modelKey == "audio.model.model.cls_token"
        || modelKey.StartsWith("audio.model.model.linear.", StringComparison.Ordinal)
        || modelKey.StartsWith("audio.model.model.conformer.pos_conv_embed.", StringComparison.Ordinal)
        || modelKey.StartsWith("audio.aggregator.", StringComparison.Ordinal);

    /// <summary>
    /// Reads the checkpoint into <paramref name="model"/>, parameters and buffers (batch-norm statistics and the rotary frequencies). If it and the model disagree on any
    /// name or shape nothing is read and the exception lists every discrepancy; the tensors that are never evaluated are the only allowed extras and are returned by name.
    /// </summary>
    public static MuLanAudioLoadReport Load(MuLanAudioTower model, CheckpointIndex checkpoint, Device device)
    {
        var result = StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The MuLan audio checkpoint does not match the tower:")
        {
            Skip = IsUnused,
            UnexpectedLabel = "checkpoint keys",
            IncludeBuffers = true,
        });
        var ignored = checkpoint.Tensors.Keys.Where(IsUnused).Order(StringComparer.Ordinal).ToList();
        model.eval();
        return new MuLanAudioLoadReport(result, ignored);
    }
}
