using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Decoder;

/// <summary>Loads the converted decoder checkpoint (<c>tools/convert_decoder.py</c>: <c>decoder.bin</c> to <c>decoder.safetensors</c>).</summary>
public static class BigVganWeights
{
    public const string CheckpointFile = "decoder.safetensors";
    public const string ConfigFile = "decoder.json";

    /// <summary>
    /// Reads <paramref name="checkpoint"/> into <paramref name="model"/>: weight-norm pairs (<c>….parametrizations.weight.original0/1</c>) are folded
    /// into plain weights, the anti-alias filters (buffers) are read like parameters, and any difference in names or shapes throws with the full list
    /// before anything is read. The final convolution has no bias in the model when the configuration says so, so a checkpoint that still has one is
    /// reported as an unexpected key rather than silently dropped.
    /// </summary>
    public static StrictLoadResult Load(BigVganGenerator model, CheckpointIndex checkpoint, Device device, ScalarType? modelDtype = null) =>
        StrictLoader.Load(model, checkpoint, device, new StrictLoadOptions("The decoder checkpoint does not match the BigVGAN generator:")
        {
            FoldWeightNorm = true,
            IncludeBuffers = true,
            ModelDtype = modelDtype,
            UnexpectedLabel = "checkpoint keys",
        });

    /// <summary>Builds the generator from <c>decoder.json</c> and loads <c>decoder.safetensors</c> from a checkpoint directory.</summary>
    public static BigVganGenerator LoadDirectory(string dir, Device device, ScalarType? modelDtype = null)
    {
        var model = new BigVganGenerator(BigVganConfig.Load(Path.Combine(dir, ConfigFile)));
        Load(model, CheckpointIndex.Open(Path.Combine(dir, CheckpointFile)), device, modelDtype);
        return model;
    }
}
