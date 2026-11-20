using System.Text.Json;

namespace DiffRhythm2.Decoder;

/// <summary>
/// The parts of <c>decoder.json</c> that shape the generator. Anything the port does not implement (the plain <c>snake</c> activation, residual
/// blocks other than kind 1) is rejected by <see cref="Load"/> with the offending value in the message, instead of producing a model that does not
/// match the checkpoint.
/// </summary>
public sealed record BigVganConfig(
    int InChannels,
    int UpsampleInitialChannel,
    int[] UpsampleRates,
    int[] UpsampleKernelSizes,
    int[] ResblockKernelSizes,
    int[][] ResblockDilationSizes,
    bool SnakeLogscale,
    bool UseBiasAtFinal,
    bool UseTanhAtFinal,
    int SamplingRate)
{
    /// <summary>Output samples per input latent frame: the product of the upsampling rates (9600 for the published configuration).</summary>
    public int SamplesPerFrame => UpsampleRates.Aggregate(1, (a, b) => a * b);

    public static BigVganConfig Load(string path) => Parse(File.ReadAllText(path));

    public static BigVganConfig Parse(string json)
    {
        using var doc = JsonDocument.Parse(json, new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        var r = doc.RootElement;

        var resblock = r.GetProperty("resblock").GetString();
        if (resblock != "1") throw new NotSupportedException($"decoder.json: resblock \"{resblock}\" is not supported (only \"1\", AMPBlock1)");
        var activation = r.GetProperty("activation").GetString();
        if (activation != "snakebeta") throw new NotSupportedException($"decoder.json: activation \"{activation}\" is not supported (only \"snakebeta\")");

        int[] Ints(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetInt32()).ToArray();
        bool Flag(string name, bool fallback) => r.TryGetProperty(name, out var e) ? e.GetBoolean() : fallback;

        var cfg = new BigVganConfig(
            r.GetProperty("in_channels").GetInt32(),
            r.GetProperty("upsample_initial_channel").GetInt32(),
            Ints("upsample_rates"),
            Ints("upsample_kernel_sizes"),
            Ints("resblock_kernel_sizes"),
            r.GetProperty("resblock_dilation_sizes").EnumerateArray().Select(a => a.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray(),
            r.GetProperty("snake_logscale").GetBoolean(),
            Flag("use_bias_at_final", true),
            Flag("use_tanh_at_final", true),   // the reference's defaults when the keys are absent
            r.GetProperty("sampling_rate").GetInt32());

        if (cfg.UpsampleRates.Length != cfg.UpsampleKernelSizes.Length)
            throw new InvalidDataException("decoder.json: upsample_rates and upsample_kernel_sizes differ in length");
        if (cfg.ResblockKernelSizes.Length != cfg.ResblockDilationSizes.Length)
            throw new InvalidDataException("decoder.json: resblock_kernel_sizes and resblock_dilation_sizes differ in length");
        return cfg;
    }
}
