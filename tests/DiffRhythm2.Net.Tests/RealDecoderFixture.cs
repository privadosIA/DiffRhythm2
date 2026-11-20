using DiffRhythm2.Decoder;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The full fp32 BigVGAN generator, loaded once for every real-decoder test (null when the checkpoint or the converted file is missing; those tests are then skipped).</summary>
public sealed class RealDecoderFixture : IDisposable
{
    public BigVganGenerator? Model { get; }
    public StrictLoadResult? Report { get; }

    public RealDecoderFixture()
    {
        if (RequiresDecoderAttribute.MissingReason() is not null) return;
        var dir = Checkpoint.Dir!;
        Model = new BigVganGenerator(BigVganConfig.Load(Path.Combine(dir, BigVganWeights.ConfigFile)));
        Report = BigVganWeights.Load(Model, CheckpointIndex.Open(Path.Combine(dir, BigVganWeights.CheckpointFile)), CPU);
    }

    public void Dispose() => Model?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class RealDecoderCollection : ICollectionFixture<RealDecoderFixture>
{
    public const string Name = "RealDecoder";
}

/// <summary>[Fact] that needs the checkpoint directory and the converted <c>decoder.safetensors</c> in it (see tools/convert_decoder.py).</summary>
public sealed class RequiresDecoderAttribute : FactAttribute
{
    public RequiresDecoderAttribute() => Skip = MissingReason();

    public static string? MissingReason()
    {
        if (Checkpoint.Env.NoCheckpointReason is { } reason) return reason;
        var dir = Checkpoint.Dir!;
        return File.Exists(Path.Combine(dir, BigVganWeights.CheckpointFile)) && File.Exists(Path.Combine(dir, BigVganWeights.ConfigFile))
            ? null
            : $"{BigVganWeights.CheckpointFile} / {BigVganWeights.ConfigFile} not found in the checkpoint directory (convert decoder.bin with tools/convert_decoder.py)";
    }
}

/// <summary>As <see cref="RequiresDecoderAttribute"/>, and only on CPU (the test reads the host's resident set).</summary>
public sealed class RequiresCpuDecoderAttribute : FactAttribute
{
    public RequiresCpuDecoderAttribute() =>
        Skip = RequiresDecoderAttribute.MissingReason() ?? (Checkpoint.DeviceName != "cpu" ? $"only applies on CPU (DIFFRHYTHM2_DEVICE={Checkpoint.DeviceName})" : null);
}
