using DiffRhythm2.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The full fp32 DiT, loaded once for every real-model test (4.3 GB; null when there is no checkpoint, in which case those tests are skipped).</summary>
public sealed class RealDitFixture : IDisposable
{
    public Dit? Model { get; }
    public DitLoadReport? Report { get; }

    public RealDitFixture()
    {
        if (Checkpoint.Dir is not { } dir) return;
        (Model, Report) = DitWeights.LoadDirectory(dir, CPU);
    }

    public void Dispose() => Model?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class RealModelCollection : ICollectionFixture<RealDitFixture>
{
    public const string Name = "RealModel";
}
