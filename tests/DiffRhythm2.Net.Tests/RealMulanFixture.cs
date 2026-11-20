using DiffRhythm2.StyleText;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The full fp32 MuLan text tower and its tokenizer, loaded once for every real-tower test (1.3 GB; null when the converted files are missing, and those tests are then skipped).</summary>
public sealed class RealMulanFixture : IDisposable
{
    public StyleEncoder? Encoder { get; }

    public RealMulanFixture()
    {
        if (MulanTextReference.WeightsPath is null || MulanTextReference.TokenizerPath is null) return;
        Encoder = StyleEncoder.LoadDirectory(Checkpoint.Dir!, CPU);
    }

    public void Dispose() => Encoder?.Dispose();
}

[CollectionDefinition(Name)]
public sealed class RealMulanCollection : ICollectionFixture<RealMulanFixture>
{
    public const string Name = "RealMulan";
}
