using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The golden-run latent that the decoder change (add-bigvgan-decoder) takes as input.</summary>
public class LatentFixtureTests
{
    private static Tensor Load() =>
        new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "latent")).Load("latent_40.safetensors")["latent"];

    [Fact]
    public void Latent_HasTheSamplerLayout_AndIsTheGoldenRunOutput()
    {
        var latent = Load();
        Assert.Equal(new long[] { 1, 40, 64 }, latent.shape);
        Assert.Equal(ScalarType.Float32, latent.dtype);
        Assert.Equal(0f, (latent - Reference.Run["latent"]).abs().max().item<float>());
    }

    [Fact]
    public void Transposed_IsTheChannelsFirstLayoutTheDecoderExpects()
    {
        Assert.Equal(new long[] { 1, 64, 40 }, Load().transpose(1, 2).shape);
    }
}
