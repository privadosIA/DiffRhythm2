using DiffRhythm2.Pipeline;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The noise against what the reference draws on CPU with seed 1234, and the reproducibility rules.</summary>
public class NoiseSourceTests
{
    private static Tensor Stack(List<Tensor> blocks) => stack(blocks);

    [Fact]
    public void Float32Blocks_EqualTheReferencesCpuDraw()
    {
        using var got = Stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float32, CPU));
        var expected = SongReference.Noise["noise_f32"];
        Assert.Equal(expected.shape, got.shape);
        Assert.Equal(0f, (got - expected).abs().max().item<float>());
    }

    [Fact]
    public void Float16Blocks_EqualTheReferencesCpuDraw_AndAreNotTheCastFloat32Draw()
    {
        using var half = Stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float16, CPU));
        Assert.Equal(ScalarType.Float16, half.dtype);
        Assert.Equal(0f, (half.to(ScalarType.Float32) - SongReference.Noise["noise_f16"]).abs().max().item<float>());

        // what the reference would get by drawing float32 and casting: a different noise, which is why the dtype of the draw matters
        using var cast = Stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float32, CPU)).to(ScalarType.Float16);
        Assert.Equal(0f, (cast.to(ScalarType.Float32) - SongReference.Noise["noise_f16_via_cast"]).abs().max().item<float>());
        Assert.True((cast.to(ScalarType.Float32) - half.to(ScalarType.Float32)).abs().max().item<float>() > 1f);
    }

    [Fact]
    public void SameSeed_SameNoise_AnotherSeed_AnotherNoise()
    {
        using var a = Stack(NoiseSource.Draw(77, 3, ScalarType.Float32, CPU));
        using var b = Stack(NoiseSource.Draw(77, 3, ScalarType.Float32, CPU));
        using var c = Stack(NoiseSource.Draw(78, 3, ScalarType.Float32, CPU));
        Assert.Equal(0f, (a - b).abs().max().item<float>());
        Assert.True((a - c).abs().max().item<float>() > 0.1f);
    }

    [Fact]
    public void DrawingMoreBlocks_DoesNotChangeTheFirstOnes()
    {
        using var four = Stack(NoiseSource.Draw(SongReference.Seed, 4, ScalarType.Float32, CPU));
        using var six = Stack(NoiseSource.Draw(SongReference.Seed, 6, ScalarType.Float32, CPU));
        Assert.Equal(0f, (four - six.narrow(0, 0, 4)).abs().max().item<float>());
        Assert.Equal(new long[] { 6, 1, 10, 64 }, six.shape);
    }

    [Fact]
    public void NewSeed_IsNonNegative_AndFitsTheRequestRule()
    {
        for (int i = 0; i < 50; i++)
        {
            var seed = NoiseSource.NewSeed();
            Assert.InRange(seed, 0L, uint.MaxValue);
            new SongRequest("hello", "Pop", Seed: seed).Validate();
        }
    }
}
