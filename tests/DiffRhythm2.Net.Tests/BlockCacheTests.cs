using DiffRhythm2.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

public class BlockCacheTests
{
    // [b=1, heads=2, n, headDim=4] filled with a constant so the order of the sections can be read off the values
    private static Tensor Kv(long n, float v) => full(new long[] { 1, 2, n, 4 }, v);

    [Fact]
    public void TextPrefill_StoresAndAttendsOnlyToItself()
    {
        using var c = new BlockCache(1);
        var (k, v) = c.Update(0, Kv(5, 1), Kv(5, 1), CacheUse.TextPrefill);
        Assert.Equal(5, k.shape[2]);
        Assert.Equal(5, c.TextLength(0));
        Assert.Equal(0, c.CleanLength(0));
        Assert.Equal(5, v.shape[2]);
    }

    [Fact]
    public void Transient_ReadsTextCleanAndCurrentInOrderWithoutStoring()
    {
        using var c = new BlockCache(1);
        c.Update(0, Kv(5, 1), Kv(5, 1), CacheUse.TextPrefill);
        c.Update(0, Kv(10, 2), Kv(10, 2), CacheUse.CleanWrite);
        var (k, _) = c.Update(0, Kv(10, 3), Kv(10, 3), CacheUse.Transient);
        Assert.Equal(25, k.shape[2]);
        Assert.Equal(1f, k[0, 0, 0, 0].item<float>());
        Assert.Equal(2f, k[0, 0, 5, 0].item<float>());
        Assert.Equal(3f, k[0, 0, 15, 0].item<float>());
        Assert.Equal(10, c.CleanLength(0));      // nothing stored by the transient call
        Assert.Equal(5, c.TextLength(0));
    }

    [Fact]
    public void CleanWrite_AccumulatesAndReturnsTextPlusAllCleanBlocks()
    {
        using var c = new BlockCache(1);
        c.Update(0, Kv(5, 1), Kv(5, 1), CacheUse.TextPrefill);
        c.Update(0, Kv(10, 2), Kv(10, 2), CacheUse.CleanWrite);
        var (k, _) = c.Update(0, Kv(10, 3), Kv(10, 3), CacheUse.CleanWrite);
        Assert.Equal(25, k.shape[2]);
        Assert.Equal(20, c.CleanLength(0));
    }

    [Fact]
    public void WithoutText_TheCleanBlocksAreStillUsable()
    {
        using var c = new BlockCache(1);
        var (k, _) = c.Update(0, Kv(10, 2), Kv(10, 2), CacheUse.CleanWrite);
        Assert.Equal(10, k.shape[2]);
        Assert.Equal(0, c.TextLength(0));
        var (k2, _) = c.Update(0, Kv(10, 3), Kv(10, 3), CacheUse.Transient);
        Assert.Equal(20, k2.shape[2]);
    }

    [Fact]
    public void LayersAreIndependent()
    {
        using var c = new BlockCache(2);
        c.Update(0, Kv(5, 1), Kv(5, 1), CacheUse.TextPrefill);
        Assert.Equal(0, c.TextLength(1));
    }

    [Fact]
    public void TheCacheOwnsItsTensors()
    {
        using var c = new BlockCache(1);
        var k = Kv(10, 2);
        c.Update(0, k, Kv(10, 2), CacheUse.CleanWrite);
        k.Dispose();
        Assert.Equal(2f, c.CleanKeys(0)![0, 0, 0, 0].item<float>());
    }
}
