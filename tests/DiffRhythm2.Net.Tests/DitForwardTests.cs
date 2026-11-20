using DiffRhythm2.Model;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Whole-DiT forward parity with the fp32 golden run, for the three kinds of input (text prefill, noisy block, clean-block write).</summary>
[Collection(RealModelCollection.Name)]
public class DitForwardParityTests
{
    private readonly RealDitFixture fx;
    private readonly ITestOutputHelper log;

    public DitForwardParityTests(RealDitFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    [RequiresCheckpoint]
    public void NoCache_NoisyBlock_MatchesTheReference()
    {
        using var _ = no_grad();
        var m = fx.Model!;
        var c = Reference.Cases;
        var pred = m.Forward(m.EmbedLatent(c["a_noisy"]), full(new long[] { 1, 10 }, 0.3f), Reference.Inputs["style"], null, null, CacheUse.Transient);
        Reference.AssertClose(log, "a_pred", pred, c["a_pred"], Tolerances.Piece);
    }

    [RequiresCheckpoint]
    public void WithCache_ConditionalBranch_MatchesTheReference() => CachedBranch("b", Reference.Inputs["tokens"], Reference.Inputs["style"]);

    [RequiresCheckpoint]
    public void WithCache_UnconditionalBranch_ZeroedTextAndStyle_MatchesTheReference()
    {
        var tokens = Reference.Inputs["tokens"];
        CachedBranch("n", zeros_like(tokens), zeros_like(Reference.Inputs["style"]));
    }

    /// <summary>Text prefill at time -1, one clean block at time 1 (positions 0..9), then a noisy block at t=0.5 (positions 10..19), as in sample_block_cache.</summary>
    private void CachedBranch(string prefix, Tensor tokens, Tensor style)
    {
        using var _ = no_grad();
        var m = fx.Model!;
        var c = Reference.Cases;
        long L = tokens.shape[0];
        using var cache = new BlockCache(m.Config.Depth);

        m.Prefill(m.EmbedText(tokens.unsqueeze(0)), full(new long[] { 1, L }, -1f), style, arange(L).unsqueeze(0), cache, CacheUse.TextPrefill);
        m.Prefill(m.EmbedLatent(c["b_clean_block"]), ones(1, 10), style, arange(10).unsqueeze(0), cache, CacheUse.CleanWrite);
        Assert.Equal(L, cache.TextLength(15));
        Assert.Equal(10, cache.CleanLength(15));
        if (prefix == "b")
        {
            foreach (var layer in new[] { 0, 15 })
            {
                Reference.AssertClose(log, $"text keys l{layer}", cache.TextKeys(layer)!, c[$"b_text_k{layer}"], Tolerances.Piece);
                Reference.AssertClose(log, $"text values l{layer}", cache.TextValues(layer)!, c[$"b_text_v{layer}"], Tolerances.Piece);
                Reference.AssertClose(log, $"clean keys l{layer}", cache.CleanKeys(layer)!, c[$"b_clean_k{layer}"], Tolerances.Piece);
                Reference.AssertClose(log, $"clean values l{layer}", cache.CleanValues(layer)!, c[$"b_clean_v{layer}"], Tolerances.Piece);
            }
        }

        var pred = m.Forward(m.EmbedLatent(c["b_noisy"]), full(new long[] { 1, 10 }, 0.5f), style, arange(10, 20).unsqueeze(0), cache, CacheUse.Transient);
        Assert.Equal(10, cache.CleanLength(15));                       // the noisy block was not stored
        Reference.AssertClose(log, $"{prefix}_pred", pred, c[$"{prefix}_pred"], Tolerances.Piece);
    }

    [RequiresCheckpoint]
    public void GuidanceBranchesDiffer_SoTheNullBranchIsReallyUnconditional()
    {
        var c = Reference.Cases;
        Assert.True((c["b_pred"] - c["n_pred"]).abs().max().item<float>() > 0.1f);
    }
}

/// <summary>Structural properties that need no checkpoint (a tiny random model).</summary>
public class DitStructureTests
{
    private static Dit TinyModel()
    {
        manual_seed(3);
        return new Dit(new DitConfig(Dim: 32, Depth: 2, Heads: 2, FfMult: 2, MelDim: 8, TextNumEmbeds: 10, BlockSize: 10));
    }

    [Fact]
    public void Attention_IsNonCausalWithinABlock()
    {
        using var _ = no_grad();
        var m = TinyModel();
        var style = randn(1, 512);
        var time = full(new long[] { 1, 10 }, 0.5f);
        var latent = randn(1, 10, 8);
        var baseline = m.Forward(m.EmbedLatent(latent), time, style, null, null, CacheUse.Transient);

        var perturbed = latent.clone();
        perturbed[0, 6] = perturbed[0, 6] + 1;                         // change one frame
        var changed = m.Forward(m.EmbedLatent(perturbed), time, style, null, null, CacheUse.Transient);

        var diff = (changed - baseline).abs().amax(new long[] { 0, 2 });   // per position
        for (int pos = 0; pos < 10; pos++)
            Assert.True(diff[pos].item<float>() > 1e-6f, $"position {pos} did not react to a change at position 6: a causal mask would leave positions before it unchanged");
    }

    [Fact]
    public void Prefill_DoesNotNeedTheOutputHead_AndWritesTheCache()
    {
        using var _ = no_grad();
        var m = TinyModel();
        using var cache = new BlockCache(2);
        m.Prefill(m.EmbedText(tensor(new long[,] { { 1, 2, 3 } })), full(new long[] { 1, 3 }, -1f), randn(1, 512), null, cache, CacheUse.TextPrefill);
        Assert.Equal(3, cache.TextLength(0));
        Assert.Equal(3, cache.TextLength(1));
    }
}
