using MusicGen.Core.Layers;
using MusicGen.Core.Weights;
using DiffRhythm2.Model;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Layer-level parity of the DiT pieces against the fp32 golden run (case a: no cache; case b: populated cache).</summary>
public class ModelBlocksTests
{
    private readonly ITestOutputHelper log;
    public ModelBlocksTests(ITestOutputHelper log) => this.log = log;

    private static CheckpointIndex Ckpt() => CheckpointIndex.Open(Checkpoint.Dir!);
    private static RotaryEmbedding Rope() =>
        new(RopeLayout.HalfRotation, RopeFrequencies.Plain(10000), RopeTable.Float32Cached, headDim: 128, maxSeqLen: 4096);

    [RequiresCheckpoint]
    public void LlamaRmsNorm_MatchesTheReferenceLayerNorm()
    {
        using var _ = no_grad();
        var norm = new LlamaRmsNorm(2048);
        Reference.LoadInto(norm, Ckpt(), "transformer.transformer_blocks.0.input_layernorm.");
        Reference.AssertClose(log, "l0 input_layernorm", norm.forward(Reference.Cases["a_l0_ln_in"]), Reference.Cases["a_l0_ln_out"], Tolerances.Piece);
    }

    [Fact]
    public void LlamaRmsNorm_AppliesTheWeightInTheInputDtype()
    {
        using var _ = no_grad();
        var norm = new LlamaRmsNorm(4);
        norm.state_dict()["weight"].copy_(tensor(new[] { 0.5f, 1.5f, -1f, 2f }));
        norm.to(ScalarType.BFloat16);
        var x = tensor(new[,] { { 1f, 2f, 3f, 4f } }).to(ScalarType.BFloat16);
        var xf = x.to(ScalarType.Float32);
        var normed = (xf * rsqrt(xf.pow(2).mean(new long[] { -1 }, keepdim: true) + 1e-6)).to(ScalarType.BFloat16);   // variance in fp32, then back to the input dtype
        var expected = norm.state_dict()["weight"] * normed;                                                           // weight multiplied in bf16
        var got = norm.forward(x);
        Assert.Equal(ScalarType.BFloat16, got.dtype);
        Assert.Equal(0f, (got.to(ScalarType.Float32) - expected.to(ScalarType.Float32)).abs().max().item<float>());
    }

    [RequiresCheckpoint]
    public void LlamaMlp_MatchesTheReferenceLayerMlp()
    {
        using var _ = no_grad();
        var mlp = new LlamaMlp(2048, 8192);
        Reference.LoadInto(mlp, Ckpt(), "transformer.transformer_blocks.0.mlp.");
        Reference.AssertClose(log, "l0 mlp", mlp.forward(Reference.Cases["a_l0_mlp_in"]), Reference.Cases["a_l0_mlp_out"], Tolerances.Piece);
    }

    [RequiresCheckpoint]
    public void Embeddings_MatchTheReference()
    {
        using var _ = no_grad();
        var ck = Ckpt();
        var c = Reference.Cases;

        var timeEmbed = new TimestepEmbedding(512);
        Reference.LoadInto(timeEmbed, ck, "transformer.time_embed.");
        var time = full(new long[] { 1, 10 }, 0.3f);
        var timeEmb = timeEmbed.forward(time);
        Reference.AssertClose(log, "time_embed", timeEmb, c["a_time_emb"], Tolerances.Piece);

        var latentEmbed = nn.Sequential(nn.Linear(64, 512), nn.Linear(512, 512));
        Reference.LoadInto(latentEmbed, ck, "transformer.latent_embed.");
        var latentEmb = latentEmbed.forward(c["a_noisy"]);
        Reference.AssertClose(log, "latent_embed", latentEmb, c["a_latent_emb"], Tolerances.Piece);

        var inputEmbed = new InputEmbedding(512, 2048);
        Reference.LoadInto(inputEmbed, ck, "transformer.input_embed.");
        Reference.AssertClose(log, "input_embed", inputEmbed.forward(latentEmb, Reference.Inputs["style"], timeEmb), c["a_input_emb"], Tolerances.Piece);

        var normOut = new AdaLayerNormZeroFinal(2048, 512);
        Reference.LoadInto(normOut, ck, "transformer.norm_out.");
        Reference.AssertClose(log, "norm_out", normOut.forward(c["a_l15_out"], timeEmb), c["a_norm_out_out"], Tolerances.Piece);
    }

    [Fact]
    public void TimestepEmbedding_KeepsTheTimeDtype()
    {
        using var _ = no_grad();
        var e = new TimestepEmbedding(512);
        Assert.Equal(ScalarType.Float32, e.forward(zeros(1, 3)).dtype);
        Assert.Equal(new long[] { 1, 3, 512 }, e.forward(zeros(1, 3)).shape);
    }

    [RequiresCheckpoint]
    public void Layer0_NoCache_MatchesTheReference()
    {
        using var _ = no_grad();
        var layer = new DitLayer(2048, 16, 4, index: 0, Rope());
        Reference.LoadInto(layer, Ckpt(), "transformer.transformer_blocks.0.");
        Reference.AssertClose(log, "l0 no cache", layer.Forward(Reference.Cases["a_input_emb"], null, null, CacheUse.Transient), Reference.Cases["a_l0_out"], Tolerances.Piece);
    }

    [RequiresCheckpoint]
    public void Layer0_WithCache_MatchesTheReference()
    {
        using var _ = no_grad();
        var ck = Ckpt();
        var c = Reference.Cases;
        var tokens = Reference.Inputs["tokens"].unsqueeze(0);
        var style = Reference.Inputs["style"];
        long L = tokens.shape[1];

        var textEmbed = new TextEmbedding(1000, 512);
        Reference.LoadInto(textEmbed, ck, "transformer.text_embed.");
        var timeEmbed = new TimestepEmbedding(512);
        Reference.LoadInto(timeEmbed, ck, "transformer.time_embed.");
        var latentEmbed = nn.Sequential(nn.Linear(64, 512), nn.Linear(512, 512));
        Reference.LoadInto(latentEmbed, ck, "transformer.latent_embed.");
        var inputEmbed = new InputEmbedding(512, 2048);
        Reference.LoadInto(inputEmbed, ck, "transformer.input_embed.");
        var layer = new DitLayer(2048, 16, 4, index: 0, Rope());
        Reference.LoadInto(layer, ck, "transformer.transformer_blocks.0.");

        using var cache = new BlockCache(layers: 1);

        // text prefill at time -1, positions 0..L-1
        var textX = inputEmbed.forward(textEmbed.forward(tokens), style, timeEmbed.forward(full(new long[] { 1, L }, -1f)));
        layer.Forward(textX, arange(L).unsqueeze(0), cache, CacheUse.TextPrefill);
        Reference.AssertClose(log, "text keys l0", cache.TextKeys(0)!, c["b_text_k0"], Tolerances.Piece);
        Reference.AssertClose(log, "text values l0", cache.TextValues(0)!, c["b_text_v0"], Tolerances.Piece);

        // one clean block at time 1, positions 0..9
        var cleanX = inputEmbed.forward(latentEmbed.forward(c["b_clean_block"]), style, timeEmbed.forward(ones(1, 10)));
        layer.Forward(cleanX, arange(10).unsqueeze(0), cache, CacheUse.CleanWrite);
        Reference.AssertClose(log, "clean keys l0", cache.CleanKeys(0)!, c["b_clean_k0"], Tolerances.Piece);
        Reference.AssertClose(log, "clean values l0", cache.CleanValues(0)!, c["b_clean_v0"], Tolerances.Piece);

        // noisy block at t=0.5, positions 10..19, read-only
        var noisyX = inputEmbed.forward(latentEmbed.forward(c["b_noisy"]), style, timeEmbed.forward(full(new long[] { 1, 10 }, 0.5f)));
        var before = cache.CleanLength(0);
        var y = layer.Forward(noisyX, arange(10, 20).unsqueeze(0), cache, CacheUse.Transient);
        Assert.Equal(before, cache.CleanLength(0));
        Reference.AssertClose(log, "l0 with cache", y, c["b_l0_out"], Tolerances.Piece);
    }
}
