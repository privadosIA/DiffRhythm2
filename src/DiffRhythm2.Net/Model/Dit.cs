using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DiffRhythm2.Model;

/// <summary>
/// DiffRhythm2's diffusion transformer. Time and style enter once through <see cref="InputEmbedding"/> and again through the final
/// <see cref="AdaLayerNormZeroFinal"/>; the layers themselves are unconditioned Llama blocks. Field names mirror the Python module so that the
/// checkpoint keys (without their <c>transformer.</c> prefix) are the parameter names. The <c>repa</c> projector heads of the checkpoint
/// are training-only and have no counterpart here.
/// </summary>
public sealed class Dit : Module
{
    private readonly TimestepEmbedding time_embed;
    private readonly TextEmbedding text_embed;
    private readonly InputEmbedding input_embed;
    private readonly Sequential latent_embed;
    private readonly ModuleList<DitLayer> transformer_blocks;
    private readonly AdaLayerNormZeroFinal norm_out;
    private readonly Linear proj_out;

    public DitConfig Config { get; }

    public Dit(DitConfig config) : base(nameof(Dit))
    {
        Config = config;
        var rope = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(DitConfig.RopeTheta), RopeTable.Float32Cached,
            config.HeadDim, DitConfig.MaxPositions);
        time_embed = new TimestepEmbedding(DitConfig.CondDim, DitConfig.FreqEmbedDim);
        text_embed = new TextEmbedding(config.TextNumEmbeds, DitConfig.CondDim);
        input_embed = new InputEmbedding(DitConfig.CondDim, config.Dim);
        latent_embed = Sequential(Linear(config.MelDim, DitConfig.CondDim), Linear(DitConfig.CondDim, DitConfig.CondDim));
        transformer_blocks = ModuleList(Enumerable.Range(0, config.Depth).Select(i => new DitLayer(config.Dim, config.Heads, config.FfMult, i, rope)).ToArray());
        norm_out = new AdaLayerNormZeroFinal(config.Dim, DitConfig.CondDim);
        proj_out = Linear(config.Dim, config.MelDim);
        RegisterComponents();
        foreach (var p in parameters()) p.requires_grad_(false);
    }

    /// <summary>Lyric token ids <c>[b, n]</c> to <c>[b, n, 512]</c>.</summary>
    public Tensor EmbedText(Tensor tokens) => text_embed.forward(tokens);

    /// <summary>Latent frames <c>[b, n, mel_dim]</c> to <c>[b, n, 512]</c>.</summary>
    public Tensor EmbedLatent(Tensor latent) => latent_embed.forward(latent);

    /// <summary>
    /// Runs the layers over an embedded sequence and writes the cache as <paramref name="use"/> says, without computing the output head
    /// (the reference discards it for the text prefill and for the clean-block write).
    /// </summary>
    /// <param name="embedded"><c>[b, n, 512]</c> from <see cref="EmbedText"/> or <see cref="EmbedLatent"/>.</param>
    /// <param name="time"><c>[b, n]</c>: -1 for text, 1 for a clean block, in [0, 1] while integrating.</param>
    /// <param name="style"><c>[b, 512]</c>; zeros for the unconditional branch.</param>
    /// <param name="positions"><c>[b, n]</c> long; null means 0..n-1.</param>
    /// <param name="cache">The cache the keys and values of every layer are written to.</param>
    /// <param name="use">Which section of <paramref name="cache"/> is written (the text prefill or a clean block).</param>
    public void Prefill(Tensor embedded, Tensor time, Tensor style, Tensor? positions, BlockCache cache, CacheUse use)
    {
        using var scope = NewDisposeScope();
        Hidden(embedded, time, style, positions, cache, use, out _);
    }

    /// <summary>The model output <c>[b, n, mel_dim]</c> (the velocity prediction) for an embedded sequence.</summary>
    public Tensor Forward(Tensor embedded, Tensor time, Tensor style, Tensor? positions, BlockCache? cache, CacheUse use)
    {
        using var scope = NewDisposeScope();
        var x = Hidden(embedded, time, style, positions, cache, use, out var timeEmb);
        return proj_out.forward(norm_out.forward(x, timeEmb)).MoveToOuterDisposeScope();
    }

    private Tensor Hidden(Tensor embedded, Tensor time, Tensor style, Tensor? positions, BlockCache? cache, CacheUse use, out Tensor timeEmb)
    {
        timeEmb = time_embed.forward(time);
        var x = input_embed.forward(embedded, style, timeEmb);
        for (int i = 0; i < transformer_blocks.Count; i++)
            x = transformer_blocks[i].Forward(x, positions, cache, use);
        return x;
    }
}
