using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.StyleText;

/// <summary>MuLan's <c>text</c> module: the XLM-RoBERTa encoder (<c>model</c>) followed by MuLan's own transformer (<c>transformer</c>).</summary>
public sealed class MuLanTextEncoder : Module<Tensor, Tensor>
{
    private readonly XlmRobertaModel model;
    private readonly MuLanTransformer transformer;

    public XlmRobertaModel Encoder => model;
    public MuLanTransformer Transformer => transformer;

    public MuLanTextEncoder(XlmRobertaConfig encoder, MuLanTransformerConfig transformerConfig) : base(nameof(MuLanTextEncoder))
    {
        model = new XlmRobertaModel(encoder);
        transformer = new MuLanTransformer(transformerConfig);
        RegisterComponents();
    }

    /// <param name="ids"><c>[1, n]</c>.</param>
    /// <returns>The transformer output <c>[1, n, 768]</c>.</returns>
    public override Tensor forward(Tensor ids)
    {
        using var scope = NewDisposeScope();
        using var hidden = model.forward(ids);      // the projection between them is the identity for this checkpoint
        return transformer.forward(hidden).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// The text tower of MuQ-MuLan: tokens through the encoder and the transformer, the mean over all positions (the beginning and end markers
/// included), a linear projection to 512 dimensions and L2 normalization. Field names are the checkpoint's keys minus the leading
/// <c>mulan.</c>: <c>text.model...</c>, <c>text.transformer...</c> and <c>text_to_latents</c>.
/// </summary>
public sealed class MuLanTextTower : Module<Tensor, Tensor>
{
    public const int LatentDim = 512;

    private readonly MuLanTextEncoder text;
    private readonly Linear text_to_latents;

    public MuLanTextEncoder Text => text;

    public MuLanTextTower(XlmRobertaConfig? encoder = null, MuLanTransformerConfig? transformer = null, int latentDim = LatentDim) : base(nameof(MuLanTextTower))
    {
        var transformerConfig = transformer ?? new MuLanTransformerConfig();
        text = new MuLanTextEncoder(encoder ?? new XlmRobertaConfig(), transformerConfig);
        text_to_latents = Linear(transformerConfig.Dim, latentDim);
        RegisterComponents();
        foreach (var p in parameters()) p.requires_grad_(false);
    }

    /// <summary>The mean over positions of the transformer output, <c>[1, 768]</c> (stage fixture).</summary>
    public static Tensor Mean(Tensor transformerOutput) => transformerOutput.mean(new long[] { -2 });

    /// <summary>Projection and L2 normalization of a mean, <c>[1, 512]</c>.</summary>
    public Tensor Project(Tensor mean)
    {
        using var scope = NewDisposeScope();
        return F.normalize(text_to_latents.forward(mean), 2.0, -1).MoveToOuterDisposeScope();
    }

    /// <param name="ids"><c>[1, n]</c> token ids of one text.</param>
    /// <returns>The style embedding <c>[1, 512]</c>, unit length.</returns>
    public override Tensor forward(Tensor ids)
    {
        using var scope = NewDisposeScope();
        if (ids.dim() != 2 || ids.shape[0] != 1) throw new NotSupportedException($"only one text per call is supported (got ids of shape [{string.Join(", ", ids.shape)}]); the reference averages over padded positions, so a batch would give different results");
        using var output = text.forward(ids);
        using var mean = Mean(output);
        return Project(mean).MoveToOuterDisposeScope();
    }
}
