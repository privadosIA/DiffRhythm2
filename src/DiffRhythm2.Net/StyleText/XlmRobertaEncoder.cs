using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.StyleText;

/// <summary>Sizes of the XLM-RoBERTa-base encoder.</summary>
public sealed record XlmRobertaConfig(int Vocabulary = 250002, int Hidden = 768, int Layers = 12, int Heads = 12, int Intermediate = 3072, int MaxPositions = 514, int PadId = 1, double LayerNormEps = 1e-5)
{
    /// <summary>Position ids start after the padding index, so the longest sequence is the number of position rows minus two.</summary>
    public int MaxTokens => MaxPositions - PadId - 1;
}

/// <summary>Word, position and token-type embeddings followed by a layer normalization. Field names are the HuggingFace ones.</summary>
public sealed class XlmRobertaEmbeddings : Module<Tensor, Tensor>
{
    private readonly Embedding word_embeddings;
    private readonly Embedding position_embeddings;
    private readonly Embedding token_type_embeddings;
    private readonly LayerNorm LayerNorm;
    private readonly int padId;

    public XlmRobertaEmbeddings(XlmRobertaConfig c) : base(nameof(XlmRobertaEmbeddings))
    {
        padId = c.PadId;
        word_embeddings = Embedding(c.Vocabulary, c.Hidden);
        position_embeddings = Embedding(c.MaxPositions, c.Hidden);
        token_type_embeddings = Embedding(1, c.Hidden);
        LayerNorm = nn.LayerNorm(c.Hidden, c.LayerNormEps);
        RegisterComponents();
    }

    /// <summary>
    /// HuggingFace's <c>create_position_ids_from_input_ids</c>: the position of a token is its rank among the non-padding tokens plus the padding
    /// index, and padding keeps the padding index. For a text without padding the positions are 2 to n + 1.
    /// </summary>
    public Tensor PositionIds(Tensor ids)
    {
        using var scope = NewDisposeScope();
        var mask = ids.ne(padId).to(ScalarType.Int32);
        var incremental = mask.cumsum(1).to(ScalarType.Int32) * mask;
        return (incremental.to(ScalarType.Int64) + padId).MoveToOuterDisposeScope();
    }

    /// <param name="ids"><c>[b, n]</c> token ids.</param>
    public override Tensor forward(Tensor ids)
    {
        using var scope = NewDisposeScope();
        using var positions = PositionIds(ids);
        using var types = zeros_like(ids);
        var x = word_embeddings.forward(ids) + token_type_embeddings.forward(types) + position_embeddings.forward(positions);
        return LayerNorm.forward(x).MoveToOuterDisposeScope();
    }
}

/// <summary>Query, key and value projections of one layer and the attention itself (<c>attention.self</c> in HuggingFace).</summary>
public sealed class XlmRobertaSelfAttention : Module<Tensor, Tensor>
{
    private readonly Linear query;
    private readonly Linear key;
    private readonly Linear value;
    private readonly int heads;

    public XlmRobertaSelfAttention(XlmRobertaConfig c) : base(nameof(XlmRobertaSelfAttention))
    {
        heads = c.Heads;
        query = Linear(c.Hidden, c.Hidden);
        key = Linear(c.Hidden, c.Hidden);
        value = Linear(c.Hidden, c.Hidden);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], n = x.shape[1], d = x.shape[2] / heads;
        Tensor Split(Tensor t) => t.view(b, n, heads, d).transpose(1, 2);
        var attn = F.scaled_dot_product_attention(Split(query.forward(x)), Split(key.forward(x)), Split(value.forward(x)));   // bidirectional, no mask: one text, no padding
        return attn.transpose(1, 2).reshape(b, n, heads * d).MoveToOuterDisposeScope();
    }
}

/// <summary>Dense projection, residual and layer normalization (<c>attention.output</c> and <c>output</c> in HuggingFace).</summary>
public sealed class XlmRobertaOutput : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear dense;
    private readonly LayerNorm LayerNorm;

    public XlmRobertaOutput(int inFeatures, int hidden, double eps) : base(nameof(XlmRobertaOutput))
    {
        dense = Linear(inFeatures, hidden);
        LayerNorm = nn.LayerNorm(hidden, eps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor residual)
    {
        using var scope = NewDisposeScope();
        return LayerNorm.forward(dense.forward(x) + residual).MoveToOuterDisposeScope();
    }
}

public sealed class XlmRobertaAttention : Module<Tensor, Tensor>
{
    private readonly XlmRobertaSelfAttention self;
    private readonly XlmRobertaOutput output;

    public XlmRobertaAttention(XlmRobertaConfig c) : base(nameof(XlmRobertaAttention))
    {
        self = new XlmRobertaSelfAttention(c);
        output = new XlmRobertaOutput(c.Hidden, c.Hidden, c.LayerNormEps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return output.forward(self.forward(x), x).MoveToOuterDisposeScope();
    }
}

public sealed class XlmRobertaIntermediate : Module<Tensor, Tensor>
{
    private readonly Linear dense;

    public XlmRobertaIntermediate(XlmRobertaConfig c) : base(nameof(XlmRobertaIntermediate))
    {
        dense = Linear(c.Hidden, c.Intermediate);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => F.gelu(dense.forward(x));   // the exact (erf) GELU
}

/// <summary>One post-norm encoder layer.</summary>
public sealed class XlmRobertaLayer : Module<Tensor, Tensor>
{
    private readonly XlmRobertaAttention attention;
    private readonly XlmRobertaIntermediate intermediate;
    private readonly XlmRobertaOutput output;

    public XlmRobertaLayer(XlmRobertaConfig c) : base(nameof(XlmRobertaLayer))
    {
        attention = new XlmRobertaAttention(c);
        intermediate = new XlmRobertaIntermediate(c);
        output = new XlmRobertaOutput(c.Intermediate, c.Hidden, c.LayerNormEps);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        using var a = attention.forward(x);
        using var i = intermediate.forward(a);
        return output.forward(i, a).MoveToOuterDisposeScope();
    }
}

public sealed class XlmRobertaStack : Module<Tensor, Tensor>
{
    private readonly ModuleList<XlmRobertaLayer> layer;

    public XlmRobertaStack(XlmRobertaConfig c) : base(nameof(XlmRobertaStack))
    {
        layer = ModuleList(Enumerable.Range(0, c.Layers).Select(_ => new XlmRobertaLayer(c)).ToArray());
        RegisterComponents();
    }

    public Tensor Layer(int index, Tensor x) => layer[index].forward(x);

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        for (int i = 0; i < layer.Count; i++) x = layer[i].forward(x);
        return x.MoveToOuterDisposeScope();
    }
}

/// <summary>
/// HuggingFace's <c>XLMRobertaModel</c> up to <c>last_hidden_state</c> (the pooler is never evaluated, so it is not built). Field names are the
/// checkpoint's: <c>embeddings</c> and <c>encoder.layer.N</c>.
/// </summary>
public sealed class XlmRobertaModel : Module<Tensor, Tensor>
{
    private readonly XlmRobertaEmbeddings embeddings;
    private readonly XlmRobertaStack encoder;

    public XlmRobertaConfig Config { get; }

    public XlmRobertaModel(XlmRobertaConfig config) : base(nameof(XlmRobertaModel))
    {
        Config = config;
        embeddings = new XlmRobertaEmbeddings(config);
        encoder = new XlmRobertaStack(config);
        RegisterComponents();
    }

    /// <summary>The embedding block alone (stage fixture).</summary>
    public Tensor Embed(Tensor ids) => embeddings.forward(ids);

    /// <summary>Layer <paramref name="index"/> applied to a hidden state (stage fixture).</summary>
    public Tensor Layer(int index, Tensor hidden) => encoder.Layer(index, hidden);

    /// <param name="ids"><c>[b, n]</c> token ids, no padding.</param>
    /// <returns>The last hidden state <c>[b, n, 768]</c>.</returns>
    public override Tensor forward(Tensor ids)
    {
        using var scope = NewDisposeScope();
        if (ids.shape[1] > Config.MaxTokens) throw new ArgumentException($"the text has {ids.shape[1]} positions but the encoder supports at most {Config.MaxTokens}");
        using var hidden = embeddings.forward(ids);
        return encoder.forward(hidden).MoveToOuterDisposeScope();
    }
}
