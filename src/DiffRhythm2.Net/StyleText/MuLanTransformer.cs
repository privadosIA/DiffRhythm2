using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.StyleText;

/// <summary>Sizes of MuLan's text transformer.</summary>
public sealed record MuLanTransformerConfig(int Dim = 768, int Depth = 8, int Heads = 8, int DimHead = 64, int FfMult = 4)
{
    public int Inner => Heads * DimHead;

    /// <summary><c>int(dim * mult * 2 / 3)</c>: the width of each half of the GEGLU.</summary>
    public int FfHidden => (int)(Dim * FfMult * 2 / 3.0);
}

/// <summary>MuLan's biasless layer normalization: only the learned scale (<c>learned_gamma</c>) exists in the checkpoint; the offset is zero.</summary>
public sealed class MuLanLayerNorm : Module<Tensor, Tensor>
{
    private readonly Parameter learned_gamma;
    private readonly long dim;

    public MuLanLayerNorm(int dim) : base(nameof(MuLanLayerNorm))
    {
        this.dim = dim;
        learned_gamma = new Parameter(ones(dim), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => F.layer_norm(x, new long[] { dim }, learned_gamma, null);
}

/// <summary>
/// MuLan's attention: pre-normalization, queries and keys L2-normalized per head and multiplied by learned per-dimension scales, similarities
/// multiplied by the constant 8 (not by 1 / sqrt(d)), softmax, output projection. Field names are the checkpoint's.
/// </summary>
public sealed class MuLanAttention : Module<Tensor, Tensor>
{
    private const double SimilarityScale = 8.0;

    private readonly MuLanLayerNorm norm;
    private readonly Linear to_q;
    private readonly Linear to_kv;
    private readonly Parameter q_scale;
    private readonly Parameter k_scale;
    private readonly Sequential to_out;
    private readonly int heads;
    private readonly int dimHead;

    public MuLanAttention(MuLanTransformerConfig c) : base(nameof(MuLanAttention))
    {
        heads = c.Heads;
        dimHead = c.DimHead;
        norm = new MuLanLayerNorm(c.Dim);
        to_q = Linear(c.Dim, c.Inner, hasBias: false);
        to_kv = Linear(c.Dim, c.Inner * 2, hasBias: false);
        q_scale = new Parameter(ones(c.DimHead), requires_grad: false);
        k_scale = new Parameter(ones(c.DimHead), requires_grad: false);
        to_out = Sequential(Linear(c.Inner, c.Dim, hasBias: false));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], n = x.shape[1];
        var h = norm.forward(x);
        var q = to_q.forward(h);
        var kv = to_kv.forward(h).chunk(2, -1);
        Tensor Split(Tensor t) => t.view(b, n, heads, dimHead).transpose(1, 2);          // 'b n (h d) -> b h n d'
        var qh = F.normalize(Split(q), 2.0, -1) * q_scale;
        var kh = F.normalize(Split(kv[0]), 2.0, -1) * k_scale;
        var vh = Split(kv[1]);
        var sim = matmul(qh, kh.transpose(-1, -2)) * SimilarityScale;
        var attn = sim.softmax(-1);
        var merged = matmul(attn, vh).transpose(1, 2).reshape(b, n, heads * dimHead);
        return to_out.forward(merged).MoveToOuterDisposeScope();
    }
}

/// <summary><c>gelu(gate) * x</c> over the two halves of the last dimension.</summary>
public sealed class Geglu : Module<Tensor, Tensor>
{
    public Geglu() : base(nameof(Geglu)) { }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var parts = x.chunk(2, -1);
        return (F.gelu(parts[1]) * parts[0]).MoveToOuterDisposeScope();
    }
}

/// <summary>MuLan's transformer: <c>layers.N.0</c> is the attention and <c>layers.N.1</c> the feed-forward, each added to its input.</summary>
public sealed class MuLanTransformer : Module<Tensor, Tensor>
{
    private readonly ModuleList<ModuleList<Module<Tensor, Tensor>>> layers;

    public MuLanTransformerConfig Config { get; }

    public MuLanTransformer(MuLanTransformerConfig config) : base(nameof(MuLanTransformer))
    {
        Config = config;
        layers = ModuleList(Enumerable.Range(0, config.Depth).Select(_ => ModuleList<Module<Tensor, Tensor>>(new MuLanAttention(config), FeedForward(config))).ToArray());
        RegisterComponents();
    }

    /// <summary>
    /// The feed-forward block as the checkpoint indexes it: 0 pre-norm, 1 up-projection to twice the hidden width, 2 GEGLU, 3 dropout (nothing at
    /// inference), 4 down-projection. Without biases.
    /// </summary>
    private static Sequential FeedForward(MuLanTransformerConfig c) =>
        Sequential(new MuLanLayerNorm(c.Dim), Linear(c.Dim, c.FfHidden * 2, hasBias: false), new Geglu(), Identity(), Linear(c.FfHidden, c.Dim, hasBias: false));

    /// <summary>Layer <paramref name="index"/> applied to a hidden state (stage fixture).</summary>
    public Tensor Layer(int index, Tensor x)
    {
        using var scope = NewDisposeScope();
        var pair = layers[index];
        var a = pair[0].forward(x) + x;
        return (pair[1].forward(a) + a).MoveToOuterDisposeScope();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        for (int i = 0; i < layers.Count; i++) x = Layer(i, x);
        return x.MoveToOuterDisposeScope();
    }
}
