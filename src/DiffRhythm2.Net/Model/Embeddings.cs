using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.Model;

/// <summary>Sinusoidal embedding of a time value: <c>[b, n]</c> to <c>[b, n, dim]</c>, <c>cat(sin, cos)</c> of <c>1000·t·exp(-i·ln(10000)/(dim/2-1))</c>, computed in float32.</summary>
public sealed class SinusPositionEmbedding
{
    private readonly int dim;

    public SinusPositionEmbedding(int dim) => this.dim = dim;

    public Tensor Forward(Tensor x, double scale = 1000)
    {
        using var scope = NewDisposeScope();
        int half = dim / 2;
        double step = Math.Log(10000) / (half - 1);
        var freqs = exp(arange(half, dtype: ScalarType.Float32, device: x.device) * -step);
        var emb = scale * x.to(ScalarType.Float32).unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0);
        return cat(new[] { emb.sin(), emb.cos() }, dim: -1).MoveToOuterDisposeScope();
    }
}

/// <summary><c>TimestepEmbedding</c>: the sinusoidal embedding (256) cast to the time's dtype, then <c>time_mlp</c> = Linear, SiLU, Linear (keys <c>time_mlp.0</c>, <c>time_mlp.2</c>).</summary>
public sealed class TimestepEmbedding : Module<Tensor, Tensor>
{
    private readonly SinusPositionEmbedding sinus;
    private readonly Sequential time_mlp;

    public TimestepEmbedding(int dim, int freqEmbedDim = 256) : base(nameof(TimestepEmbedding))
    {
        sinus = new SinusPositionEmbedding(freqEmbedDim);
        time_mlp = Sequential(Linear(freqEmbedDim, dim), SiLU(), Linear(dim, dim));
        RegisterComponents();
    }

    /// <param name="timestep"><c>[b, n]</c>.</param>
    public override Tensor forward(Tensor timestep)
    {
        using var scope = NewDisposeScope();
        return time_mlp.forward(sinus.Forward(timestep).to(timestep.dtype)).MoveToOuterDisposeScope();
    }
}

/// <summary><c>TextEmbedding</c>: <c>text_embed.weight</c> of <c>[vocab + 1, dim]</c> (entry 0 is the filler token).</summary>
public sealed class TextEmbedding : Module<Tensor, Tensor>
{
    private readonly Embedding text_embed;

    public TextEmbedding(int textNumEmbeds, int dim) : base(nameof(TextEmbedding))
    {
        text_embed = Embedding(textNumEmbeds + 1, dim);
        RegisterComponents();
    }

    public override Tensor forward(Tensor tokens) => text_embed.forward(tokens);
}

/// <summary>
/// <c>InputEmbedding</c>: <c>x = x + style + time</c> (the style <c>[b, d]</c> repeated along the sequence), then <c>proj(x) + x_orig</c>
/// where <c>x_orig</c> is the input before the additions, then <c>proj_2</c> up to the model width.
/// </summary>
public sealed class InputEmbedding : Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly Linear proj;
    private readonly Linear proj_2;

    public InputEmbedding(int condDim, int outDim) : base(nameof(InputEmbedding))
    {
        proj = Linear(condDim, condDim);
        proj_2 = Linear(condDim, outDim);
        RegisterComponents();
    }

    /// <param name="x"><c>[b, n, condDim]</c>.</param>
    /// <param name="style"><c>[b, condDim]</c>.</param>
    /// <param name="timeEmb"><c>[b, n, condDim]</c>.</param>
    public override Tensor forward(Tensor x, Tensor style, Tensor timeEmb)
    {
        using var scope = NewDisposeScope();
        var styleRep = style.unsqueeze(1).expand(x.shape[0], x.shape[1], style.shape[1]);
        var summed = x + styleRep + timeEmb;
        return proj_2.forward(proj.forward(summed) + x).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// <c>AdaLayerNormZero_Final</c>: <c>(scale, shift) = chunk(linear(silu(emb)))</c>, result <c>layer_norm(x) · (1 + scale) + shift</c> with a
/// LayerNorm without affine parameters (eps 1e-6).
/// </summary>
public sealed class AdaLayerNormZeroFinal : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear linear;
    private readonly int dim;

    public AdaLayerNormZeroFinal(int dim, int condDim) : base(nameof(AdaLayerNormZeroFinal))
    {
        this.dim = dim;
        linear = Linear(condDim, dim * 2);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, Tensor emb)
    {
        using var scope = NewDisposeScope();
        var modulation = linear.forward(F.silu(emb)).chunk(2, dim: -1);
        var normed = F.layer_norm(x, new long[] { dim }, null, null, 1e-6);
        return (normed * (1 + modulation[0]) + modulation[1]).MoveToOuterDisposeScope();
    }
}
