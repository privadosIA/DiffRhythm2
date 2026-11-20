using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace DiffRhythm2.Decoder;

/// <summary>
/// SnakeBeta: <c>x + 1/(β + 1e-9) · sin²(α·x)</c> with a trainable α (frequency) and β (magnitude) per channel. With the log-scale
/// parametrization of the published checkpoint the stored values are logarithms, <c>α = exp(alpha)</c>, <c>β = exp(beta)</c>. Input and output
/// <c>[b, c, t]</c>; the arithmetic is done in the input dtype, like the reference.
/// </summary>
public sealed class SnakeBeta : nn.Module<Tensor, Tensor>
{
    private const double NoDivByZero = 1e-9;

    private readonly Parameter alpha;
    private readonly Parameter beta;
    private readonly bool logscale;

    public SnakeBeta(int channels, bool logscale) : base(nameof(SnakeBeta))
    {
        this.logscale = logscale;
        // The reference initializes to zeros (log scale) or ones (linear); the checkpoint overwrites them.
        alpha = new Parameter(logscale ? zeros(channels) : ones(channels), requires_grad: false);
        beta = new Parameter(logscale ? zeros(channels) : ones(channels), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var a = alpha.unsqueeze(0).unsqueeze(-1);   // [1, c, 1]
        var b = beta.unsqueeze(0).unsqueeze(-1);
        if (logscale) { a = a.exp(); b = b.exp(); }
        var y = x + (b + NoDivByZero).reciprocal() * x.mul(a).sin().pow(2);
        return y.MoveToOuterDisposeScope();
    }
}
