using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DiffRhythm2.Model;

/// <summary>
/// HuggingFace's <c>LlamaRMSNorm</c>: parameter <c>weight</c>, variance computed in float32, and <c>weight * (normalized.to(input dtype))</c>,
/// i.e. the multiplication by the weight happens in the input dtype. Not interchangeable with the core's <c>RmsNorm</c> (parameter <c>scale</c>,
/// the weight applied before the cast back) nor with <c>FlowRmsNorm</c> (no float32 round trip).
/// </summary>
public sealed class LlamaRmsNorm : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly double eps;

    public LlamaRmsNorm(int dim, double eps = 1e-6) : base(nameof(LlamaRmsNorm))
    {
        this.eps = eps;
        weight = new Parameter(ones(dim), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var xf = x.to(ScalarType.Float32);
        var variance = xf.pow(2).mean(new long[] { -1 }, keepdim: true);
        var normed = (xf * rsqrt(variance + eps)).to(x.dtype);
        return (weight * normed).MoveToOuterDisposeScope();
    }
}
