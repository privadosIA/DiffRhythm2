using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.Model;

/// <summary>HuggingFace's <c>LlamaMLP</c> (SwiGLU): <c>down_proj(silu(gate_proj(x)) * up_proj(x))</c>, bias-free. The field names are the checkpoint's.</summary>
public sealed class LlamaMlp : Module<Tensor, Tensor>
{
    private readonly Linear gate_proj;
    private readonly Linear up_proj;
    private readonly Linear down_proj;

    public LlamaMlp(int dim, int hiddenSize) : base(nameof(LlamaMlp))
    {
        gate_proj = Linear(dim, hiddenSize, hasBias: false);
        up_proj = Linear(dim, hiddenSize, hasBias: false);
        down_proj = Linear(hiddenSize, dim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return down_proj.forward(F.silu(gate_proj.forward(x)) * up_proj.forward(x)).MoveToOuterDisposeScope();
    }
}
