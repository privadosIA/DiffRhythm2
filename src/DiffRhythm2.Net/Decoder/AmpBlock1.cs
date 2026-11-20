using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DiffRhythm2.Decoder;

/// <summary>
/// BigVGAN's AMPBlock1: three pairs of convolutions, the first of each pair dilated (<c>convs1</c>) and the second not (<c>convs2</c>), each
/// preceded by an anti-aliased SnakeBeta (<c>activations</c>, interleaved: 0,2,4 before convs1 and 1,3,5 before convs2), with a residual
/// connection per pair. Field names are the checkpoint's.
/// </summary>
public sealed class AmpBlock1 : Module<Tensor, Tensor>
{
    private readonly ModuleList<Conv1dLayer> convs1;
    private readonly ModuleList<Conv1dLayer> convs2;
    private readonly ModuleList<Activation1d> activations;

    public AmpBlock1(int channels, int kernelSize, int[] dilations, bool logscale) : base(nameof(AmpBlock1))
    {
        // Conv1dLayer's symmetric padding (k·d − d)/2 is BigVGAN's get_padding.
        convs1 = ModuleList(dilations.Select(d => new Conv1dLayer(channels, channels, kernelSize, d)).ToArray());
        convs2 = ModuleList(dilations.Select(_ => new Conv1dLayer(channels, channels, kernelSize, 1)).ToArray());
        activations = ModuleList(Enumerable.Range(0, 2 * dilations.Length).Select(_ => new Activation1d(channels, logscale)).ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        for (int i = 0; i < convs1.Count; i++)
        {
            var xt = activations[2 * i].forward(x);
            xt = convs1[i].forward(xt);
            xt = activations[2 * i + 1].forward(xt);
            xt = convs2[i].forward(xt);
            x = xt + x;
        }
        return x.MoveToOuterDisposeScope();
    }
}
