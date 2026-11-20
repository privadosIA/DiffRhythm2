using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.Decoder;

/// <summary>
/// 2x upsampling with a Kaiser-sinc filter, as the reference's <c>UpSample1d</c> (alias-free-torch): replicate padding of <c>k/ratio − 1</c> on
/// both sides, a depthwise transposed convolution with stride <c>ratio</c>, the result times <c>ratio</c>, then the first
/// <c>pad·stride + (k − stride)/2</c> and last <c>pad·stride + (k − stride + 1)/2</c> samples cropped. The filter is a buffer
/// (<c>filter</c>, shape <c>[1, 1, k]</c>) that the checkpoint provides.
/// </summary>
public sealed class UpSample1d : nn.Module<Tensor, Tensor>
{
    private readonly int ratio;
    private readonly int pad;
    private readonly int padLeft;
    private readonly int padRight;

    public UpSample1d(int ratio = 2, int kernelSize = 12) : base(nameof(UpSample1d))
    {
        this.ratio = ratio;
        pad = kernelSize / ratio - 1;
        padLeft = pad * ratio + (kernelSize - ratio) / 2;
        padRight = pad * ratio + (kernelSize - ratio + 1) / 2;
        register_buffer("filter", Filters.Tensor(KaiserSinc.Filter(0.5 / ratio, 0.6 / ratio, kernelSize)));
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var filter = get_buffer("filter")!;
        long c = x.shape[1];
        var padded = F.pad(x, new long[] { pad, pad }, PaddingModes.Replicate);
        var y = ratio * F.conv_transpose1d(padded, filter.to(x.dtype).expand(c, -1, -1), stride: ratio, groups: c);
        return y.narrow(2, padLeft, y.shape[2] - padLeft - padRight).contiguous().MoveToOuterDisposeScope();
    }
}

/// <summary>The low-pass of <see cref="DownSample1d"/>: replicate padding (<c>k/2 − 1</c> left for even k, <c>k/2</c> right), then a depthwise convolution with the stride.</summary>
public sealed class LowPassFilter1d : nn.Module<Tensor, Tensor>
{
    private readonly int stride;
    private readonly int padLeft;
    private readonly int padRight;

    public LowPassFilter1d(double cutoff, double halfWidth, int stride, int kernelSize) : base(nameof(LowPassFilter1d))
    {
        this.stride = stride;
        bool even = kernelSize % 2 == 0;
        padLeft = kernelSize / 2 - (even ? 1 : 0);
        padRight = kernelSize / 2;
        register_buffer("filter", Filters.Tensor(KaiserSinc.Filter(cutoff, halfWidth, kernelSize)));
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var filter = get_buffer("filter")!;
        long c = x.shape[1];
        var padded = F.pad(x, new long[] { padLeft, padRight }, PaddingModes.Replicate);
        return F.conv1d(padded, filter.to(x.dtype).expand(c, -1, -1), stride: stride, groups: c).MoveToOuterDisposeScope();
    }
}

/// <summary>Anti-aliased downsampling: the low-pass with cutoff <c>0.5/ratio</c> and stride <c>ratio</c>. The parameter path is <c>downsample.lowpass.filter</c>.</summary>
public sealed class DownSample1d : nn.Module<Tensor, Tensor>
{
    private readonly LowPassFilter1d lowpass;

    public DownSample1d(int ratio = 2, int kernelSize = 12) : base(nameof(DownSample1d))
    {
        lowpass = new LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernelSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => lowpass.forward(x);
}

/// <summary>
/// The anti-aliased activation: upsample 2x, SnakeBeta, downsample 2x (the nonlinearity is evaluated at twice the rate so that its harmonics do
/// not alias). Submodule names (<c>act</c>, <c>upsample</c>, <c>downsample</c>) are the checkpoint's.
/// </summary>
public sealed class Activation1d : nn.Module<Tensor, Tensor>
{
    private readonly SnakeBeta act;
    private readonly UpSample1d upsample;
    private readonly DownSample1d downsample;

    public Activation1d(int channels, bool logscale, int ratio = 2, int kernelSize = 12) : base(nameof(Activation1d))
    {
        act = new SnakeBeta(channels, logscale);
        upsample = new UpSample1d(ratio, kernelSize);
        downsample = new DownSample1d(ratio, kernelSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return downsample.forward(act.forward(upsample.forward(x))).MoveToOuterDisposeScope();
    }
}

internal static class Filters
{
    public static Tensor Tensor(double[] taps) => torch.tensor(taps.Select(t => (float)t).ToArray()).reshape(1, 1, taps.Length);
}
