using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DiffRhythm2.AudioTower;

/// <summary>Sizes of MuQ-large as MuQ-MuLan uses it (the published <c>MuQ-large-msd-iter</c> configuration with 12 Conformer layers).</summary>
public sealed record MuQConfig(int Bands = 128, int ConvDim = 512, int Hidden = 1024, int Layers = 12, int Heads = 16, int Intermediate = 4096, int ConvKernel = 31, double RotaryBase = 10000.0)
{
    /// <summary>Width of one attention head.</summary>
    public int HeadSize => Hidden / Heads;

    /// <summary>Time steps the subsampling keeps for each frame of the spectrogram: 4 frames become 1 step.</summary>
    public const int Subsampling = 4;
}

/// <summary>
/// A residual block of the subsampling: two 3x3 convolutions with batch normalization and ReLU, and, because the block always changes the shape (stride 2),
/// a 3x3 convolution with batch normalization on the shortcut. Field names are the checkpoint's (<c>conv1</c>, <c>bn1</c>, <c>conv2</c>, <c>bn2</c>, <c>conv3</c>, <c>bn3</c>).
/// </summary>
public sealed class Res2dBlock : Module<Tensor, Tensor>
{
    private readonly Conv2d conv1;
    private readonly BatchNorm2d bn1;
    private readonly Conv2d conv2;
    private readonly BatchNorm2d bn2;
    private readonly Conv2d conv3;
    private readonly BatchNorm2d bn3;

    /// <summary>Creates a block from <paramref name="inChannels"/> to <paramref name="outChannels"/> channels that halves both axes.</summary>
    public Res2dBlock(int inChannels, int outChannels) : base(nameof(Res2dBlock))
    {
        conv1 = Conv2d(inChannels, outChannels, 3, stride: 2, padding: 1);
        bn1 = BatchNorm2d(outChannels);
        conv2 = Conv2d(outChannels, outChannels, 3, padding: 1);
        bn2 = BatchNorm2d(outChannels);
        conv3 = Conv2d(inChannels, outChannels, 3, stride: 2, padding: 1);
        bn3 = BatchNorm2d(outChannels);
        RegisterComponents();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = bn2.forward(conv2.forward(functional.relu(bn1.forward(conv1.forward(x)))));
        var shortcut = bn3.forward(conv3.forward(x));
        return functional.relu(shortcut + h).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// MuQ's convolutional subsampling: two <see cref="Res2dBlock"/>s that take the normalized spectrogram <c>[B, 128, T]</c> to <c>[B, 512, 32, T / 4]</c>, channels and
/// frequency flattened together and projected linearly to 1024. Field names are the checkpoint's (<c>conv.0</c>, <c>conv.1</c>, <c>linear</c>).
/// </summary>
public sealed class MuQSubsampling : Module<Tensor, Tensor>
{
    private readonly Sequential conv;
    private readonly Linear linear;
    private readonly MuQConfig config;

    /// <summary>Creates the subsampling for a configuration.</summary>
    public MuQSubsampling(MuQConfig config) : base(nameof(MuQSubsampling))
    {
        this.config = config;
        conv = Sequential(new Res2dBlock(1, config.ConvDim), new Res2dBlock(config.ConvDim, config.ConvDim));
        linear = Linear((long)config.ConvDim * config.Bands / MuQConfig.Subsampling, config.Hidden);
        RegisterComponents();
    }

    /// <summary>The normalized spectrogram <c>[B, 128, T]</c> to the sequence <c>[B, ceil(ceil(T / 2) / 2), 1024]</c>.</summary>
    public override Tensor forward(Tensor spectrogram)
    {
        using var scope = NewDisposeScope();
        var x = conv.forward(spectrogram.unsqueeze(1));                             // [B, 512, 32, T']
        long b = x.shape[0], c = x.shape[1], f = x.shape[2], t = x.shape[3];
        var flat = x.permute(0, 3, 1, 2).reshape(b, t, c * f);                      // 'b c f t -> b t (c f)'
        return linear.forward(flat).MoveToOuterDisposeScope();
    }
}
