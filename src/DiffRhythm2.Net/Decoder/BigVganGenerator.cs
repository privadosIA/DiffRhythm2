using MusicGen.Core.Layers;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.Decoder;

/// <summary>A 1D convolution whose bias is optional (the final convolution of the published decoder has none). Registers <c>weight</c> (and <c>bias</c>) directly.</summary>
public sealed class PostConv1d : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter? bias;
    private readonly int padding;

    public PostConv1d(int inChannels, int outChannels, int kernel, bool useBias) : base(nameof(PostConv1d))
    {
        padding = (kernel - 1) / 2;
        weight = new Parameter(empty(outChannels, inChannels, kernel), requires_grad: false);
        if (useBias) bias = new Parameter(zeros(outChannels), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => F.conv1d(x, weight, bias, stride: 1, padding: padding);
}

/// <summary>
/// The BigVGAN generator of DiffRhythm2: latent <c>[b, 64, t]</c> to a mono waveform <c>[b, 1, t·9600]</c> in [-1, 1].
/// <c>conv_pre</c>, six transposed-convolution upsamplers (<c>ups.i.0</c>) each followed by the average of three
/// <see cref="AmpBlock1"/> residual blocks (<c>resblocks</c>), a final anti-aliased SnakeBeta, <c>conv_post</c> and a hard clamp
/// (or a tanh, when the configuration asks for it). Field names are the checkpoint's keys.
/// </summary>
public sealed class BigVganGenerator : Module<Tensor, Tensor>
{
    private readonly Conv1dLayer conv_pre;
    private readonly ModuleList<ModuleList<ConvTranspose1dLayer>> ups;
    private readonly ModuleList<AmpBlock1> resblocks;
    private readonly Activation1d activation_post;
    private readonly PostConv1d conv_post;
    private readonly int numKernels;

    public BigVganConfig Config { get; }

    public BigVganGenerator(BigVganConfig config) : base(nameof(BigVganGenerator))
    {
        Config = config;
        numKernels = config.ResblockKernelSizes.Length;
        conv_pre = new Conv1dLayer(config.InChannels, config.UpsampleInitialChannel, 7);

        var upLayers = new List<ModuleList<ConvTranspose1dLayer>>();
        var blocks = new List<AmpBlock1>();
        int ch = config.UpsampleInitialChannel;
        for (int i = 0; i < config.UpsampleRates.Length; i++)
        {
            int u = config.UpsampleRates[i], k = config.UpsampleKernelSizes[i];
            upLayers.Add(ModuleList(new ConvTranspose1dLayer(ch, ch / 2, k, u, causal: false)));
            ch /= 2;
            for (int j = 0; j < numKernels; j++)
                blocks.Add(new AmpBlock1(ch, config.ResblockKernelSizes[j], config.ResblockDilationSizes[j], config.SnakeLogscale));
        }
        ups = ModuleList(upLayers.ToArray());
        resblocks = ModuleList(blocks.ToArray());
        activation_post = new Activation1d(ch, config.SnakeLogscale);
        conv_post = new PostConv1d(ch, 1, 7, config.UseBiasAtFinal);
        RegisterComponents();
        foreach (var p in parameters()) p.requires_grad_(false);
    }

    public override Tensor forward(Tensor x) => Forward(x, null);

    /// <summary>
    /// The forward pass. <paramref name="observe"/> sees the signal after <c>conv_pre</c>, after each transposed convolution (<c>up0</c>…), after each
    /// stage's averaged residual blocks (<c>stage0</c>…), after the final activation (<c>act_post</c>) and after <c>conv_post</c> before the output
    /// bound (<c>conv_post</c>); the tensor is only valid during the callback. Only one stage's activations are alive at a time.
    /// </summary>
    public Tensor Forward(Tensor x, Action<string, Tensor>? observe)
    {
        using var outer = NewDisposeScope();
        var h = conv_pre.forward(x);
        observe?.Invoke("conv_pre", h);
        for (int i = 0; i < ups.Count; i++)
        {
            using var stage = NewDisposeScope();
            var up = ups[i][0].forward(h);
            observe?.Invoke($"up{i}", up);
            Tensor? xs = null;
            for (int j = 0; j < numKernels; j++)
            {
                var r = resblocks[i * numKernels + j].forward(up);
                xs = xs is null ? r : xs + r;       // r0, then (r0 + r1) + r2, as the reference accumulates
            }
            h = (xs! / numKernels).MoveToOuterDisposeScope();
            observe?.Invoke($"stage{i}", h);
        }
        h = activation_post.forward(h);
        observe?.Invoke("act_post", h);
        h = conv_post.forward(h);
        observe?.Invoke("conv_post", h);
        h = Config.UseTanhAtFinal ? h.tanh() : h.clamp(-1.0, 1.0);
        return h.MoveToOuterDisposeScope();
    }
}
