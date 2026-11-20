using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace DiffRhythm2.AudioTower;

/// <summary>MuQ-large's feature path: the convolutional subsampling and the Conformer (the reference's <c>MuQModel</c>). Field names are the checkpoint's.</summary>
public sealed class MuQEncoder : Module<Tensor, Tensor>
{
    private readonly MuQSubsampling conv;
    private readonly ConformerEncoder conformer;

    /// <summary>The subsampling.</summary>
    public MuQSubsampling Subsampling => conv;

    /// <summary>The Conformer.</summary>
    public ConformerEncoder Conformer => conformer;

    /// <summary>Creates the encoder.</summary>
    public MuQEncoder(MuQConfig config) : base(nameof(MuQEncoder))
    {
        conv = new MuQSubsampling(config);
        conformer = new ConformerEncoder(config);
        RegisterComponents();
    }

    /// <summary>The normalized spectrogram <c>[B, 128, T]</c> to the last hidden state <c>[B, T / 4, 1024]</c>.</summary>
    public override Tensor forward(Tensor spectrogram)
    {
        using var scope = NewDisposeScope();
        return conformer.forward(conv.forward(spectrogram)).MoveToOuterDisposeScope();
    }
}

/// <summary>The package layer of the reference's <c>MuQ</c> around <see cref="MuQEncoder"/>; it exists so that tensor names match the checkpoint (<c>audio.model.model.*</c>).</summary>
public sealed class MuQShell : Module<Tensor, Tensor>
{
    private readonly MuQEncoder model;

    /// <summary>The encoder inside.</summary>
    public MuQEncoder Encoder => model;

    /// <summary>Creates the shell.</summary>
    public MuQShell(MuQConfig config) : base(nameof(MuQShell))
    {
        model = new MuQEncoder(config);
        RegisterComponents();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor spectrogram) => model.forward(spectrogram);
}

/// <summary>MuLan's audio branch: MuQ-large and the projection from 1024 to 768 dimensions (<c>audio.*</c> in the checkpoint). Its transformer has no layers in the published configuration.</summary>
public sealed class MuLanAudioBranch : Module<Tensor, Tensor>
{
    private readonly MuQShell model;
    private readonly Linear proj;

    /// <summary>MuQ-large.</summary>
    public MuQShell Muq => model;

    /// <summary>Creates the branch.</summary>
    public MuLanAudioBranch(MuQConfig config, int dim) : base(nameof(MuLanAudioBranch))
    {
        model = new MuQShell(config);
        proj = Linear(config.Hidden, dim);
        RegisterComponents();
    }

    /// <summary>The encoder's last hidden state through the projection: <c>[B, T / 4, 768]</c>.</summary>
    public override Tensor forward(Tensor spectrogram)
    {
        using var scope = NewDisposeScope();
        return proj.forward(model.forward(spectrogram)).MoveToOuterDisposeScope();
    }

    /// <summary>The projection alone, for the stage fixtures.</summary>
    public Tensor Project(Tensor hidden) => proj.forward(hidden);
}

/// <summary>
/// The audio tower of MuQ-MuLan: tensor names are those of the checkpoint after the leading <c>mulan.</c> (<c>audio.model.model.conv.*</c>, <c>audio.proj.*</c>,
/// <c>audio_to_latents.*</c>), so a strict loader needs no mapping. It runs in evaluation mode and holds no gradients.
/// </summary>
public sealed class MuLanAudioTower : Module<Tensor, Tensor>
{
    /// <summary>Width of MuLan's audio transformer, and so of the mean over time.</summary>
    public const int Dim = 768;

    /// <summary>Width of the shared embedding space.</summary>
    public const int LatentDim = 512;

    private readonly MuLanAudioBranch audio;
    private readonly Linear audio_to_latents;

    /// <summary>Sizes of MuQ-large.</summary>
    public MuQConfig Config { get; }

    /// <summary>The audio branch.</summary>
    public MuLanAudioBranch Audio => audio;

    /// <summary>Creates the tower, in evaluation mode, without gradients.</summary>
    public MuLanAudioTower(MuQConfig? config = null, int dim = Dim, int latentDim = LatentDim) : base(nameof(MuLanAudioTower))
    {
        Config = config ?? new MuQConfig();
        audio = new MuLanAudioBranch(Config, dim);
        audio_to_latents = Linear(dim, latentDim);
        RegisterComponents();
        foreach (var p in parameters()) p.requires_grad_(false);
        eval();
    }

    /// <summary>The encoder's last hidden state for a normalized spectrogram <c>[B, 128, T]</c>: <c>[B, T / 4, 1024]</c>.</summary>
    public Tensor Encode(Tensor spectrogram) => audio.Muq.forward(spectrogram);

    /// <summary>The spectrogram <c>[B, 128, T]</c> to the projected sequence <c>[B, T / 4, 768]</c> (the stage before the mean).</summary>
    public override Tensor forward(Tensor spectrogram) => audio.forward(spectrogram);

    /// <summary>The mean of a projected sequence <c>[B, T, 768]</c> over all its steps: <c>[B, 768]</c> (the reference's <c>outputs.mean(dim=-2)</c>).</summary>
    public static Tensor MeanOverTime(Tensor projected) => projected.mean(new long[] { -2 });

    /// <summary>The mean <c>[B, 768]</c> through the projection to the latent space, before normalization: <c>[B, 512]</c>.</summary>
    public Tensor ToLatent(Tensor mean) => audio_to_latents.forward(mean);

    /// <summary>
    /// The embedding of a clip: the spectrogram <c>[B, 128, T]</c> through MuQ-large, the projection, the mean over time, the projection to the latent
    /// space and L2 normalization. <c>[B, 512]</c>, float32, unit length.
    /// </summary>
    public Tensor Embed(Tensor spectrogram)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var latent = ToLatent(MeanOverTime(forward(spectrogram)));
        return functional.normalize(latent, 2.0, -1).MoveToOuterDisposeScope();
    }
}
