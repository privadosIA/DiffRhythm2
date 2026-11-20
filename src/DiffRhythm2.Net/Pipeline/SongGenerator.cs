using System.Diagnostics;
using DiffRhythm2.Decoder;
using DiffRhythm2.Model;
using DiffRhythm2.Sampling;
using DiffRhythm2.StyleText;
using DiffRhythm2.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Pipeline;

/// <summary>Numeric precision of the network and the decoder (the style encoder always runs in float32, as in the reference service).</summary>
public enum SongPrecision
{
    /// <summary>32-bit floats: the precision the parity gates are defined for.</summary>
    Float32,
    /// <summary>16-bit floats: what the reference service runs on CUDA.</summary>
    Float16,
}

/// <summary>Where the pipeline runs and what it reads.</summary>
/// <param name="ModelDirectory">Directory with the six model files, see <see cref="ModelDirectory"/>.</param>
/// <param name="Device"><c>cpu</c>, <c>cuda</c> or <c>cuda:N</c>.</param>
/// <param name="Precision">Float32 on CPU and Float16 on CUDA when not given (the reference service runs in half precision on CUDA).</param>
/// <param name="CaptureIntermediates">Keep tokens, style, noise and latent in the result (for tests and diagnostics; off by default).</param>
public sealed record SongOptions(string ModelDirectory, string Device = "cpu", SongPrecision? Precision = null, bool CaptureIntermediates = false);

/// <summary>The stages of the pipeline, in order.</summary>
public enum SongStage
{
    /// <summary>Lyrics to phonemes to token ids.</summary>
    Lyrics,
    /// <summary>Style text to the 512-dimensional embedding.</summary>
    Style,
    /// <summary>The diffusion transformer samples the latent, block by block.</summary>
    Network,
    /// <summary>The latent is decoded to audio.</summary>
    Decoder,
}

/// <summary>What a <see cref="SongProgress"/> event reports.</summary>
public enum SongEvent
{
    /// <summary>A stage begins.</summary>
    StageStarted,
    /// <summary>A model of the stage was read into memory.</summary>
    ModelLoaded,
    /// <summary>One block of song was sampled (<c>Done</c> of <c>Total</c>).</summary>
    BlockDone,
    /// <summary>The model of the stage was released.</summary>
    ModelReleased,
    /// <summary>A stage ended.</summary>
    StageFinished,
}

/// <summary>A progress event: which stage, what happened and, for blocks, how many are done out of how many are expected.</summary>
/// <param name="Stage">The stage the event belongs to.</param>
/// <param name="Event">What happened.</param>
/// <param name="Done">For <see cref="SongEvent.BlockDone"/>: blocks sampled so far.</param>
/// <param name="Total">For block events: blocks expected (the song may end sooner).</param>
public sealed record SongProgress(SongStage Stage, SongEvent Event, int Done = 0, int Total = 0);

/// <summary>Tokens, style, noise and latent of a song, kept when <see cref="SongOptions.CaptureIntermediates"/> is set.</summary>
/// <param name="Tokens">The lyrics as token ids.</param>
/// <param name="Style">The embedding that conditioned the network, as float32 (after the cast to the network's precision).</param>
/// <param name="Noise">The drawn blocks, <c>[blocks, 1, 10, 64]</c> flattened, as float32.</param>
/// <param name="Latent">The sampler's output <c>[1, frames, 64]</c> flattened, as float32.</param>
public sealed record SongIntermediates(long[] Tokens, float[] Style, float[] Noise, float[] Latent);

/// <summary>A finished song: the mono waveform and how it was made.</summary>
/// <param name="Audio">Mono samples in -1..1 at <paramref name="SampleRate"/>.</param>
/// <param name="SampleRate">48,000 Hz.</param>
/// <param name="Frames">Latent frames in the song (<c>Audio.Length = Frames * 9600</c>).</param>
/// <param name="Seed">The seed used (the request's, or the one picked).</param>
/// <param name="EndedEarly">The sampler's end-of-song rule stopped the song before the requested length.</param>
/// <param name="BlocksSampled">Blocks of 10 frames the network sampled.</param>
/// <param name="BlocksRequested">Blocks the requested length needed.</param>
/// <param name="Request">The validated request that was served (lyrics prepared, style normalized).</param>
/// <param name="Device">The device that was used, as given in the options.</param>
/// <param name="Precision">The precision that was used.</param>
/// <param name="Elapsed">Time from the start of the call to the audio.</param>
/// <param name="Intermediates">Tokens, style, noise and latent, when the options asked to keep them.</param>
public sealed record SongResult(float[] Audio, int SampleRate, int Frames, long Seed, bool EndedEarly, int BlocksSampled, int BlocksRequested,
    ValidatedSong Request, string Device, SongPrecision Precision, TimeSpan Elapsed, SongIntermediates? Intermediates = null)
{
    /// <summary>The channels to write: stereo with the service's "fake stereo" widening, or the mono signal alone.</summary>
    public float[][] Channels(bool stereo = true) => SongAudio.ToChannels(Audio, stereo, SampleRate);

    /// <summary>Writes the song as a 16-bit WAV file (stereo by default), through a temporary file so a failure leaves nothing.</summary>
    public void WriteWav(string path, bool stereo = true) => SongAudio.WriteWavFile(path, Channels(stereo), SampleRate);

    /// <summary>The song as the bytes of a 16-bit WAV file.</summary>
    public byte[] ToWavBytes(bool stereo = true) => SongAudio.ToWavBytes(Channels(stereo), SampleRate);
}

/// <summary>
/// Makes a song from lyrics and a style text: lyrics to tokens, style text to embedding, block-wise sampling of the latent, decoding to audio. The models
/// are loaded and released one after the other, so only one is in memory at any moment. The random numbers are those of the reference service for the same
/// seed (see <see cref="NoiseSource"/>). Not thread-safe: the underlying libraries keep global state, so make one song at a time.
/// </summary>
public sealed class SongGenerator
{
    private readonly SongOptions options;
    private readonly IEspeakEngine? engine;

    /// <summary>Creates a generator. Nothing is loaded until <see cref="Generate"/> is called.</summary>
    /// <param name="options">Model directory, device and precision.</param>
    /// <param name="engine">The phonemizer's engine; the process-wide espeak-ng library when omitted.</param>
    public SongGenerator(SongOptions options, IEspeakEngine? engine = null)
    {
        this.options = options;
        this.engine = engine;
    }

    /// <summary>The device the options name, or an error if it does not exist on this machine.</summary>
    public static Device ResolveDevice(string spec)
    {
        var s = spec.Trim().ToLowerInvariant();
        if (s == "cpu") return CPU;
        if (s == "cuda" || s.StartsWith("cuda:", StringComparison.Ordinal))
        {
            int index = 0;
            if (s.Length > 4 && !int.TryParse(s.AsSpan(5), out index)) throw new DeviceUnavailableException($"\"{spec}\" is not a device: use cpu, cuda or cuda:N");
            if (!cuda.is_available()) throw new DeviceUnavailableException($"{spec} was requested but no CUDA device is available (build with -p:TorchBackend=cuda-linux or cuda-windows to use one)");
            if (index < 0 || index >= cuda.device_count()) throw new DeviceUnavailableException($"{spec} was requested but this machine has {cuda.device_count()} CUDA device(s)");
            return torch.device(DeviceType.CUDA, index);
        }
        throw new DeviceUnavailableException($"\"{spec}\" is not a device: use cpu, cuda or cuda:N");
    }

    /// <summary>The precision used for a device when none is given: float16 on CUDA, float32 on CPU.</summary>
    public static SongPrecision DefaultPrecision(DeviceType device) => device == DeviceType.CUDA ? SongPrecision.Float16 : SongPrecision.Float32;

    /// <summary>Makes a song. Throws <see cref="SongRequestException"/> before touching any model if the request is invalid, <see cref="OperationCanceledException"/> if cancelled.</summary>
    public SongResult Generate(SongRequest request, IProgress<SongProgress>? progress = null, CancellationToken cancellation = default) =>
        Run(request.Validate(), request.Lyrics, progress, cancellation);

    /// <summary>The pipeline for a request that is already validated. Tests use it for songs shorter than the service's minimum; <paramref name="originalLyrics"/> only makes error messages cite the user's line numbers.</summary>
    internal SongResult Run(ValidatedSong validated, string? originalLyrics, IProgress<SongProgress>? progress, CancellationToken cancellation)
    {
        var clock = Stopwatch.StartNew();
        var device = ResolveDevice(options.Device);
        var precision = options.Precision ?? DefaultPrecision(device.type);
        var dtype = precision == SongPrecision.Float16 ? ScalarType.Float16 : ScalarType.Float32;
        var dir = new ModelDirectory(options.ModelDirectory);
        dir.EnsureComplete();
        long seed = validated.Seed ?? NoiseSource.NewSeed();
        int blocks = (validated.Frames + NoiseSource.BlockSize - 1) / NoiseSource.BlockSize;

        void Report(SongStage stage, SongEvent ev, int done = 0, int total = 0) => progress?.Report(new SongProgress(stage, ev, done, total));
        using var noGrad = no_grad();

        // 1. lyrics to tokens (the phonemizer library and the tokenizer stay loaded: they are small)
        cancellation.ThrowIfCancellationRequested();
        Report(SongStage.Lyrics, SongEvent.StageStarted);
        var tokenizer = new LyricsTokenizer(new EnglishG2p(engine ?? EspeakEngine.Shared));
        long[] tokens = originalLyrics is null ? tokenizer.Tokenize(validated.Lyrics) : tokenizer.PrepareAndTokenize(originalLyrics);
        Report(SongStage.Lyrics, SongEvent.StageFinished);

        // 2. style text to embedding; the encoder always runs in float32 and is released before the network is loaded
        cancellation.ThrowIfCancellationRequested();
        Report(SongStage.Style, SongEvent.StageStarted);
        Tensor style;
        using (var encoder = StyleEncoder.LoadDirectory(options.ModelDirectory, device))
        {
            Report(SongStage.Style, SongEvent.ModelLoaded);
            using var embedding = encoder.Embed(validated.Style);
            style = embedding.to(dtype);
        }
        ReleaseCudaCache(device);
        Report(SongStage.Style, SongEvent.ModelReleased);
        Report(SongStage.Style, SongEvent.StageFinished);

        using var styleOwner = style;
        float[]? capturedNoise = null, capturedLatent = null;
        Tensor latent;
        bool ended;
        int sampled = 0;

        // 3. the network: seed, noise as the reference draws it, block-wise sampling
        cancellation.ThrowIfCancellationRequested();
        Report(SongStage.Network, SongEvent.StageStarted);
        var noise = NoiseSource.Draw(seed, blocks, dtype, device);          // seeded after the tokens and the style, right before sampling, as the service does
        try
        {
            if (options.CaptureIntermediates) capturedNoise = stack(noise).to(ScalarType.Float32).cpu().data<float>().ToArray();
            using (var dit = DitWeights.LoadDirectory(options.ModelDirectory, device, dtype).Model)
            {
                Report(SongStage.Network, SongEvent.ModelLoaded, 0, blocks);
                using var tokenTensor = LyricsTokenizer.ToTensor(tokens).to(device);
                latent = new BlockSampler(dit).Sample(tokenTensor, styleOwner, noise, validated.Frames, new SamplerOptions(validated.Steps, validated.CfgStrength), trace =>
                {
                    sampled = trace.Index + 1;
                    Report(SongStage.Network, SongEvent.BlockDone, sampled, blocks);
                    cancellation.ThrowIfCancellationRequested();
                });
            }
        }
        finally { foreach (var n in noise) n.Dispose(); }
        ReleaseCudaCache(device);
        Report(SongStage.Network, SongEvent.ModelReleased);
        Report(SongStage.Network, SongEvent.StageFinished);

        using var latentOwner = latent;
        int frames = (int)latent.shape[1];
        ended = frames < blocks * NoiseSource.BlockSize;
        if (frames == 0) throw new SongException("the model ended the song before any audio was produced");
        if (options.CaptureIntermediates) capturedLatent = latent.to(ScalarType.Float32).cpu().data<float>().ToArray();

        // 4. decoder
        cancellation.ThrowIfCancellationRequested();
        Report(SongStage.Decoder, SongEvent.StageStarted);
        float[] audio;
        using (var generator = BigVganWeights.LoadDirectory(options.ModelDirectory, device, dtype))
        {
            Report(SongStage.Decoder, SongEvent.ModelLoaded);
            using var channelsFirst = LatentDecoder.FromSamplerLayout(latent);
            using var wave = new LatentDecoder(generator).Decode(channelsFirst);
            audio = wave.cpu().reshape(-1).data<float>().ToArray();
        }
        ReleaseCudaCache(device);
        Report(SongStage.Decoder, SongEvent.ModelReleased);
        Report(SongStage.Decoder, SongEvent.StageFinished);

        SongIntermediates? kept = options.CaptureIntermediates
            ? new SongIntermediates(tokens, styleOwner.to(ScalarType.Float32).cpu().data<float>().ToArray(), capturedNoise!, capturedLatent!)
            : null;
        return new SongResult(audio, SongAudio.SampleRate, frames, seed, ended, sampled, blocks, validated, options.Device, precision, clock.Elapsed, kept);
    }

    /// <summary>Collects what the disposed model left behind. TorchSharp offers no way to return the CUDA allocator's cache to the driver, so on CUDA a released model's memory stays reserved by this process (and is reused by the next stage).</summary>
    private static void ReleaseCudaCache(Device device)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }
}
