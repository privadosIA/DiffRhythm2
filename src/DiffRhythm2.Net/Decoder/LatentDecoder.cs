using MusicGen.Core.Audio;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Decoder;

/// <summary>
/// Decodes a latent sequence of any length into one waveform by running the generator over overlapping chunks (<see cref="ChunkPlan"/>) and
/// pasting each chunk's audio, trimmed at the overlap midpoints, into the output. Only one chunk's activations are alive at a time; the output
/// buffer is <c>frames · 9600</c> floats (always float32, as in the reference, whatever the model precision).
/// </summary>
public sealed class LatentDecoder
{
    public const int ChunkFrames = 20;
    public const int Overlap = 5;

    private readonly Func<Tensor, Tensor> decodeChunk;
    private readonly int channels;
    private readonly int samplesPerFrame;

    public LatentDecoder(BigVganGenerator generator)
        : this(x => generator.forward(x), generator.Config.InChannels, generator.Config.SamplesPerFrame) { }

    /// <summary>Uses <paramref name="decodeChunk"/> instead of the generator; it receives <c>[b, channels, n]</c> and must return <c>[b, 1, n · samplesPerFrame]</c>. Exists for the stitching tests.</summary>
    public LatentDecoder(Func<Tensor, Tensor> decodeChunk, int channels, int samplesPerFrame)
    {
        this.decodeChunk = decodeChunk;
        this.channels = channels;
        this.samplesPerFrame = samplesPerFrame;
    }

    /// <param name="latents"><c>[b, 64, t]</c>, channels first: a sampler latent <c>[1, t, 64]</c> must be transposed (see <see cref="FromSamplerLayout"/>).</param>
    /// <returns><c>[b, 1, t · 9600]</c>, float32, in the device of the latents.</returns>
    public Tensor Decode(Tensor latents)
    {
        if (latents.dim() != 3 || latents.shape[1] != channels)
            throw new ArgumentException($"expected latents [batch, {channels}, frames], got [{string.Join(", ", latents.shape)}]: the channel count must be {channels}");

        long frames = latents.shape[2];
        var output = zeros(new long[] { latents.shape[0], 1, frames * samplesPerFrame }, ScalarType.Float32, latents.device);
        using var _ = no_grad();
        foreach (var span in ChunkPlan.Create(checked((int)frames), samplesPerFrame, ChunkFrames, Overlap))
        {
            using var scope = NewDisposeScope();
            var chunk = latents.narrow(2, span.InputStart, span.InputLength);
            var audio = decodeChunk(chunk);
            output.narrow(2, span.OutStart, span.OutEnd - span.OutStart).copy_(audio.narrow(2, span.ChunkStart, span.ChunkEnd - span.ChunkStart));
        }
        return output;
    }

    /// <summary>The sampler returns <c>[1, t, 64]</c>; the decoder takes <c>[1, 64, t]</c>.</summary>
    public static Tensor FromSamplerLayout(Tensor samplerLatent) => samplerLatent.transpose(1, 2).contiguous();

    /// <summary>Writes a decoded waveform <c>[1, 1, n]</c> (or <c>[1, n]</c>) as a mono WAV (48 kHz for the published decoder).</summary>
    public static void WriteWav(string path, Tensor waveform, int sampleRate, WavFormat format = WavFormat.Pcm16)
    {
        using var mono = waveform.dim() == 3 ? waveform.squeeze(0) : waveform;
        WavWriter.Write(path, mono, sampleRate, format);
    }
}
