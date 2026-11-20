using MusicGen.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Pipeline;

/// <summary>
/// The initial noise of a song, drawn the way the reference service draws it: the global generator is seeded (the CPU one and every CUDA device) and
/// one <c>[1, 10, 64]</c> block is drawn per block of song, in order, in the model's dtype and on the model's device (<c>torch.randn(batch, block_size,
/// num_channels, device=device, dtype=style_prompt.dtype)</c>). The numbers equal PyTorch's for the same seed on the same kind of device. The dtype
/// matters: a float16 draw is not the float32 draw cast to half.
/// </summary>
public static class NoiseSource
{
    /// <summary>Frames per block and channels of a latent frame, fixed by the model.</summary>
    public const int BlockSize = 10, Channels = 64;

    /// <summary>
    /// Seeds the generator and draws <paramref name="blocks"/> blocks. Drawing all blocks before sampling consumes the same stream as drawing each one when
    /// the sampler needs it (nothing else draws in between), also when the song ends before the last block.
    /// </summary>
    public static List<Tensor> Draw(long seed, int blocks, ScalarType dtype, Device device, int blockSize = BlockSize, int channels = Channels)
    {
        Seeding.Apply(seed);
        var noise = new List<Tensor>(blocks);
        for (int i = 0; i < blocks; i++) noise.Add(randn(new long[] { 1, blockSize, channels }, dtype, device));
        return noise;
    }

    /// <summary>A seed to use when the request has none: non-negative, short enough to read and type.</summary>
    public static long NewSeed() => Seeding.NewSeed();
}
