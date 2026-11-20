using DiffRhythm2.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Sampling;

/// <param name="Steps">Time points of the Euler grid per block (the reference default is 16, i.e. 15 integration steps).</param>
/// <param name="CfgStrength">Classifier-free guidance strength; below 1e-5 the unconditional branch is not evaluated.</param>
public sealed record SamplerOptions(int Steps = 16, double CfgStrength = 2.0);

/// <summary>What happened in one block; handed to the optional observer (the golden-run tests compare every figure).</summary>
/// <param name="Index">Zero-based position of the block in the song.</param>
/// <param name="Noise">The noise tensor the block started from.</param>
/// <param name="States">The state at every grid point; <c>States[0]</c> is the noise and the last one is the sampled block.</param>
/// <param name="Sampled">The sampled block, as returned in the song's latent.</param>
/// <param name="VelocityCalls">Integration steps taken (grid points - 1).</param>
/// <param name="ModelForwards">DiT evaluations used by those steps: one per step, two with guidance.</param>
public sealed record BlockTrace(int Index, Tensor Noise, IReadOnlyList<Tensor> States, Tensor Sampled, int VelocityCalls, int ModelForwards);

/// <summary>
/// DiffRhythm2's block-wise guided flow-matching sampler (<c>CFM.sample_block_cache</c>). The lyrics are prefilled once into two caches (conditional, and
/// unconditional with filler tokens and a zero style); every block of 10 frames is then integrated from noise to a clean block with Euler steps and
/// classifier-free guidance, fed back into both caches at time 1, and appended to the song. Randomness is not drawn here: the initial noise of every
/// block is an input, so a run is reproducible and comparable with the reference.
/// </summary>
public sealed class BlockSampler
{
    private readonly Dit model;
    private readonly int blockSize;
    private readonly int channels;

    public BlockSampler(Dit model)
    {
        this.model = model;
        blockSize = model.Config.BlockSize;
        channels = model.Config.MelDim;
    }

    /// <summary>Blocks needed for <paramref name="frames"/>: <c>ceil(frames / blockSize)</c>.</summary>
    public int BlocksFor(int frames) => (frames + blockSize - 1) / blockSize;

    /// <param name="tokens">Lyric token ids <c>[L]</c> (long); empty means generation from the style alone.</param>
    /// <param name="style"><c>[1, 512]</c> style embedding.</param>
    /// <param name="noise">One <c>[1, blockSize, channels]</c> tensor per block, at least <see cref="BlocksFor"/> of them.</param>
    /// <param name="frames">Requested length. Whole blocks are sampled, so the result can be longer than this (up to the next multiple of the block size, as in the reference) and is shorter when the song ends earlier.</param>
    /// <param name="options">Steps and guidance strength; null means the reference defaults (16 steps, strength 2).</param>
    /// <param name="observer">Called once per block with what happened in it; null for none.</param>
    /// <returns><c>[1, T, channels]</c> exactly as the reference returns it (before any transposition for the decoder).</returns>
    public Tensor Sample(Tensor tokens, Tensor style, IReadOnlyList<Tensor> noise, int frames, SamplerOptions? options = null, Action<BlockTrace>? observer = null)
    {
        options ??= new SamplerOptions();
        Validate(tokens, style, noise, frames, options);
        using var _ = no_grad();
        int blocks = BlocksFor(frames);
        long textLen = tokens.shape[0];
        var dtype = style.dtype;                                 // the model's dtype: times, grid, noise and the stream follow it, as in the reference
        var zeroStyle = zeros_like(style);
        var grid = EulerSolver.TimeGrid(options.Steps, dtype).to(style.device);
        bool guided = options.CfgStrength >= 1e-5;

        using var cond = new BlockCache(model.Config.Depth);
        using var uncond = new BlockCache(model.Config.Depth);
        if (textLen > 0)
        {
            var textTime = full(new long[] { 1, textLen }, -1, dtype: dtype, device: style.device);
            var positions = arange(textLen, device: style.device).unsqueeze(0);
            model.Prefill(model.EmbedText(tokens.unsqueeze(0)), textTime, style, positions, cond, CacheUse.TextPrefill);
            if (guided)
                model.Prefill(model.EmbedText(zeros_like(tokens).unsqueeze(0)), textTime, zeroStyle, positions, uncond, CacheUse.TextPrefill);
        }

        var stream = zeros(1, 0, channels, dtype: dtype, device: style.device);
        for (int bid = 0; bid < blocks; bid++)
        {
            long cleanLen = stream.shape[1];
            var blockPositions = arange(cleanLen, cleanLen + blockSize, device: style.device).unsqueeze(0);   // continue from the clean frames; the text does not count

            int velocityCalls = 0, modelForwards = 0;
            Tensor Velocity(Tensor t, Tensor x)
            {
                using var scope = NewDisposeScope();
                velocityCalls++;
                modelForwards++;
                var embedded = model.EmbedLatent(x);
                var time = t.reshape(1, 1).expand(1, blockSize).contiguous();
                var pred = model.Forward(embedded, time, style, blockPositions, cond, CacheUse.Transient);
                if (!guided) return pred.MoveToOuterDisposeScope();
                modelForwards++;
                var nullPred = model.Forward(embedded, time, zeroStyle, blockPositions, uncond, CacheUse.Transient);
                return (pred + (pred - nullPred) * options.CfgStrength).MoveToOuterDisposeScope();
            }

            var states = observer is null ? null : new List<Tensor>();
            var sampled = EulerSolver.Integrate(Velocity, noise[bid].to(dtype, style.device), grid, states);
            observer?.Invoke(new BlockTrace(bid, noise[bid], states!, sampled.detach().clone(), velocityCalls, modelForwards));

            var grown = cat(new[] { stream, sampled }, dim: 1);
            stream.Dispose();
            stream = grown;

            if (EndOfSong.EndPosition(stream) is { } end)
            {
                sampled.Dispose();
                var trimmed = stream.narrow(1, 0, end).clone();
                stream.Dispose();
                return trimmed;
            }

            // The reference writes the block into the caches before checking for the end, even after the last block; the result is the same without it.
            if (bid + 1 < blocks)
            {
                var cleanTime = ones(1, blockSize, dtype: dtype, device: style.device);
                var cleanEmbedded = model.EmbedLatent(sampled);
                model.Prefill(cleanEmbedded, cleanTime, style, blockPositions, cond, CacheUse.CleanWrite);
                if (guided) model.Prefill(cleanEmbedded, cleanTime, zeroStyle, blockPositions, uncond, CacheUse.CleanWrite);
            }
            sampled.Dispose();
        }
        return stream;
    }

    private void Validate(Tensor tokens, Tensor style, IReadOnlyList<Tensor> noise, int frames, SamplerOptions options)
    {
        if (frames <= 0) throw new ArgumentException($"frames must be positive (got {frames})");
        if (options.Steps < 2) throw new ArgumentException($"steps must be at least 2 (got {options.Steps})");
        if (tokens.dim() != 1 || tokens.dtype != ScalarType.Int64) throw new ArgumentException("tokens must be a 1-D int64 tensor");
        if (style.dim() != 2 || style.shape[0] != 1 || style.shape[1] != DitConfig.CondDim)
            throw new ArgumentException($"style must be [1, {DitConfig.CondDim}] (got [{string.Join(", ", style.shape)}])");
        int needed = BlocksFor(frames);
        if (noise.Count < needed)
            throw new ArgumentException($"{frames} frames need {needed} noise blocks of [1, {blockSize}, {channels}] and only {noise.Count} were given (short by {needed - noise.Count})");
        for (int i = 0; i < needed; i++)
            if (noise[i].dim() != 3 || noise[i].shape[0] != 1 || noise[i].shape[1] != blockSize || noise[i].shape[2] != channels)
                throw new ArgumentException($"noise block {i} must be [1, {blockSize}, {channels}] (got [{string.Join(", ", noise[i].shape)}])");
    }
}
