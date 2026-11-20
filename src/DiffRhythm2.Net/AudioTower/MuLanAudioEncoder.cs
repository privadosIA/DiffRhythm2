using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.AudioTower;

/// <summary>The embedding of a piece of audio: 512 values, how many 10-second clips it came from and how long it was.</summary>
/// <param name="Values">The embedding. Each clip's embedding has unit length; their mean (this) does not, so its length is at most 1.</param>
/// <param name="Clips">Number of clips the audio was cut into.</param>
/// <param name="Seconds">Length of the audio in seconds.</param>
public sealed record AudioEmbedding(float[] Values, int Clips, double Seconds)
{
    /// <summary>Euclidean length of the embedding.</summary>
    public double Length => Math.Sqrt(Values.Sum(v => (double)v * v));
}

/// <summary>
/// Audio to the 512-dimensional space MuQ-MuLan shares with text: the front end, MuQ-large and MuLan's projections, in float32 as the reference runs them. A signal is
/// cut into 10-second clips (<see cref="AudioClips"/>), each clip is embedded alone (so memory does not grow with the length of the song) and the embeddings are
/// averaged without normalizing the mean again, which is what the reference does.
/// </summary>
public sealed class MuLanAudioEncoder : IDisposable
{
    private readonly MelFrontEnd front;
    private readonly MuLanAudioTower tower;
    private readonly bool ownsTower;

    /// <summary>The tower (for stage-level access).</summary>
    public MuLanAudioTower Tower => tower;

    /// <summary>What loading read and what it ignored, when the encoder was made by <see cref="LoadDirectory"/>.</summary>
    public MuLanAudioLoadReport? Report { get; }

    /// <summary>Wraps a loaded tower. With <paramref name="ownsTower"/> (the default) disposing the encoder disposes the tower; pass false to share a tower.</summary>
    public MuLanAudioEncoder(MuLanAudioTower tower, Device device, MuLanAudioLoadReport? report = null, bool ownsTower = true)
    {
        this.tower = tower;
        this.ownsTower = ownsTower;
        front = new MelFrontEnd(device);
        Report = report;
    }

    /// <summary>Loads <c>mulan_audio.safetensors</c> from a model directory (strictly) onto a device.</summary>
    public static MuLanAudioEncoder LoadDirectory(string dir, Device device)
    {
        var path = Path.Combine(dir, MuLanAudioWeights.CheckpointFile);
        if (!File.Exists(path)) throw new FileNotFoundException($"{MuLanAudioWeights.CheckpointFile} was not found in {dir}; create it with: diffrhythm2 convert --model {dir} --audio", path);
        var tower = new MuLanAudioTower();
        var report = MuLanAudioWeights.Load(tower, CheckpointIndex.Open(path), device);
        return new MuLanAudioEncoder(tower, device, report);
    }

    /// <summary>The embedding of one clip, <c>[1, 512]</c>, float32, unit length. The clip needs at least <see cref="AudioClips.MinimumClipSamples"/> samples.</summary>
    public Tensor EmbedClip(ReadOnlySpan<float> clip)
    {
        using var _ = no_grad();
        using var features = front.Features(clip);
        return tower.Embed(features);
    }

    /// <summary>
    /// The embedding of a signal (mono, 24 kHz): the mean of the embeddings of its clips. <paramref name="progress"/> receives the number of clips done and the total
    /// after each one.
    /// </summary>
    public AudioEmbedding Embed(ReadOnlySpan<float> signal, Action<int, int>? progress = null, CancellationToken cancellation = default)
    {
        var clips = AudioClips.Split(signal);
        using var _ = no_grad();
        var each = new List<Tensor>(clips.Count);
        try
        {
            for (int i = 0; i < clips.Count; i++)
            {
                cancellation.ThrowIfCancellationRequested();
                using var onDevice = EmbedClip(clips[i]);
                each.Add(onDevice.cpu().squeeze(0).contiguous());
                progress?.Invoke(i + 1, clips.Count);
            }
            using var mean = stack(each).mean(new long[] { 0 });
            return new AudioEmbedding(mean.data<float>().ToArray(), clips.Count, (double)signal.Length / WavReader.RequiredSampleRate);
        }
        finally { foreach (var t in each) t.Dispose(); }
    }

    /// <summary>Embeds a decoded WAV file.</summary>
    public AudioEmbedding Embed(WavSignal wav, Action<int, int>? progress = null, CancellationToken cancellation = default) => Embed(wav.Samples, progress, cancellation);

    /// <summary>Releases the front end and, if the encoder owns it, the model.</summary>
    public void Dispose()
    {
        front.Dispose();
        if (ownsTower) tower.Dispose();
    }
}
