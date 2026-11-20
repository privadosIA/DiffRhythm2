namespace DiffRhythm2.AudioTower;

/// <summary>
/// Cuts a signal into the clips MuQ-MuLan embeds one by one, exactly as the reference's <c>_get_all_clips</c> does: consecutive clips of 10 seconds from
/// the start; a remainder becomes one more clip made of the remainder followed by the first <c>240,000 - remainder</c> samples of the signal. The
/// reference slices with Python semantics, so when the signal has fewer samples than that the whole signal is appended: a signal shorter than 5 seconds
/// gives one clip of twice its length, not a 10-second clip.
/// </summary>
public static class AudioClips
{
    /// <summary>Samples in a clip: 10 seconds at 24 kHz.</summary>
    public const int ClipSamples = 10 * WavReader.RequiredSampleRate;

    /// <summary>The shortest clip the front end accepts: the spectrogram reflects 1,024 samples at each edge, which needs more than that.</summary>
    public const int MinimumClipSamples = 1025;

    /// <summary>The shortest signal that gives a clip of <see cref="MinimumClipSamples"/> samples or more.</summary>
    public const int MinimumSignalSamples = (MinimumClipSamples + 1) / 2;

    /// <summary>Cuts <paramref name="signal"/> into clips; throws <see cref="ArgumentException"/> if it is too short to be processed (the reference cannot either).</summary>
    public static IReadOnlyList<float[]> Split(ReadOnlySpan<float> signal)
    {
        int length = signal.Length;
        if (length < MinimumSignalSamples)
            throw new ArgumentException($"the audio has {length} samples; at least {MinimumSignalSamples} are needed ({MinimumSignalSamples / (double)WavReader.RequiredSampleRate * 1000:F0} ms at 24 kHz)", nameof(signal));

        var clips = new List<float[]>();
        int taken = 0;
        while (taken + ClipSamples <= length)
        {
            clips.Add(signal.Slice(taken, ClipSamples).ToArray());
            taken += ClipSamples;
        }
        if (taken < length)
        {
            int rest = length - taken;
            int wrap = Math.Min(ClipSamples - rest, length);                 // Python's audio[0:n] stops at the end of the signal
            var clip = new float[rest + wrap];
            signal[taken..].CopyTo(clip);
            signal[..wrap].CopyTo(clip.AsSpan(rest));
            clips.Add(clip);
        }
        return clips;
    }
}
