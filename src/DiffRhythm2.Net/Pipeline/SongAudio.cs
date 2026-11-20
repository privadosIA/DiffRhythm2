using System.Buffers.Binary;

namespace DiffRhythm2.Pipeline;

/// <summary>
/// The last steps of a song, as the reference service does them: the optional "fake stereo" and the conversion of float samples to 16-bit integers, then a
/// 16-bit PCM WAV file. Both are bit-exact with the service (the stereo expression of <c>make_fake_stereo</c> and the conversion of its audio library).
/// </summary>
public static class SongAudio
{
    /// <summary>Sample rate of the decoder, 48 kHz.</summary>
    public const int SampleRate = 48_000;

    /// <summary>The right channel is the signal times this, in float32.</summary>
    public const float RightGain = 0.8f;

    /// <summary>The delay of the right channel: <c>int(0.01 * sampleRate)</c> samples, 10 ms.</summary>
    public static int DelaySamples(int sampleRate = SampleRate) => (int)(0.01 * sampleRate);

    /// <summary>
    /// <c>make_fake_stereo</c>: the left channel is the mono signal, the right one is the signal times 0.8 delayed by 10 ms (zeros first). The length is unchanged.
    /// </summary>
    public static float[][] FakeStereo(ReadOnlySpan<float> mono, int sampleRate = SampleRate)
    {
        int delay = DelaySamples(sampleRate);
        var left = mono.ToArray();
        var right = new float[mono.Length];
        for (int i = delay; i < mono.Length; i++) right[i] = mono[i - delay] * RightGain;
        return new[] { left, right };
    }

    /// <summary>The channels of the output: stereo with the fake widening, or the mono signal alone.</summary>
    public static float[][] ToChannels(ReadOnlySpan<float> mono, bool stereo, int sampleRate = SampleRate) =>
        stereo ? FakeStereo(mono, sampleRate) : new[] { mono.ToArray() };

    /// <summary>
    /// One float sample to a 16-bit integer: clamp to -1..1, multiply by 2,147,483,647 in double precision, round to the nearest integer (halves to even) and
    /// keep the top 16 bits of that 32-bit value. This is what pedalboard (JUCE) writes; <c>round(x * 32767)</c> differs by up to 2 steps.
    /// </summary>
    public static short ToPcm16(float sample)
    {
        if (!float.IsFinite(sample)) throw new InvalidDataException("the audio contains a sample that is not a finite number");
        double scaled = Math.Round(Math.Clamp((double)sample, -1.0, 1.0) * 2147483647.0, MidpointRounding.ToEven);
        return (short)((int)scaled >> 16);
    }

    /// <summary>Converts a whole signal with <see cref="ToPcm16(float)"/>.</summary>
    public static short[] ToPcm16(ReadOnlySpan<float> samples)
    {
        var pcm = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++) pcm[i] = ToPcm16(samples[i]);
        return pcm;
    }

    /// <summary>Writes the channels as a 16-bit PCM WAV (RIFF, little endian, interleaved) to a stream.</summary>
    public static void WriteWav(Stream stream, IReadOnlyList<float[]> channels, int sampleRate = SampleRate)
    {
        if (channels.Count is < 1 or > 65535) throw new ArgumentException("a WAV needs at least one channel");
        int length = channels[0].Length;
        if (channels.Any(c => c.Length != length)) throw new ArgumentException("the channels differ in length");
        long dataBytes = (long)length * channels.Count * 2;
        if (dataBytes + 36 > uint.MaxValue) throw new ArgumentException("the audio is too long for a WAV file");

        var header = new byte[44];
        "RIFF"u8.CopyTo(header);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(4), (uint)(36 + dataBytes));
        "WAVEfmt "u8.CopyTo(header.AsSpan(8));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(16), 16);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(20), 1);                                        // PCM
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(22), (ushort)channels.Count);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(24), (uint)sampleRate);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(28), (uint)(sampleRate * channels.Count * 2));   // byte rate
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(32), (ushort)(channels.Count * 2));            // block align
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(34), 16);
        "data"u8.CopyTo(header.AsSpan(36));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(40), (uint)dataBytes);

        // the samples are converted first, so that a bad sample fails before a byte is written
        var buffer = new byte[dataBytes];
        for (int c = 0; c < channels.Count; c++)
            for (int i = 0; i < length; i++)
                BinaryPrimitives.WriteInt16LittleEndian(buffer.AsSpan((i * channels.Count + c) * 2), ToPcm16(channels[c][i]));
        stream.Write(header);
        stream.Write(buffer);
    }

    /// <summary>The WAV file as bytes.</summary>
    public static byte[] ToWavBytes(IReadOnlyList<float[]> channels, int sampleRate = SampleRate)
    {
        using var ms = new MemoryStream();
        WriteWav(ms, channels, sampleRate);
        return ms.ToArray();
    }

    /// <summary>Writes the WAV to a temporary file next to <paramref name="path"/> and moves it into place, so that a failure leaves no partial file.</summary>
    public static void WriteWavFile(string path, IReadOnlyList<float[]> channels, int sampleRate = SampleRate)
    {
        var full = Path.GetFullPath(path);
        var temp = full + "." + Guid.NewGuid().ToString("N")[..8] + ".tmp";
        try
        {
            using (var fs = File.Create(temp)) WriteWav(fs, channels, sampleRate);
            File.Move(temp, full, overwrite: true);
        }
        finally { if (File.Exists(temp)) File.Delete(temp); }
    }
}
