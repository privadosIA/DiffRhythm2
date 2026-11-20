using System.Buffers.Binary;

namespace DiffRhythm2.AudioTower;

/// <summary>A decoded WAV file: mono samples in -1..1 (several channels are mixed by their mean), and what the file said about itself.</summary>
/// <param name="Samples">The mono signal.</param>
/// <param name="SampleRate">Sample rate of the file in Hz.</param>
/// <param name="Channels">Number of channels the file had.</param>
public sealed record WavSignal(float[] Samples, int SampleRate, int Channels)
{
    /// <summary>Duration in seconds.</summary>
    public double Seconds => (double)Samples.Length / SampleRate;
}

/// <summary>The sample rate of a WAV file is not the one the model takes; carries both so a caller can say what to do about it.</summary>
public sealed class UnsupportedSampleRateException : Exception
{
    /// <summary>Rate found in the file, in Hz.</summary>
    public int Found { get; }

    /// <summary>Rate the model takes, in Hz.</summary>
    public int Required { get; }

    /// <summary>Creates the exception with the message that names the file, both rates and the ffmpeg command that fixes it.</summary>
    public UnsupportedSampleRateException(string name, int found, int required)
        : base($"{name}: the sample rate is {found} Hz but {required} Hz is required (MuQ-MuLan takes 24 kHz audio and this program does not resample); " +
               $"convert it first, for example: ffmpeg -i INPUT -ac 1 -ar {required} OUTPUT.wav")
    {
        Found = found;
        Required = required;
    }
}

/// <summary>
/// A minimal WAV reader: 16, 24 and 32-bit integer PCM and 32-bit float, any number of channels (mixed to mono by their mean), WAVE_FORMAT_EXTENSIBLE
/// included. Integer samples are divided by the full range of their type (32768, 8388608, 2147483648). It never resamples: a file whose rate is not
/// the required one is refused as soon as its format chunk has been read, before any sample is decoded.
/// </summary>
public static class WavReader
{
    /// <summary>The sample rate of MuQ-MuLan's audio input.</summary>
    public const int RequiredSampleRate = 24000;

    /// <summary>Reads a WAV file; <paramref name="requiredRate"/> null accepts any rate.</summary>
    public static WavSignal Read(string path, int? requiredRate = RequiredSampleRate)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
        return Read(stream, Path.GetFileName(path), requiredRate);
    }

    /// <summary>Reads a WAV file from a stream; <paramref name="name"/> is only used in messages.</summary>
    public static WavSignal Read(Stream stream, string name, int? requiredRate = RequiredSampleRate)
    {
        Span<byte> head = stackalloc byte[12];
        if (!TryReadExactly(stream, head) || !head[..4].SequenceEqual("RIFF"u8) || !head[8..12].SequenceEqual("WAVE"u8))
            throw new InvalidDataException($"{name}: not a WAV file (the RIFF/WAVE header is missing)");

        Format? format = null;
        Span<byte> chunk = stackalloc byte[8];
        while (TryReadExactly(stream, chunk))
        {
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(chunk[4..]);
            if (chunk[..4].SequenceEqual("fmt "u8))
            {
                if (size < 16) throw new InvalidDataException($"{name}: the format chunk is {size} bytes, 16 are needed");
                var body = new byte[size];
                if (!TryReadExactly(stream, body)) throw new InvalidDataException($"{name}: the file ends inside the format chunk");
                format = ParseFormat(name, body);
                if (requiredRate is { } rate && format.SampleRate != rate) throw new UnsupportedSampleRateException(name, format.SampleRate, rate);
                SkipPad(stream, size);
            }
            else if (chunk[..4].SequenceEqual("data"u8))
            {
                if (format is null) throw new InvalidDataException($"{name}: the data chunk comes before the format chunk");
                return ReadData(stream, name, format, size);
            }
            else
            {
                if (!Skip(stream, (long)size + (size & 1))) break;
            }
        }
        throw new InvalidDataException(format is null ? $"{name}: no format chunk found" : $"{name}: no audio data chunk found");
    }

    private sealed record Format(int Tag, int Channels, int SampleRate, int BlockAlign, int Bits);

    private static Format ParseFormat(string name, byte[] body)
    {
        int tag = BinaryPrimitives.ReadUInt16LittleEndian(body);
        int channels = BinaryPrimitives.ReadUInt16LittleEndian(body.AsSpan(2));
        int rate = (int)BinaryPrimitives.ReadUInt32LittleEndian(body.AsSpan(4));
        int align = BinaryPrimitives.ReadUInt16LittleEndian(body.AsSpan(12));
        int bits = BinaryPrimitives.ReadUInt16LittleEndian(body.AsSpan(14));
        if (tag == 0xFFFE && body.Length >= 26) tag = BinaryPrimitives.ReadUInt16LittleEndian(body.AsSpan(24));   // WAVE_FORMAT_EXTENSIBLE: the sub-format's first two bytes
        if (channels < 1) throw new InvalidDataException($"{name}: the file says it has {channels} channels");
        if (rate < 1) throw new InvalidDataException($"{name}: the file says its sample rate is {rate}");
        bool ok = (tag == 1 && bits is 16 or 24 or 32) || (tag == 3 && bits == 32);
        if (!ok)
            throw new InvalidDataException($"{name}: unsupported format (tag {tag}, {bits} bits); 16, 24 and 32-bit PCM and 32-bit float are supported");
        if (align != channels * bits / 8) throw new InvalidDataException($"{name}: the block size {align} does not match {channels} channels of {bits} bits");
        return new Format(tag, channels, rate, align, bits);
    }

    private static WavSignal ReadData(Stream stream, string name, Format f, uint declared)
    {
        long available = stream.CanSeek ? stream.Length - stream.Position : long.MaxValue;
        long bytes = declared == 0xFFFFFFFF || declared > available ? available : declared;     // streamed files declare no size; a truncated file gives what it has
        long framesLong = bytes == long.MaxValue ? long.MaxValue : bytes / f.BlockAlign;
        if (framesLong == 0) throw new InvalidDataException($"{name}: the file has no samples");
        if (framesLong > int.MaxValue) throw new InvalidDataException($"{name}: the file is too long");

        var samples = new List<float>(framesLong == long.MaxValue ? 1 << 20 : (int)framesLong);
        const int blockFrames = 1 << 14;
        var buffer = new byte[blockFrames * f.BlockAlign];
        long remaining = framesLong;
        int bytesPerSample = f.Bits / 8;
        while (remaining > 0)
        {
            int want = (int)Math.Min(remaining, blockFrames);
            int got = ReadUpTo(stream, buffer, want * f.BlockAlign);
            int frames = got / f.BlockAlign;
            for (int i = 0; i < frames; i++)
            {
                double sum = 0;
                for (int c = 0; c < f.Channels; c++)
                {
                    var s = buffer.AsSpan(i * f.BlockAlign + c * bytesPerSample, bytesPerSample);
                    float v = f.Tag == 3 ? BinaryPrimitives.ReadSingleLittleEndian(s) : f.Bits switch
                    {
                        16 => BinaryPrimitives.ReadInt16LittleEndian(s) / 32768f,
                        24 => (((s[2] << 24) | (s[1] << 16) | (s[0] << 8)) >> 8) / 8388608f,
                        _ => BinaryPrimitives.ReadInt32LittleEndian(s) / 2147483648f,
                    };
                    if (!float.IsFinite(v)) throw new InvalidDataException($"{name}: sample {samples.Count} is not a finite number");
                    sum += v;
                }
                samples.Add((float)(sum / f.Channels));
            }
            remaining -= frames;
            if (frames < want) break;                                                            // the stream ended early
        }
        if (samples.Count == 0) throw new InvalidDataException($"{name}: the file has no samples");
        return new WavSignal(samples.ToArray(), f.SampleRate, f.Channels);
    }

    private static bool TryReadExactly(Stream s, Span<byte> buffer)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int n = s.Read(buffer[total..]);
            if (n == 0) return false;
            total += n;
        }
        return true;
    }

    private static int ReadUpTo(Stream s, byte[] buffer, int count)
    {
        int total = 0;
        while (total < count)
        {
            int n = s.Read(buffer, total, count - total);
            if (n == 0) break;
            total += n;
        }
        return total;
    }

    private static void SkipPad(Stream s, uint size)
    {
        if ((size & 1) != 0) Skip(s, 1);
    }

    private static bool Skip(Stream s, long count)
    {
        if (s.CanSeek)
        {
            if (s.Position + count > s.Length) return false;
            s.Seek(count, SeekOrigin.Current);
            return true;
        }
        var scratch = new byte[Math.Min(count, 1 << 16)];
        while (count > 0)
        {
            int n = s.Read(scratch, 0, (int)Math.Min(count, scratch.Length));
            if (n == 0) return false;
            count -= n;
        }
        return true;
    }
}
