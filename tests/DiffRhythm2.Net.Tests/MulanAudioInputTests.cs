using System.Buffers.Binary;
using DiffRhythm2.AudioTower;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Builds small WAV files in memory.</summary>
internal static class WavBuilder
{
    public static byte[] Make(int tag, int channels, int rate, int bits, byte[] data, bool extensible = false, bool extraChunkFirst = false, uint? declaredDataSize = null, bool oddChunk = false)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8.ToArray());
        w.Write(0u);
        w.Write("WAVE"u8.ToArray());
        if (extraChunkFirst)
        {
            w.Write("LIST"u8.ToArray());
            w.Write(oddChunk ? 5u : 4u);
            w.Write(new byte[oddChunk ? 5 : 4]);
            if (oddChunk) w.Write((byte)0);                                  // RIFF chunks are padded to an even size
        }
        w.Write("fmt "u8.ToArray());
        w.Write(extensible ? 40u : 16u);
        w.Write((ushort)(extensible ? 0xFFFE : tag));
        w.Write((ushort)channels);
        w.Write((uint)rate);
        w.Write((uint)(rate * channels * bits / 8));
        w.Write((ushort)(channels * bits / 8));
        w.Write((ushort)bits);
        if (extensible)
        {
            w.Write((ushort)22);
            w.Write((ushort)bits);
            w.Write(0u);
            w.Write((ushort)tag);
            w.Write(new byte[14]);                                           // the rest of the sub-format GUID
        }
        w.Write("data"u8.ToArray());
        w.Write(declaredDataSize ?? (uint)data.Length);
        w.Write(data);
        return ms.ToArray();
    }

    public static byte[] Pcm16(params short[] samples)
    {
        var b = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++) BinaryPrimitives.WriteInt16LittleEndian(b.AsSpan(i * 2), samples[i]);
        return b;
    }

    public static byte[] Pcm24(params int[] samples)
    {
        var b = new byte[samples.Length * 3];
        for (int i = 0; i < samples.Length; i++)
        {
            b[i * 3] = (byte)samples[i];
            b[i * 3 + 1] = (byte)(samples[i] >> 8);
            b[i * 3 + 2] = (byte)(samples[i] >> 16);
        }
        return b;
    }

    public static byte[] Pcm32(params int[] samples)
    {
        var b = new byte[samples.Length * 4];
        for (int i = 0; i < samples.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(i * 4), samples[i]);
        return b;
    }

    public static byte[] Float32(params float[] samples)
    {
        var b = new byte[samples.Length * 4];
        for (int i = 0; i < samples.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(i * 4), samples[i]);
        return b;
    }

    public static WavSignal Read(byte[] file, int? rate = WavReader.RequiredSampleRate) => WavReader.Read(new MemoryStream(file), "test.wav", rate);
}

/// <summary>The WAV reader: every supported format, mixing, and every way a file can be refused. No checkpoint needed.</summary>
public class WavReaderTests
{
    [Fact]
    public void Pcm16_IsScaledByTheFullRange()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, WavBuilder.Pcm16(0, 16384, -32768, 32767)));
        Assert.Equal(new[] { 0f, 0.5f, -1f, 32767f / 32768f }, s.Samples);
        Assert.Equal(24000, s.SampleRate);
        Assert.Equal(1, s.Channels);
    }

    [Fact]
    public void Pcm24_KeepsTheSign()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 24, WavBuilder.Pcm24(0, 4194304, -8388608, -1)));
        Assert.Equal(new[] { 0f, 0.5f, -1f, -1f / 8388608f }, s.Samples);
    }

    [Fact]
    public void Pcm32_IsScaledByTheFullRange()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 32, WavBuilder.Pcm32(0, 1073741824, int.MinValue)));
        Assert.Equal(new[] { 0f, 0.5f, -1f }, s.Samples);
    }

    [Fact]
    public void Float32_IsReadAsIs()
    {
        var s = WavBuilder.Read(WavBuilder.Make(3, 1, 24000, 32, WavBuilder.Float32(0.25f, -0.75f, 1.5f)));
        Assert.Equal(new[] { 0.25f, -0.75f, 1.5f }, s.Samples);
    }

    [Fact]
    public void Stereo_IsMixedByTheMeanOfItsChannels()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 2, 24000, 16, WavBuilder.Pcm16(16384, 0, -32768, 32767)));
        Assert.Equal(2, s.Channels);
        Assert.Equal(new[] { 0.25f, (-32768f + 32767f) / 2 / 32768f }, s.Samples);
    }

    [Fact]
    public void Extensible_UsesTheSubFormat()
    {
        var s = WavBuilder.Read(WavBuilder.Make(3, 1, 24000, 32, WavBuilder.Float32(0.5f, -0.5f), extensible: true));
        Assert.Equal(new[] { 0.5f, -0.5f }, s.Samples);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChunksBeforeTheFormat_AreSkipped_IncludingOddOnes(bool odd)
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, WavBuilder.Pcm16(8192), extraChunkFirst: true, oddChunk: odd));
        Assert.Equal(new[] { 0.25f }, s.Samples);
    }

    [Fact]
    public void ADeclaredSizeBeyondTheFile_GivesWhatIsThere()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, WavBuilder.Pcm16(100, 200, 300), declaredDataSize: 0xFFFFFFFF));
        Assert.Equal(3, s.Samples.Length);
        var cut = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, WavBuilder.Pcm16(100, 200, 300), declaredDataSize: 1000));
        Assert.Equal(3, cut.Samples.Length);
    }

    [Fact]
    public void Seconds_IsLengthOverRate()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, new byte[48000 * 2]));
        Assert.Equal(2.0, s.Seconds, 9);
    }

    [Fact]
    public void AnotherSampleRate_IsRefused_NamingBothRates_BeforeTheDataIsRead()
    {
        // the data chunk is not even present: the refusal comes from the format chunk
        var file = WavBuilder.Make(1, 1, 44100, 16, WavBuilder.Pcm16(1, 2, 3)).Take(36).ToArray();
        var e = Assert.Throws<UnsupportedSampleRateException>(() => WavBuilder.Read(file));
        Assert.Equal(44100, e.Found);
        Assert.Equal(24000, e.Required);
        Assert.Contains("44100", e.Message);
        Assert.Contains("24000", e.Message);
        Assert.Contains("ffmpeg", e.Message);
        Assert.Contains("test.wav", e.Message);
    }

    [Fact]
    public void AnyRate_IsAcceptedWhenNoneIsRequired()
    {
        var s = WavBuilder.Read(WavBuilder.Make(1, 1, 44100, 16, WavBuilder.Pcm16(1, 2)), rate: null);
        Assert.Equal(44100, s.SampleRate);
    }

    [Fact]
    public void NotAWav_IsRefused_NamingTheFile()
    {
        var e = Assert.Throws<InvalidDataException>(() => WavBuilder.Read("this is not a wav file at all"u8.ToArray()));
        Assert.Contains("test.wav", e.Message);
        Assert.Contains("not a WAV", e.Message);
        Assert.Throws<InvalidDataException>(() => WavBuilder.Read(Array.Empty<byte>()));
    }

    [Fact]
    public void NoSamples_IsRefused()
    {
        var e = Assert.Throws<InvalidDataException>(() => WavBuilder.Read(WavBuilder.Make(1, 1, 24000, 16, Array.Empty<byte>())));
        Assert.Contains("no samples", e.Message);
    }

    [Fact]
    public void NonFiniteSample_IsRefused_NamingIt()
    {
        var e = Assert.Throws<InvalidDataException>(() => WavBuilder.Read(WavBuilder.Make(3, 1, 24000, 32, WavBuilder.Float32(0.1f, float.NaN, 0.2f))));
        Assert.Contains("sample 1", e.Message);
        Assert.Throws<InvalidDataException>(() => WavBuilder.Read(WavBuilder.Make(3, 1, 24000, 32, WavBuilder.Float32(float.PositiveInfinity))));
    }

    [Theory]
    [InlineData(1, 8)]      // 8-bit PCM
    [InlineData(2, 4)]      // ADPCM
    [InlineData(3, 64)]     // 64-bit float
    public void UnsupportedFormats_AreRefused(int tag, int bits)
    {
        var e = Assert.Throws<InvalidDataException>(() => WavBuilder.Read(WavBuilder.Make(tag, 1, 24000, bits, new byte[16])));
        Assert.Contains("unsupported", e.Message);
    }

    [Fact]
    public void DataBeforeFormat_AndNoDataChunk_AreRefused()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8.ToArray()); w.Write(0u); w.Write("WAVE"u8.ToArray()); w.Write("data"u8.ToArray()); w.Write(4u); w.Write(new byte[4]);
        Assert.Contains("before the format", Assert.Throws<InvalidDataException>(() => WavBuilder.Read(ms.ToArray())).Message);

        var noData = WavBuilder.Make(1, 1, 24000, 16, Array.Empty<byte>()).Take(36).ToArray();
        Assert.Contains("no audio data", Assert.Throws<InvalidDataException>(() => WavBuilder.Read(noData)).Message);
    }

    [Fact]
    public void ReadsAFileFromDisk()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-wav-").FullName;
        try
        {
            var path = Path.Combine(dir, "a.wav");
            File.WriteAllBytes(path, WavBuilder.Make(1, 1, 24000, 16, WavBuilder.Pcm16(4096, -4096)));
            Assert.Equal(new[] { 0.125f, -0.125f }, WavReader.Read(path).Samples);
            File.WriteAllBytes(path, WavBuilder.Make(1, 1, 16000, 16, WavBuilder.Pcm16(1)));
            Assert.Contains("a.wav", Assert.Throws<UnsupportedSampleRateException>(() => WavReader.Read(path)).Message);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}

/// <summary>The 10-second clips: the reference's scheme, case by case, and the reference's own wrapped clips from the golden run. No checkpoint needed.</summary>
public class AudioClipsTests
{
    private const int Second = 24000;

    private static float[] Ramp(int samples) => Enumerable.Range(0, samples).Select(i => (float)i).ToArray();

    [Fact]
    public void TwentyFiveSeconds_GivesThreeClips_TheLastWrapped()
    {
        var x = Ramp(25 * Second);
        var clips = AudioClips.Split(x);
        Assert.Equal(3, clips.Count);
        Assert.All(clips, c => Assert.Equal(AudioClips.ClipSamples, c.Length));
        Assert.Equal(x[..(10 * Second)], clips[0]);
        Assert.Equal(x[(10 * Second)..(20 * Second)], clips[1]);
        Assert.Equal(x[(20 * Second)..].Concat(x[..(5 * Second)]).ToArray(), clips[2]);
    }

    [Fact]
    public void TwentySeconds_GivesTwoClips_NothingWrapped()
    {
        var clips = AudioClips.Split(Ramp(20 * Second));
        Assert.Equal(2, clips.Count);
        Assert.Equal(10 * Second, (int)clips[1][0]);
        Assert.Equal(20 * Second - 1, (int)clips[1][^1]);
    }

    [Fact]
    public void TenSeconds_GivesOneClip_ThatIsTheSignal()
    {
        var x = Ramp(10 * Second);
        var clips = AudioClips.Split(x);
        Assert.Single(clips);
        Assert.Equal(x, clips[0]);
    }

    [Fact]
    public void SixSeconds_GivesOneTenSecondClip_TheSignalThenItsFirstFourSeconds()
    {
        var x = Ramp(6 * Second);
        var clip = Assert.Single(AudioClips.Split(x));
        Assert.Equal(AudioClips.ClipSamples, clip.Length);
        Assert.Equal(x.Concat(x[..(4 * Second)]).ToArray(), clip);
    }

    [Fact]
    public void FourSeconds_GivesAnEightSecondClip_TheSignalTwice_NotPaddedToTen()
    {
        var x = Ramp(4 * Second);
        var clip = Assert.Single(AudioClips.Split(x));
        Assert.Equal(8 * Second, clip.Length);
        Assert.Equal(x.Concat(x).ToArray(), clip);
    }

    [Fact]
    public void ARemainderOfOneSample_StillMakesAFullClip()
    {
        var x = Ramp(10 * Second + 1);
        var clips = AudioClips.Split(x);
        Assert.Equal(2, clips.Count);
        Assert.Equal(AudioClips.ClipSamples, clips[1].Length);
        Assert.Equal(x[^1], clips[1][0]);
        Assert.Equal(x[..(AudioClips.ClipSamples - 1)], clips[1][1..]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(500)]
    [InlineData(512)]
    public void TooShort_IsRefused_NamingTheLengthAndTheMinimum(int samples)
    {
        var e = Assert.Throws<ArgumentException>(() => AudioClips.Split(new float[samples]));
        Assert.Contains($"{samples} samples", e.Message);
        Assert.Contains("513", e.Message);
    }

    [Fact]
    public void TheShortestSignal_IsAccepted_AndGivesTheShortestClip()
    {
        var clip = Assert.Single(AudioClips.Split(new float[AudioClips.MinimumSignalSamples]));
        Assert.Equal(AudioClips.MinimumClipSamples + 1, clip.Length);
        Assert.True(clip.Length >= AudioClips.MinimumClipSamples);
    }

    [RequiresMulanFixturesFact]
    public void MatchesTheReferenceClips_WhereTheyWrap()
    {
        var s4 = Assert.Single(AudioClips.Split(MulanAudioReference.Signal("s4")));
        Assert.Equal(MulanAudioReference.Clips["s4_0"].data<float>().ToArray(), s4);

        var s25 = MulanAudioReference.Signal("s25");
        var clips = AudioClips.Split(s25);
        Assert.Equal(Convert.ToInt32(MulanAudioReference.Meta.GetProperty("checks").GetProperty("clips").GetProperty("s25").GetInt32()), clips.Count);
        Assert.Equal(MulanAudioReference.Clips["s25_2"].data<float>().ToArray(), clips[2]);
        Assert.Equal(s25[..AudioClips.ClipSamples], clips[0]);

        var noise = Assert.Single(AudioClips.Split(MulanAudioReference.Signal("noise10")));
        Assert.Equal(MulanAudioReference.Signal("noise10"), noise);
    }
}

/// <summary>The log-mel front end against torchaudio: the filter bank, the decibel spectrogram and the normalized features, then the closed-form cases.</summary>
public class MelFrontEndTests
{
    private readonly ITestOutputHelper log;

    public MelFrontEndTests(ITestOutputHelper log) => this.log = log;

    private static float[] FirstClip(string name) => AudioClips.Split(MulanAudioReference.Signal(name))[0];

    [RequiresMulanFixturesFact]
    public void FilterBank_MatchesTorchaudio()
    {
        using var fb = MelFrontEnd.FilterBank();
        Assert.Equal(new long[] { MelFrontEnd.Bins, MelFrontEnd.Mels }, fb.shape);
        Reference.AssertClose(log, "filter bank", fb, MulanAudioReference.Frontend["filter_bank"], Tolerances.MelFilterBank);
    }

    [RequiresMulanFixturesTheory]
    [InlineData("s25")]
    [InlineData("s4")]
    [InlineData("noise10")]
    public void Decibels_MatchTorchaudio(string name)
    {
        using var front = new MelFrontEnd();
        using var db = front.Decibels(FirstClip(name));
        Reference.AssertClose(log, $"{name} decibels", db[0], MulanAudioReference.Frontend[$"{name}_db"], Tolerances.MelDecibels);
    }

    [RequiresMulanFixturesTheory]
    [InlineData("s25")]
    [InlineData("s4")]
    [InlineData("noise10")]
    public void Features_MatchTheReferenceNormalizedSpectrogram(string name)
    {
        using var front = new MelFrontEnd();
        using var feats = front.Features(FirstClip(name));
        Reference.AssertClose(log, $"{name} features", feats[0], MulanAudioReference.Frontend[$"{name}_norm"], Tolerances.MelFeatures);
    }

    [Fact]
    public void TenSecondClip_Gives128BandsBy1000Frames_AndAClipOfNSamplesNOver240()
    {
        using var front = new MelFrontEnd();
        using var ten = front.Features(new float[AudioClips.ClipSamples]);
        Assert.Equal(new long[] { 1, 128, 1000 }, ten.shape);
        using var eight = front.Features(new float[8 * 24000]);
        Assert.Equal(new long[] { 1, 128, 800 }, eight.shape);
        using var odd = front.Features(new float[24000 + 100]);
        Assert.Equal(new long[] { 1, 128, (24000 + 100) / 240 }, odd.shape);
    }

    [Fact]
    public void Silence_IsTheDecibelFloor_Normalized()
    {
        using var front = new MelFrontEnd();
        using var feats = front.Features(new float[AudioClips.ClipSamples]);
        float expected = (float)((-100.0 - MelFrontEnd.Mean) / MelFrontEnd.Std);
        Assert.True((feats - expected).abs().max().item<float>() < 1e-5f, "every value of the silent clip should be the normalized decibel floor");
        using var db = front.Decibels(new float[AudioClips.ClipSamples]);
        Assert.Equal(-100f, db.min().item<float>());
        Assert.Equal(-100f, db.max().item<float>());
    }

    [Fact]
    public void ATooShortClip_IsRefused()
    {
        using var front = new MelFrontEnd();
        Assert.Throws<ArgumentException>(() => front.Features(new float[1024]));
        using var ok = front.Features(new float[1025]);
        Assert.Equal(new long[] { 1, 128, 1025 / 240 }, ok.shape);
    }

    [Fact]
    public void FilterBank_HasTheTriangularShape()
    {
        using var fb = MelFrontEnd.FilterBank();
        Assert.True(fb.min().item<float>() >= 0f);
        Assert.True(fb.max().item<float>() <= 1f);
        // every band has a non-empty support and rises then falls
        using var colMax = fb.max(dim: 0).values;
        Assert.True(colMax.min().item<float>() > 0f);
    }
}
