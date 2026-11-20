using DiffRhythm2.Pipeline;
using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>Fake stereo and the 16-bit WAV against the reference (make_fake_stereo and pedalboard), sample for sample.</summary>
public class SongAudioTests : IDisposable
{
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-wav-").FullName;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    private static float[] Floats(Tensor t) => t.contiguous().data<float>().ToArray();

    [Fact]
    public void FakeStereo_EqualsTheReference_SampleForSample()
    {
        var input = SongReference.Audio["stereo_in"];
        var expected = SongReference.Audio["stereo_out"];
        var got = SongAudio.FakeStereo(Floats(input.reshape(-1)));
        Assert.Equal(2, got.Length);
        Assert.Equal(Floats(expected[0]), got[0]);
        Assert.Equal(Floats(expected[1]), got[1]);     // float32 multiplication by 0.8, delayed 480 samples, zeros first
    }

    [Fact]
    public void FakeStereo_Layout()
    {
        var mono = Enumerable.Range(1, 1000).Select(i => (float)i / 1000f).ToArray();
        var s = SongAudio.FakeStereo(mono);
        Assert.Equal(480, SongAudio.DelaySamples());
        Assert.Equal(mono, s[0]);
        Assert.All(s[1].Take(480), x => Assert.Equal(0f, x));
        Assert.Equal(mono[0] * 0.8f, s[1][480]);
        Assert.Equal(mono[519] * 0.8f, s[1][999]);
        Assert.Equal(1000, s[1].Length);
    }

    [Fact]
    public void Mono_IsTheSignalAlone()
    {
        var mono = new[] { 0.1f, -0.2f, 0.3f };
        var channels = SongAudio.ToChannels(mono, stereo: false);
        Assert.Single(channels);
        Assert.Equal(mono, channels[0]);
        Assert.Equal(2, SongAudio.ToChannels(mono, stereo: true).Length);
    }

    [Fact]
    public void Pcm16_EqualsWhatPedalboardWrites_ForEverySample()
    {
        var input = Floats(SongReference.Audio["pcm_in"]);
        var expected = SongReference.Audio["pcm_out"].to(ScalarType.Int32).data<int>().ToArray();
        var got = SongAudio.ToPcm16(input);
        Assert.Equal(expected.Length, got.Length);
        int differ = 0;
        for (int i = 0; i < got.Length; i++) if (got[i] != expected[i]) differ++;
        Assert.Equal(0, differ);
        Assert.True(input.Count(x => Math.Abs(x) > 1f) > 10_000, "the fixture should contain many samples beyond the interval");
    }

    [Fact]
    public void Pcm16_Extremes()
    {
        Assert.Equal(new short[] { 32767, -32768, 32767, -32768, 0 }, SongAudio.ToPcm16(new[] { 1.0f, -1.0f, 1.5f, -1.5f, 0f }));
        Assert.Throws<InvalidDataException>(() => SongAudio.ToPcm16(new[] { float.NaN }));
        Assert.Throws<InvalidDataException>(() => SongAudio.ToPcm16(new[] { float.PositiveInfinity }));
    }

    [Fact]
    public void Wav_HasTheRightHeader_AndRoundTrips()
    {
        var left = Enumerable.Range(0, 1000).Select(i => MathF.Sin(i * 0.05f) * 0.5f).ToArray();
        var right = left.Select(x => x * -0.5f).ToArray();
        var path = Path.Combine(tmp, "a.wav");
        SongAudio.WriteWavFile(path, new[] { left, right });

        var wav = WavFile.Read(path);
        Assert.Equal((ushort)1, wav.Tag);
        Assert.Equal((ushort)2, wav.Channels);
        Assert.Equal(48000u, wav.SampleRate);
        Assert.Equal((ushort)16, wav.Bits);
        Assert.Equal(2000, wav.Shorts().Length);
        var shorts = wav.Shorts();
        for (int i = 0; i < 1000; i++)
        {
            Assert.Equal(SongAudio.ToPcm16(left[i]), shorts[2 * i]);          // interleaved: left then right
            Assert.Equal(SongAudio.ToPcm16(right[i]), shorts[2 * i + 1]);
        }
        Assert.Equal(44 + 4000, new FileInfo(path).Length);
        Assert.Equal(new FileInfo(path).Length - 8, wav.RiffSize);
    }

    [Fact]
    public void Wav_ThatFails_LeavesNothingBehind()
    {
        var path = Path.Combine(tmp, "bad.wav");
        Assert.Throws<InvalidDataException>(() => SongAudio.WriteWavFile(path, new[] { new[] { 0.1f, float.NaN } }));
        Assert.Empty(Directory.GetFiles(tmp));
    }

    [Fact]
    public void Wav_MatchesTheReferenceSongFile_Header()
    {
        var reference = WavFile.Read(SongReference.E2eDir.PathOf("song.wav"));
        Assert.Equal((2, 48000u, 16), ((int)reference.Channels, reference.SampleRate, (int)reference.Bits));
        Assert.Equal(192000 * 2, reference.Shorts().Length);
    }
}
