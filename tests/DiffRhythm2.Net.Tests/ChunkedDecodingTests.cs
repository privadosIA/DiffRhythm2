using System.Text.Json;
using DiffRhythm2.Decoder;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The chunk geometry against the real <c>decode_audio</c> (tests/fixtures/stitch), run with a stand-in decoder so that no weights are involved.</summary>
public class ChunkedDecodingTests
{
    private const int Spf = 9600;
    private readonly ITestOutputHelper log;

    public ChunkedDecodingTests(ITestOutputHelper log) => this.log = log;

    /// <summary>The reference's stand-in: the n-th chunk (1-based) returns a waveform filled with n.</summary>
    private sealed class Tagging
    {
        public readonly List<long> Frames = new();

        public Tensor Decode(Tensor x)
        {
            Frames.Add(x.shape[2]);
            return full(new long[] { x.shape[0], 1, x.shape[2] * Spf }, (double)Frames.Count);
        }
    }

    private static JsonElement Stitch(int t)
    {
        var doc = JsonDocument.Parse(BigVganReference.Stitch.ReadAllText("stitch.json"));
        return doc.RootElement.GetProperty("lengths").GetProperty(t.ToString());
    }

    public static IEnumerable<object[]> Lengths => new[] { 20, 25, 35, 40, 47, 64 }.Select(t => new object[] { t });

    [Theory]
    [MemberData(nameof(Lengths))]
    public void StandInDecoder_PlacesEveryChunkWhereTheReferenceDoes(int t)
    {
        var expected = Stitch(t).GetProperty("chunk_per_frame").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var tagging = new Tagging();
        var decoder = new LatentDecoder(tagging.Decode, 64, Spf);

        using var wave = decoder.Decode(zeros(1, 64, t));
        Assert.Equal(new long[] { 1, 1, (long)t * Spf }, wave.shape);
        Assert.Equal(Stitch(t).GetProperty("chunk_frames").EnumerateArray().Select(e => e.GetInt64()).ToArray(), tagging.Frames.ToArray());

        // every output SAMPLE, not just one per frame
        var samples = wave.data<float>().ToArray();
        for (int frame = 0; frame < t; frame++)
            for (int s = 0; s < Spf; s += 1)
                if (samples[frame * Spf + s] != expected[frame]) Assert.Fail($"T={t}: frame {frame} sample {s} is from chunk {samples[frame * Spf + s]}, the reference says {expected[frame]}");
        log.WriteLine($"T={t}: {tagging.Frames.Count} chunks, per frame {string.Join("", expected)}");
    }

    [Fact]
    public void ExactFit_T35_DecodesTwoChunksStartingAt0And15()
    {
        var plan = ChunkPlan.Create(35, Spf);
        Assert.Equal(2, plan.Count);
        Assert.Equal(new[] { 0, 15 }, plan.Select(p => p.InputStart).ToArray());
        Assert.Equal(18L * Spf, plan[0].OutEnd);          // the earlier chunk stops 2 frames before its end
        Assert.Equal(17L * Spf, plan[1].OutStart);        // the later one starts 2 frames after its start
        Assert.Equal(2L * Spf, plan[1].ChunkStart);
    }

    [Fact]
    public void AlignedFinalChunk_T25_CoversFrames0To19AndThen5To24_WithoutGaps()
    {
        var plan = ChunkPlan.Create(25, Spf);
        Assert.Equal(new[] { (0, 20), (5, 20) }, plan.Select(p => (p.InputStart, p.InputLength)).ToArray());
        var covered = new bool[25 * Spf / 1000];
        foreach (var p in plan)
            for (long s = p.OutStart; s < p.OutEnd; s += 1000) covered[s / 1000] = true;
        Assert.All(covered, Assert.True);
        // each pasted piece has the length of the audio it copies
        Assert.All(plan, p => Assert.Equal(p.OutEnd - p.OutStart, p.ChunkEnd - p.ChunkStart));
    }

    [Theory]
    [InlineData(7)]
    [InlineData(19)]
    public void ShorterThanOneChunk_IsDecodedAsASingleChunkEqualToDirectDecoding(int t)
    {
        // a deterministic stand-in whose output depends on the input
        Tensor Direct(Tensor x) => x.mean(new long[] { 1 }, keepdim: true).repeat_interleave(Spf, dim: 2);
        using var latents = randn(1, 64, t);
        using var viaDecoder = new LatentDecoder(Direct, 64, Spf).Decode(latents);
        using var direct = Direct(latents);

        Assert.Equal(new long[] { 1, 1, (long)t * Spf }, viaDecoder.shape);
        Assert.Equal(0f, (viaDecoder - direct).abs().max().item<float>());
        Assert.Single(ChunkPlan.Create(t, Spf));
    }

    [Fact]
    public void EmptySequence_ReturnsAnEmptyWaveform_WithoutCallingTheDecoder()
    {
        var tagging = new Tagging();
        using var wave = new LatentDecoder(tagging.Decode, 64, Spf).Decode(zeros(1, 64, 0));
        Assert.Equal(new long[] { 1, 1, 0 }, wave.shape);
        Assert.Empty(tagging.Frames);
    }

    [Fact]
    public void WrongChannelCount_FailsStatingExpectedAndActual()
    {
        var e = Assert.Throws<ArgumentException>(() => new LatentDecoder(new Tagging().Decode, 64, Spf).Decode(zeros(1, 63, 20)));
        Assert.Contains("64", e.Message);
        Assert.Contains("63", e.Message);
    }

    [Fact]
    public void SamplerLayout_IsTransposedToChannelsFirst_AndKeepsTheValues()
    {
        var sampler = arange(0, 3 * 64).reshape(1, 3, 64).to(ScalarType.Float32);
        var x = LatentDecoder.FromSamplerLayout(sampler);
        Assert.Equal(new long[] { 1, 64, 3 }, x.shape);
        Assert.Equal(sampler[0, 2, 5].item<float>(), x[0, 5, 2].item<float>());
    }

    [Fact]
    public void WorkingSet_OfTheStandIn_HoldsOneChunkAtATime()
    {
        // The decoder disposes each chunk's audio before the next one: after a long sequence nothing but the output is alive.
        var tagging = new Tagging();
        using var wave = new LatentDecoder(tagging.Decode, 64, Spf).Decode(zeros(1, 64, 400));
        Assert.Equal(400L * Spf, wave.shape[2]);
        Assert.Equal(27, tagging.Frames.Count);
    }
}
