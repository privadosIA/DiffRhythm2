using MusicGen.Core.Testing;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The BigVGAN golden-run fixtures (tests/fixtures/bigvgan, stitch, latent), verified against their MANIFEST.md, and the comparison helpers for waveforms and stage signals.</summary>
public static class BigVganReference
{
    private static readonly FixtureDirectory Root = FixtureDirectory.Find();
    private static readonly Lazy<FixtureDirectory> Dir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "bigvgan")));
    private static readonly Lazy<FixtureDirectory> Fp16Dir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "bigvgan_fp16")));
    private static readonly Lazy<FixtureDirectory> StitchDir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "stitch")));
    private static readonly Dictionary<string, Dictionary<string, Tensor>> Cache = new();

    public static FixtureDirectory Bigvgan => Dir.Value;
    public static FixtureDirectory Fp16 => Fp16Dir.Value;
    public static FixtureDirectory Stitch => StitchDir.Value;

    /// <summary>A file of tests/fixtures/bigvgan loaded once: chunk_real, chunk_random, chunk_loud, decode_40, activation, weights.</summary>
    public static Dictionary<string, Tensor> Load(string name)
    {
        lock (Cache)
        {
            if (!Cache.TryGetValue(name, out var t)) Cache[name] = t = Bigvgan.Load(name + ".safetensors");
            return t;
        }
    }

    /// <summary>The golden-run latent in the layout the decoder takes, <c>[1, 64, 40]</c>.</summary>
    public static Tensor Latent40 => new FixtureDirectory(Path.Combine(Root.Dir, "latent")).Load("latent_40.safetensors")["latent"].transpose(1, 2).contiguous();

    /// <summary>
    /// Compares with the fixture and prints the measured error. The tolerance is relative to the largest magnitude of the expected tensor, so a
    /// quiet part of a signal is judged against the loudest sample, not against itself.
    /// </summary>
    public static void AssertClose(ITestOutputHelper log, string what, Tensor got, Tensor expected, double relTol)
    {
        Assert.Equal(expected.shape, got.shape);
        var g = got.to(ScalarType.Float64);
        var e = expected.to(ScalarType.Float64);
        double maxAbs = (g - e).abs().max().item<double>();
        double meanAbs = (g - e).abs().mean().item<double>();
        double scale = e.abs().max().item<double>();
        double snr = 20 * Math.Log10(e.pow(2).sum().sqrt().item<double>() / Math.Max((g - e).pow(2).sum().sqrt().item<double>(), 1e-30));
        log.WriteLine($"{what}: max_abs={maxAbs:E3} mean_abs={meanAbs:E3} scale={scale:E3} rel={maxAbs / scale:E3} snr={snr:F1} dB");
        Assert.True(maxAbs <= relTol * scale, $"{what}: max_abs {maxAbs:E3} exceeds {relTol:E1} x scale {scale:E3}");
    }
}
