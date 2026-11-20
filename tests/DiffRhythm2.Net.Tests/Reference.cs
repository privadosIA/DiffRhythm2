using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The golden-run fixtures (tests/fixtures/*, verified against their MANIFEST.md) and the helpers to compare against them.</summary>
public static class Reference
{
    private static readonly FixtureDirectory Root = FixtureDirectory.Find();
    private static readonly Lazy<FixtureDirectory> DitDir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "dit")));
    private static readonly Lazy<FixtureDirectory> EosDir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "eos")));
    private static readonly Lazy<Dictionary<string, Tensor>> InputsT = new(() => DitDir.Value.Load("inputs.safetensors"));
    private static readonly Lazy<Dictionary<string, Tensor>> CasesT = new(() => DitDir.Value.Load("cases.safetensors"));
    private static readonly Lazy<Dictionary<string, Tensor>> RunT = new(() => DitDir.Value.Load("run.safetensors"));
    private static readonly Lazy<FixtureDirectory> Fp64Dir = new(() => new FixtureDirectory(Path.Combine(Root.Dir, "dit_fp64")));
    private static readonly Lazy<Dictionary<string, Tensor>> Fp64T = new(() => Fp64Dir.Value.Load("run.safetensors"));

    public static FixtureDirectory Dit => DitDir.Value;
    public static FixtureDirectory Eos => EosDir.Value;

    /// <summary>tokens [L] int64, style [1,512], noise [blocks,1,10,64].</summary>
    public static Dictionary<string, Tensor> Inputs => InputsT.Value;

    /// <summary>Stage-level cases (a_*, b_*, n_*), see tools/dump_dit_reference.py.</summary>
    public static Dictionary<string, Tensor> Cases => CasesT.Value;

    /// <summary>The float64 run of the same 45-frame request (5 blocks): the exact-answer yardstick for the sampler's sensitivity.</summary>
    public static Dictionary<string, Tensor> Fp64 => Fp64T.Value;

    /// <summary>The whole sampler run: time_grid, block_start, block_states [blocks,steps,1,10,64], latent [1,T,64].</summary>
    public static Dictionary<string, Tensor> Run => RunT.Value;

    /// <summary>
    /// Compares with the fixture and prints the measured error (the tolerances come from these figures, see Tolerances.cs).
    /// <paramref name="relTol"/> is relative to the largest magnitude of the expected tensor.
    /// </summary>
    public static void AssertClose(ITestOutputHelper log, string what, Tensor got, Tensor expected, double relTol)
    {
        var r = Parity.Compare(got, expected);
        double scale = expected.abs().max().item<float>();
        double mean = (got.to(ScalarType.Float64) - expected.to(ScalarType.Float64)).abs().mean().item<double>();
        log.WriteLine($"{what}: max_abs={r.MaxAbs:E3} mean_abs={mean:E3} scale={scale:E3} rel={r.MaxAbs / scale:E3} cos={r.CosSim:F8}");
        Assert.Equal(expected.shape, got.shape);
        Assert.True(r.MaxAbs <= relTol * scale, $"{what}: max_abs {r.MaxAbs:E3} exceeds {relTol:E1} x scale {scale:E3}");
    }

    /// <summary>
    /// The deviation of <paramref name="got"/> from the fp32 reference must not exceed <paramref name="factor"/> times the deviation of that fp32
    /// reference from the float64 run of the same network (all max abs): the port may not be a worse approximation of the exact answer than the
    /// reference's own rounding. Also logs the deviation of the port from the float64 run.
    /// </summary>
    public static void AssertWithinSensitivity(ITestOutputHelper log, string what, Tensor got, Tensor reference32, Tensor exact64, double factor)
    {
        double Max(Tensor a, Tensor b) => (a.to(ScalarType.Float64) - b.to(ScalarType.Float64)).abs().max().item<double>();
        double dev = Max(got, reference32), floor = Max(reference32, exact64), vsExact = Max(got, exact64);
        log.WriteLine($"{what}: port-vs-fp32ref={dev:E3} fp32ref-vs-fp64={floor:E3} port-vs-fp64={vsExact:E3} ratio={dev / floor:F3}");
        Assert.True(dev <= factor * floor, $"{what}: deviation {dev:E3} from the fp32 reference exceeds {factor} x the reference's own sensitivity {floor:E3}");
    }

    /// <summary>Copies the checkpoint tensors <c>prefix + name</c> into each parameter/buffer of <paramref name="module"/>; a missing tensor throws.</summary>
    public static void LoadInto(nn.Module module, CheckpointIndex ckpt, string prefix)
    {
        using var _ = no_grad();
        foreach (var (name, target) in module.state_dict())
        {
            using var t = ckpt.Read(prefix + name, ScalarType.Float32);
            target.copy_(t);
        }
    }
}
