using System.Text.Json;
using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>
/// The MuQ-MuLan audio tower golden run (tests/fixtures/mulan_audio, verified against its MANIFEST.md): test signals, the reference's clips, the front end,
/// the stage outputs of the first clip of <c>s25</c> and the embeddings; and where the converted audio checkpoint lives.
/// </summary>
public static class MulanAudioReference
{
    public const string WeightsFile = "mulan_audio.safetensors";

    private static readonly Lazy<FixtureDirectory> Dir = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "mulan_audio")));
    private static readonly Dictionary<string, Dictionary<string, Tensor>> Loaded = new();
    private static JsonDocument? meta;

    public static FixtureDirectory Directory => Dir.Value;

    private static Dictionary<string, Tensor> Get(string file)
    {
        lock (Loaded)
        {
            if (!Loaded.TryGetValue(file, out var d)) Loaded[file] = d = Directory.Load(file);
            return d;
        }
    }

    /// <summary>s4 (4 s), s25 (25 s), noise10 (10 s), float32 mono at 24 kHz.</summary>
    public static Dictionary<string, Tensor> Signals => Get("signals.safetensors");

    /// <summary>The reference's own clips where they are not a plain slice: <c>s4_0</c> (192,000 samples) and <c>s25_2</c>.</summary>
    public static Dictionary<string, Tensor> Clips => Get("clips.safetensors");

    /// <summary><c>filter_bank</c> [1025, 128]; per signal <c>{name}_db</c> [128, frames + 1] and <c>{name}_norm</c> [128, frames] of its first clip.</summary>
    public static Dictionary<string, Tensor> Frontend => Get("frontend.safetensors");

    /// <summary>Stages of the first clip of s25: conv, layer1/5/6/11/12, last, proj, mean, latent, emb.</summary>
    public static Dictionary<string, Tensor> Stages => Get("stages.safetensors");

    /// <summary>Clip embeddings <c>{name}_clip{i}</c>, signal embeddings <c>{name}</c>, <c>text_one</c>, <c>text_batch</c>, <c>sim_one</c>, <c>sim_batch</c>, <c>audio_audio</c>.</summary>
    public static Dictionary<string, Tensor> Embeddings => Get("embeddings.safetensors");

    public static JsonElement Meta
    {
        get
        {
            lock (Loaded) return (meta ??= JsonDocument.Parse(Directory.ReadAllText("meta.json"))).RootElement;
        }
    }

    public static float[] Signal(string name) => Signals[name].data<float>().ToArray();

    public static string? WeightsPath => Checkpoint.Dir is { } d && File.Exists(Path.Combine(d, WeightsFile)) ? Path.Combine(d, WeightsFile) : null;

    /// <summary>
    /// Why the golden run of the audio tower is not here, or null when it is. The run holds activations of MuQ-MuLan, which is non-commercial (CC-BY-NC-4.0),
    /// so the public copy of the repository does not carry it and the tests that read it are skipped there.
    /// </summary>
    public static string? FixturesMissingReason =>
        File.Exists(Path.Combine(FixtureDirectory.Find().Dir, "mulan_audio", "MANIFEST.md")) ? null
            : "tests/fixtures/mulan_audio is not in this checkout (the public copy leaves out the MuQ-MuLan golden run: the model is CC-BY-NC-4.0)";
}

/// <summary>[Fact] that reads the MuQ-MuLan audio tower golden run (tests/fixtures/mulan_audio).</summary>
public sealed class RequiresMulanFixturesFactAttribute : FactAttribute
{
    public RequiresMulanFixturesFactAttribute() => Skip = MulanAudioReference.FixturesMissingReason;
}

/// <summary>[Theory] that reads the MuQ-MuLan audio tower golden run (tests/fixtures/mulan_audio).</summary>
public sealed class RequiresMulanFixturesTheoryAttribute : TheoryAttribute
{
    public RequiresMulanFixturesTheoryAttribute() => Skip = MulanAudioReference.FixturesMissingReason;
}

/// <summary>[Fact] that needs the converted audio tower in the checkpoint directory (<c>diffrhythm2 convert --audio</c>, or tools/convert_mulan_audio.py).</summary>
public sealed class RequiresMulanAudioAttribute : FactAttribute
{
    public RequiresMulanAudioAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason
            ?? MulanAudioReference.FixturesMissingReason
            ?? (MulanAudioReference.WeightsPath is null ? $"{MulanAudioReference.WeightsFile} not found in the checkpoint directory (run `diffrhythm2 convert --audio`)" : null);
}
