using System.Text.Json;
using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The MuQ-MuLan text tower golden run (tests/fixtures/mulan_text, verified against its MANIFEST.md) and where the converted files live.</summary>
public static class MulanTextReference
{
    public const string WeightsFile = "mulan_text.safetensors";
    public const string TokenizerFile = "mulan_text_tokenizer.json";

    private static readonly Lazy<FixtureDirectory> Dir = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "mulan_text")));
    private static readonly Dictionary<string, JsonDocument> Json = new();
    private static Dictionary<string, Tensor>? tower;

    public static FixtureDirectory Directory => Dir.Value;

    public static JsonElement Doc(string file)
    {
        lock (Json)
        {
            if (!Json.TryGetValue(file, out var doc)) Json[file] = doc = JsonDocument.Parse(Directory.ReadAllText(file));
            return doc.RootElement;
        }
    }

    public static JsonElement Tokens => Doc("tokens.json");
    public static JsonElement Normalization => Doc("normalization.json");
    public static JsonElement Meta => Doc("meta.json");

    /// <summary>Stage outputs per prompt: p{i}_ids, p{i}_embeddings, p{i}_l0, p{i}_enc, p{i}_t0, p{i}_tr, p{i}_mean, p{i}_emb.</summary>
    public static Dictionary<string, Tensor> Tower => tower ??= Directory.Load("tower.safetensors");

    public static string[] Prompts => Meta.GetProperty("prompts").EnumerateArray().Select(p => p.GetString()!).ToArray();

    public static string? TokenizerPath => Checkpoint.Dir is { } d && File.Exists(Path.Combine(d, TokenizerFile)) ? Path.Combine(d, TokenizerFile) : null;

    public static string? WeightsPath => Checkpoint.Dir is { } d && File.Exists(Path.Combine(d, WeightsFile)) ? Path.Combine(d, WeightsFile) : null;

    /// <summary>
    /// Why the stage outputs of the text tower are not here, or null when they are. They are activations of MuQ-MuLan, which is non-commercial (CC-BY-NC-4.0),
    /// so the public copy of the repository does not carry them and the tests that read them are skipped there.
    /// </summary>
    public static string? TowerMissingReason =>
        File.Exists(Path.Combine(Directory.Dir, "tower.safetensors")) ? null
            : "tests/fixtures/mulan_text/tower.safetensors is not in this checkout (the public copy leaves out the MuQ-MuLan golden run: the model is CC-BY-NC-4.0)";
}

/// <summary>[Fact] that needs <c>mulan_text_tokenizer.json</c> in the checkpoint directory (see tools/convert_mulan_text.py).</summary>
public sealed class RequiresMulanTokenizerAttribute : FactAttribute
{
    public RequiresMulanTokenizerAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason ?? (MulanTextReference.TokenizerPath is null ? $"{MulanTextReference.TokenizerFile} not found in the checkpoint directory (run tools/convert_mulan_text.py)" : null);
}

/// <summary>[Fact] that needs the converted tower and its tokenizer in the checkpoint directory.</summary>
public sealed class RequiresMulanTextAttribute : FactAttribute
{
    public RequiresMulanTextAttribute() =>
        Skip = Checkpoint.Env.NoCheckpointReason
            ?? MulanTextReference.TowerMissingReason
            ?? (MulanTextReference.WeightsPath is null || MulanTextReference.TokenizerPath is null
                ? $"{MulanTextReference.WeightsFile} / {MulanTextReference.TokenizerFile} not found in the checkpoint directory (run tools/convert_mulan_text.py)" : null);
}
