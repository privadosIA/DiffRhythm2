using System.Text.Json;
using MusicGen.Core.Testing;

namespace DiffRhythm2.Tests;

/// <summary>The English g2p golden run (tests/fixtures/g2p, verified against its MANIFEST.md): per-layer results for the corpus, the recorded espeak transcripts, the number table and the lyrics cases.</summary>
public static class G2pReference
{
    private static readonly Lazy<FixtureDirectory> Dir = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "g2p")));
    private static readonly Dictionary<string, JsonDocument> Cache = new();

    public static FixtureDirectory Directory => Dir.Value;

    public static JsonElement Json(string file)
    {
        lock (Cache)
        {
            if (!Cache.TryGetValue(file, out var doc)) Cache[file] = doc = JsonDocument.Parse(Directory.ReadAllText(file));
            return doc.RootElement;
        }
    }

    public static JsonElement Lines => Json("lines.json");
    public static JsonElement Numbers => Json("numbers.json");
    public static JsonElement Lyrics => Json("lyrics.json");
    public static JsonElement Meta => Json("meta.json");

    /// <summary>The raw espeak answers recorded for every chunk of text the reference phonemized.</summary>
    public static Dictionary<string, string> Transcripts =>
        Json("transcripts.json").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);

    /// <summary>The corpus lines the reference tokenized without error.</summary>
    public static IEnumerable<JsonElement> GoodLines => Lines.EnumerateArray().Where(l => !l.TryGetProperty("error", out _) && !l.GetProperty("has_chinese").GetBoolean());

    public static int[] Ints(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt32()).ToArray();
}
