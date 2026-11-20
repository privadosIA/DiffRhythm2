using System.Reflection;
using System.Text.Json;

namespace DiffRhythm2.Text;

/// <summary>
/// The reference's phoneme vocabulary (363 symbols: punctuation, <c>_</c> for a word break, IPA phonemes of English and of the other languages the
/// model was trained on), embedded in the library. A phoneme string is split at <c>|</c>; a symbol that is not in the vocabulary is skipped, as the
/// reference does.
/// </summary>
public sealed class PhonemeVocabulary
{
    private readonly Dictionary<string, int> ids;

    public PhonemeVocabulary(IReadOnlyDictionary<string, int> ids) => this.ids = new Dictionary<string, int>(ids, StringComparer.Ordinal);

    public int Count => ids.Count;

    public bool TryGetId(string phoneme, out int id) => ids.TryGetValue(phoneme, out id);

    /// <summary>The embedded vocabulary of the reference.</summary>
    public static PhonemeVocabulary Default { get; } = Load();

    public static PhonemeVocabulary Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return new PhonemeVocabulary(doc.RootElement.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal));
    }

    private static PhonemeVocabulary Load()
    {
        using var stream = typeof(PhonemeVocabulary).Assembly.GetManifestResourceStream("DiffRhythm2.Text.vocab.json")
            ?? throw new InvalidOperationException("the embedded phoneme vocabulary is missing from the assembly");
        using var reader = new StreamReader(stream);
        return Parse(reader.ReadToEnd());
    }

    /// <summary>The reference's <c>phoneme2token</c>: the vocabulary index of every known <c>|</c>-separated item, unknown items skipped.</summary>
    public List<int> Tokens(string phonemes)
    {
        var tokens = new List<int>();
        var first = phonemes.Split('\t')[0];
        foreach (var item in first.Split('|'))
            if (ids.TryGetValue(item, out var id)) tokens.Add(id);
        return tokens;
    }
}
