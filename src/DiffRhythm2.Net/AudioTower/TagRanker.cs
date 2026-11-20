using DiffRhythm2.StyleText;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.AudioTower;

/// <summary>A text with its similarity to an audio embedding.</summary>
/// <param name="Text">The text that was embedded.</param>
/// <param name="Score">Dot product of its embedding with the audio embedding.</param>
public sealed record RankedText(string Text, float Score);

/// <summary>Similarities in MuQ-MuLan's shared space, and the ranking of texts for an audio embedding.</summary>
public static class TagRanker
{
    /// <summary>The similarity of two embeddings: their dot product (the reference's <c>calc_similarity</c>). The embeddings need the same length.</summary>
    public static float Dot(IReadOnlyList<float> a, IReadOnlyList<float> b)
    {
        if (a.Count != b.Count) throw new ArgumentException($"the embeddings have {a.Count} and {b.Count} values");
        double sum = 0;
        for (int i = 0; i < a.Count; i++) sum += (double)a[i] * b[i];
        return (float)sum;
    }

    /// <summary>
    /// The texts ordered by similarity to <paramref name="audio"/>, highest first, with their scores, optionally only the first <paramref name="top"/>. Every text is
    /// embedded alone with <paramref name="textEncoder"/> (the reference averages over padded positions, so a batch would depend on its neighbours); equal scores keep the
    /// order of the list. A text that cannot be embedded fails the ranking, naming it.
    /// </summary>
    public static IReadOnlyList<RankedText> Rank(IReadOnlyList<float> audio, IReadOnlyList<string> texts, StyleEncoder textEncoder, int? top = null) =>
        Rank(audio, texts, text =>
        {
            using var e = textEncoder.Embed(text);
            return e.cpu().data<float>().ToArray();
        }, top);

    /// <summary>The same ranking with the embedding of a text given by <paramref name="embed"/> (which may throw <see cref="ArgumentException"/> for a text it cannot embed).</summary>
    public static IReadOnlyList<RankedText> Rank(IReadOnlyList<float> audio, IReadOnlyList<string> texts, Func<string, float[]> embed, int? top = null)
    {
        if (top is < 1) throw new ArgumentOutOfRangeException(nameof(top), "top must be at least 1");
        var scored = new List<RankedText>(texts.Count);
        foreach (var text in texts)
        {
            float[] embedding;
            try { embedding = embed(text); }
            catch (ArgumentException ex)
            {
                var shown = text.Length > 60 ? text[..60] + "..." : text;
                throw new ArgumentException($"the text '{shown}' cannot be embedded: {ex.Message}", nameof(texts), ex);
            }
            scored.Add(new RankedText(text, Dot(audio, embedding)));
        }
        var ordered = scored.OrderByDescending(r => r.Score);                   // a stable sort: ties keep the order of the list
        return (top is { } n ? ordered.Take(n) : ordered).ToList();
    }
}
