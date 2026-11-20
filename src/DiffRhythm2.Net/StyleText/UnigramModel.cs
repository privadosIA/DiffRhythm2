using System.Text;

namespace DiffRhythm2.StyleText;

/// <summary>
/// The Unigram segmentation of the reference's fast tokenizer for one word: the pieces of the vocabulary whose scores add up to the highest total
/// (Viterbi over the positions of the word). A character no piece covers becomes the unknown token, scored <c>minimum score - 10</c>, and
/// consecutive unknown characters are fused into one. Candidates at a position are tried from the shortest piece to the longest and replace a
/// path only if strictly better, so among equal totals the path that was found first wins, as in the reference.
/// </summary>
public sealed class UnigramModel
{
    private const double UnknownPenalty = 10.0;

    private readonly Dictionary<string, int> ids;
    private readonly double[] scores;
    private readonly int maxPieceChars;
    private readonly double unknownScore;

    public int UnknownId { get; }
    public int Count => scores.Length;

    public UnigramModel(IReadOnlyList<string> pieces, IReadOnlyList<double> scores, int unknownId)
    {
        this.scores = scores.ToArray();
        UnknownId = unknownId;
        ids = new Dictionary<string, int>(pieces.Count, StringComparer.Ordinal);
        for (int i = 0; i < pieces.Count; i++) ids[pieces[i]] = i;   // a repeated piece keeps its last id, like the reference's map
        maxPieceChars = pieces.Max(p => p.Length);
        unknownScore = this.scores.Min() - UnknownPenalty;
    }

    public bool TryGetId(string piece, out int id) => ids.TryGetValue(piece, out id);

    /// <summary>The ids of the best segmentation of <paramref name="word"/> (no whitespace in it).</summary>
    public List<int> Encode(string word)
    {
        int size = word.Length;
        var bestScore = new double[size + 1];
        var startsAt = new int[size + 1];
        var pieceId = new int[size + 1];
        Array.Fill(startsAt, -1);

        for (int start = 0; start < size;)
        {
            double scoreHere = bestScore[start];
            int charLength = char.IsHighSurrogate(word[start]) && start + 1 < size && char.IsLowSurrogate(word[start + 1]) ? 2 : 1;
            bool hasSingle = false;
            for (int length = 1; length <= maxPieceChars && start + length <= size; length++)
            {
                if (length == 1 && char.IsLowSurrogate(word[start])) continue;
                if (length < charLength) continue;
                // a piece never ends in the middle of a surrogate pair
                if (start + length < size && char.IsLowSurrogate(word[start + length]) && char.IsHighSurrogate(word[start + length - 1])) continue;
                if (!ids.TryGetValue(word.Substring(start, length), out int id)) continue;
                int end = start + length;
                double candidate = scores[id] + scoreHere;
                if (startsAt[end] < 0 || candidate > bestScore[end])
                {
                    bestScore[end] = candidate;
                    startsAt[end] = start;
                    pieceId[end] = id;
                }
                if (length == charLength) hasSingle = true;
            }
            if (!hasSingle)
            {
                int end = start + charLength;
                double candidate = unknownScore + scoreHere;
                if (startsAt[end] < 0 || candidate > bestScore[end])
                {
                    bestScore[end] = candidate;
                    startsAt[end] = start;
                    pieceId[end] = UnknownId;
                }
            }
            start += charLength;
        }

        var result = new List<int>();
        bool lastWasUnknown = false;
        for (int end = size; end > 0; end = startsAt[end])
        {
            int id = pieceId[end];
            if (id == UnknownId)
            {
                if (!lastWasUnknown) result.Add(id);   // consecutive unknowns are fused into one
                lastWasUnknown = true;
            }
            else
            {
                result.Add(id);
                lastWasUnknown = false;
            }
        }
        result.Reverse();
        return result;
    }
}
