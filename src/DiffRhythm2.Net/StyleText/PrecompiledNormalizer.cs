using System.Globalization;
using System.Text;

namespace DiffRhythm2.StyleText;

/// <summary>
/// The XLM-RoBERTa normalizer: sentencepiece's precompiled character map (NFKC-like rewrites, ligatures, full-width forms, the many Unicode spaces),
/// applied the way the reference's fast tokenizer applies it. The map is a blob: <c>[u32 size of the trie in bytes][double-array trie of u32
/// units][NUL-separated replacement strings]</c>. The text is cut into grapheme clusters; a cluster shorter than 6 UTF-8 bytes is looked up whole
/// (the first prefix of it found in the trie is used), otherwise, or when nothing matches, each of its characters is looked up on its own.
/// </summary>
public sealed class PrecompiledNormalizer
{
    private readonly uint[] units;
    private readonly byte[] replacements;

    public PrecompiledNormalizer(byte[] charsMap)
    {
        if (charsMap.Length < 4) throw new InvalidDataException("the precompiled character map is truncated");
        uint trieBytes = BitConverter.ToUInt32(charsMap, 0);
        if (trieBytes % 4 != 0 || 4 + (long)trieBytes > charsMap.Length) throw new InvalidDataException($"the precompiled character map declares a trie of {trieBytes} bytes in a blob of {charsMap.Length}");
        units = new uint[trieBytes / 4];
        Buffer.BlockCopy(charsMap, 4, units, 0, (int)trieBytes);
        replacements = charsMap[(4 + (int)trieBytes)..];
    }

    // Unit layout of the double array (darts-clone, as sentencepiece uses it).
    private static bool HasLeaf(uint u) => ((u >> 8) & 1) == 1;
    private static uint Value(uint u) => u & 0x7FFFFFFF;
    private static uint Label(uint u) => u & ((1u << 31) | 0xFF);
    private static uint Offset(uint u) => (u >> 10) << (int)((u & (1u << 9)) >> 6);

    /// <summary>The values of the keys that are prefixes of <paramref name="key"/>, shortest first.</summary>
    private List<uint> CommonPrefixSearch(ReadOnlySpan<byte> key)
    {
        var results = new List<uint>();
        uint nodePos = 0;
        uint unit = units[nodePos];
        nodePos ^= Offset(unit);
        for (int i = 0; i < key.Length; i++)
        {
            nodePos ^= key[i];
            if (nodePos >= units.Length) break;
            unit = units[nodePos];
            if (Label(unit) != key[i]) break;
            nodePos ^= Offset(unit);
            if (HasLeaf(unit))
            {
                if (nodePos >= units.Length) break;
                results.Add(Value(units[nodePos]));
            }
        }
        return results;
    }

    private string? Transform(ReadOnlySpan<byte> chunk)
    {
        var results = CommonPrefixSearch(chunk);
        if (results.Count == 0) return null;
        int start = (int)results[PreferLongest ? results.Count - 1 : 0];
        int end = start;
        while (end < replacements.Length && replacements[end] != 0) end++;
        return Encoding.UTF8.GetString(replacements, start, end - start);
    }

    /// <summary>Which of several matching prefixes is used. The reference takes the first (shortest); kept switchable for the experiments in the tests.</summary>
    internal bool PreferLongest { get; set; }

    public string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        var e = StringInfo.GetTextElementEnumerator(text);
        while (e.MoveNext())
        {
            var grapheme = (string)e.Current;
            var bytes = Encoding.UTF8.GetBytes(grapheme);
            if (bytes.Length < 6 && Transform(bytes) is { } whole)
            {
                sb.Append(whole);
                continue;
            }
            foreach (var rune in grapheme.EnumerateRunes())
            {
                var part = rune.ToString();
                sb.Append(Transform(Encoding.UTF8.GetBytes(part)) ?? part);
            }
        }
        return sb.ToString();
    }
}
