using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Text;

/// <summary>
/// Whole lyrics to the flat token sequence the sampler takes as its text input: the service's tag preparation (<c>normalize_lyrics</c> of
/// <c>server.py</c>) and the reference's <c>parse_lyrics</c>. A tag line gives its id and <c>[stop]</c>; a text line gives its phoneme ids and
/// <c>[stop]</c>; a <c>[start]</c>, <c>[stop]</c> pair is put first when the lyrics have no <c>[start]</c>.
/// </summary>
public sealed class LyricsTokenizer
{
    public const int Start = 500, Stop = 511;

    /// <summary>The structure tags the model knows and their ids.</summary>
    public static readonly IReadOnlyDictionary<string, int> TagIds = new Dictionary<string, int>
    {
        ["[start]"] = 500,
        ["[end]"] = 501,
        ["[intro]"] = 502,
        ["[verse]"] = 503,
        ["[chorus]"] = 504,
        ["[outro]"] = 505,
        ["[inst]"] = 506,
        ["[solo]"] = 507,
        ["[bridge]"] = 508,
        ["[hook]"] = 509,
        ["[break]"] = 510,
        ["[stop]"] = 511,
        ["[space]"] = 512,
    };

    private static readonly Regex TagLine = new(@"^\[(.*?)\]$", RegexOptions.CultureInvariant, RegexLimits.Match);

    private readonly EnglishG2p g2p;

    public LyricsTokenizer(EnglishG2p g2p) => this.g2p = g2p;

    /// <summary>One line of the lyrics with its 1-based number in the text it came from.</summary>
    private readonly record struct NumberedLine(int Number, string Text);

    /// <summary>
    /// The service's preparation: lines trimmed, blank ones dropped, known tags written in lower case and unknown bracketed lines dropped,
    /// <c>[start]</c> added first when missing. Lyrics without a text line are rejected.
    /// </summary>
    public static string Prepare(string lyrics) => string.Join("\n", PrepareLines(lyrics).Select(l => l.Text));

    private static List<NumberedLine> PrepareLines(string lyrics)
    {
        var lines = new List<NumberedLine>();
        bool hasText = false, hasStart = false;
        int number = 0;
        foreach (var raw in PyText.SplitLines(lyrics))
        {
            number++;
            var line = PyText.Strip(raw);
            if (line.Length == 0) continue;
            var m = TagLine.Match(line);
            if (m.Success)
            {
                var tag = PyText.Strip(m.Groups[1].Value).ToLowerInvariant();
                if (TagIds.ContainsKey($"[{tag}]"))
                {
                    lines.Add(new NumberedLine(number, $"[{tag}]"));
                    hasStart |= tag == "start";
                }
                continue;
            }
            lines.Add(new NumberedLine(number, line));
            hasText = true;
        }
        if (!hasText) throw new G2pException("the lyrics must have at least one line of text besides the structure tags: there is nothing to sing");
        if (!hasStart) lines.Insert(0, new NumberedLine(0, "[start]"));
        return lines;
    }

    /// <summary>The reference's <c>parse_lyrics</c>, flattened: the sampler's text tokens for lyrics that are already prepared (or raw ones, which it also accepts).</summary>
    public long[] Tokenize(string lyrics)
    {
        var lines = lyrics.Split('\n').Select((text, i) => new NumberedLine(i + 1, text));
        return TokenizeLines(lines);
    }

    /// <summary>Prepares the lyrics as the service does and tokenizes them, reporting errors against the line numbers of the original text.</summary>
    public long[] PrepareAndTokenize(string lyrics) => TokenizeLines(PrepareLines(lyrics));

    private long[] TokenizeLines(IEnumerable<NumberedLine> lines)
    {
        var tokens = new List<long>();
        bool getStart = false;
        foreach (var numbered in lines)
        {
            var line = PyText.Strip(numbered.Text);
            if (line.Length == 0) continue;
            if (Regex.IsMatch(line, @"^\[.*?\]$", RegexOptions.CultureInvariant, RegexLimits.Match))
            {
                if (!TagIds.TryGetValue(line.ToLowerInvariant(), out var id)) continue;
                if (id == Start) getStart = true;
                tokens.Add(id);
                tokens.Add(Stop);
            }
            else
            {
                try { tokens.AddRange(g2p.Encode(line).Select(i => (long)i)); }
                catch (G2pException e) { throw new LyricsLineException(numbered.Number, line, e); }
                tokens.Add(Stop);
            }
        }
        if (tokens.Count > 0 && !getStart) tokens.InsertRange(0, new long[] { Start, Stop });
        return tokens.ToArray();
    }

    /// <summary>The token sequence as the one-dimensional int64 tensor the sampler takes.</summary>
    public static Tensor ToTensor(long[] tokens) => tensor(tokens, dtype: ScalarType.Int64);
}
