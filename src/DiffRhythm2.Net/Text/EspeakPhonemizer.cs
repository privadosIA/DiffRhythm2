using System.Text;
using System.Text.RegularExpressions;

namespace DiffRhythm2.Text;

/// <summary>
/// The reference's <c>TextTokenizer</c> for American English plus the part of <c>phonemizer</c>'s espeak backend that shapes the output:
/// the text is cleaned, split at the punctuation marks, each chunk is phonemized by espeak and tidied (language flags and stress removed,
/// phones joined with <c>|</c> and words with <c>|_|</c>), and the marks are restored as items of their own. The result for
/// <c>Hello world</c> is <c>h|ə|l|oʊ|_|w|ɜː|l|d</c> with espeak-ng 1.51. The only call into espeak is <see cref="IEspeakEngine"/>.
/// </summary>
public sealed class EspeakPhonemizer
{
    private const string PhoneSeparator = "|";
    private const string WordSeparator = "|_|";

    private static readonly string Space = $"[{PyText.SpaceClass}]";
    private static readonly Regex SpaceAroundMarks = new($@"{Space}*([,.?!;:'…]){Space}*".Replace(@"…", "…"), RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Whitespace = new($"{Space}+", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Underscores = new("_+", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex LanguageFlag = new(@"\(.+?\)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Stress = new("[ˈˌ'-]+", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex MarkItems = new("([,.?!;:'…])", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Bars = new(@"\|+", RegexOptions.CultureInvariant, RegexLimits.Match);

    private readonly IEspeakEngine engine;

    public EspeakPhonemizer(IEspeakEngine engine) => this.engine = engine;

    /// <summary>The text the phonemizer receives: the reference's cleaning, without calling espeak.</summary>
    public static string Clean(string text)
    {
        var line = ConvertChinesePunctuation(PyText.Strip(text));
        line = PyText.KeepWordSpaceAndMarks(line);
        line = SpaceAroundMarks.Replace(line, "$1");
        return Whitespace.Replace(line, " ");
    }

    private static string ConvertChinesePunctuation(string text)
    {
        // the order matters: "..." is converted last, after the three-dot variants
        return text
            .Replace("，", ",").Replace("。", ".").Replace("！", "!").Replace("？", "?").Replace("；", ";").Replace("：", ":")
            .Replace("、", ",").Replace("‘", "'").Replace("’", "'").Replace("⋯", "…").Replace("···", "…")
            .Replace("・・・", "…").Replace("...", "…");
    }

    /// <summary>Phonemizes one line. Returns phones separated by <c>|</c> and words by <c>|_|</c>, marks as their own items, no trailing bar.</summary>
    public string Phonemize(string text)
    {
        var (chunks, marks) = Punctuation.Preserve(new[] { Clean(text) });
        var phonemized = chunks.Select(chunk => TidyChunk(engine.TextToPhonemes(chunk))).ToList();
        var restored = Punctuation.Restore(phonemized, marks, WordSeparator, strip: true);

        var joined = string.Join("\n", restored);
        joined = MarkItems.Replace(joined, "|$1|");
        joined = Bars.Replace(joined, "|");
        return joined.TrimEnd('|');
    }

    /// <summary>The espeak backend's post-processing of one raw chunk (<c>_postprocess_line</c>, strip mode, flags removed, no stress).</summary>
    internal static string TidyChunk(string raw)
    {
        var line = PyText.Strip(raw).Replace("\n", " ").Replace("  ", " ");
        // espeak-ng can append extra separators at the end of a word (espeak-ng issue 694)
        line = Underscores.Replace(line, "_");
        line = line.Replace("_ ", " ");
        line = LanguageFlag.Replace(line, "");
        if (line.Length == 0) return "";

        var sb = new StringBuilder();
        foreach (var word in line.Split(' '))
        {
            var w = Stress.Replace(PyText.Strip(word), "");
            sb.Append(w.Replace("_", PhoneSeparator)).Append(WordSeparator);
        }
        sb.Length -= WordSeparator.Length;
        return sb.ToString();
    }
}
