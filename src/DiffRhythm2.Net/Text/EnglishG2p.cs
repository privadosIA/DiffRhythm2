using System.Text;
using System.Text.RegularExpressions;

namespace DiffRhythm2.Text;

/// <summary>The phonemes of one line and their vocabulary indices (before the +1 shift the lyrics tokenizer applies).</summary>
public sealed record G2pLine(string Phonemes, IReadOnlyList<int> Tokens)
{
    /// <summary>The ids the sampler sees: vocabulary index plus one, so 0 stays unused.</summary>
    public int[] Ids => Tokens.Select(t => t + 1).ToArray();
}

/// <summary>
/// The English line front end of the reference (<c>chn_eng_g2p</c> restricted to what it does for English): the line is segmented by character
/// class, the English text is normalized, phonemized, cleaned up with the reference's phoneme map and looked up in the vocabulary. A line with
/// Chinese characters is rejected, and so is a line with nothing to sing (the reference fails there too).
/// </summary>
public sealed class EnglishG2p
{
    private static string Str(params int[] codePoints) => string.Concat(codePoints.Select(char.ConvertFromUtf32));

    // The reference's special_map: whole phonemes (delimited by | or _ or the string ends) rewritten until none matches.
    private static readonly (string From, string To)[] SpecialMapRules =
    {
        (Str(0x74, 0x7c, 0x279), Str(0x74, 0x279)),
        (Str(0x64, 0x7c, 0x279), Str(0x64, 0x279)),
        (Str(0x74, 0x7c, 0x73), Str(0x74, 0x73)),
        (Str(0x64, 0x7c, 0x7a), Str(0x64, 0x7a)),
        (Str(0x26a, 0x7c, 0x279), Str(0x26a, 0x279)),
        (Str(0x250), Str(0x25a)),
        (Str(0x1d7b), Str(0x26a)),
        (Str(0x259, 0x6c), Str(0x6c)),
        (Str(0x78), Str(0x6b)),
        (Str(0x26c), Str(0x6c)),
        (Str(0x294), Str(0x74)),
        (Str(0x6e, 0x329), Str(0x6e)),
        (Str(0x6f, 0x2d0, 0x7c, 0x279), Str(0x6f, 0x2d0, 0x279)),
    };

    private static readonly Regex[] SpecialMapPatterns = SpecialMapRules
        .Select(r => new Regex("(^|[_|])" + r.From.Replace("|", "\\|") + "([_|]|$)", RegexOptions.CultureInvariant, RegexLimits.Match)).ToArray();

    // A phoneme string ending in one of these gets "|_" appended (the reference's english_to_ipa).
    private static readonly HashSet<int> FinalConsonantLike = new(
        new[] { 0x70, 0x207c, 0x2b0, 0x6d, 0x66, 0x74, 0x6e, 0x6c, 0x6b, 0x78, 0x283, 0x73, 0x60, 0x279, 0x61, 0x6f, 0x259, 0x25b, 0x26a, 0x65, 0x251, 0x28a, 0x14b, 0x69, 0x75, 0x265, 0x77, 0xe6, 0x6a, 0x2d0 });

    private readonly EspeakPhonemizer phonemizer;
    private readonly PhonemeVocabulary vocabulary;

    public EnglishG2p(IEspeakEngine engine, PhonemeVocabulary? vocabulary = null)
    {
        phonemizer = new EspeakPhonemizer(engine);
        this.vocabulary = vocabulary ?? PhonemeVocabulary.Default;
    }

    // ---- segmentation ---------------------------------------------------------------------------------------------------------------------

    private static string ClassOf(Rune r) =>
        r.Value is >= 0x4E00 and <= 0x9FA5 ? "zh" : r.Value is (>= 'A' and <= 'Z') or (>= 'a' and <= 'z') ? "en" : "other";

    /// <summary>The reference's <c>get_segment</c>: runs of one class; characters of class "other" join the segment they touch, and a leading run joins the next one.</summary>
    public static List<(string Text, string Lang)> Segment(string text)
    {
        var segments = new List<(string, string)>();
        bool started = false;
        var seg = new StringBuilder();
        string lang = "";
        foreach (var rune in text.EnumerateRunes())
        {
            string type = ClassOf(rune);
            if (!started) { seg.Append(rune.ToString()); lang = type; started = true; }
            else if (lang == "other")
            {
                seg.Append(rune.ToString());
                if (type != lang) lang = type;
            }
            else if (type == lang || type == "other") seg.Append(rune.ToString());
            else
            {
                segments.Add((seg.ToString(), lang));
                seg.Clear().Append(rune.ToString());
                lang = type;
            }
        }
        segments.Add((seg.ToString(), lang));
        return segments;
    }

    // ---- phoneme clean-up -------------------------------------------------------------------------------------------------------------------

    /// <summary>The reference's <c>special_map</c>, applied rule by rule until it no longer matches.</summary>
    public static string SpecialMap(string phonemes)
    {
        for (int i = 0; i < SpecialMapRules.Length; i++)
        {
            var pattern = SpecialMapPatterns[i];
            var replacement = "${1}" + SpecialMapRules[i].To.Replace("$", "$$") + "${2}";
            while (pattern.IsMatch(phonemes)) phonemes = pattern.Replace(phonemes, replacement);
        }
        return phonemes;
    }

    /// <summary>The reference's <c>english_to_ipa</c>: normalization, phonemizer, a word break after a final consonant-like phoneme, then the clean-up map.</summary>
    public string EnglishToIpa(string text)
    {
        var phonemes = phonemizer.Phonemize(EnglishNormalizer.Normalize(text));
        if (phonemes.Length == 0) throw new G2pException($"espeak produced no phonemes for \"{text}\"");
        var last = phonemes.EnumerateRunes().Last();
        if (FinalConsonantLike.Contains(last.Value)) phonemes += "|_";
        return SpecialMap(phonemes);
    }

    // ---- a line ---------------------------------------------------------------------------------------------------------------------------------

    /// <summary>The reference's <c>chn_eng_g2p</c> for a line without Chinese: the phoneme string (ending in <c>|</c>) and the vocabulary indices.</summary>
    public G2pLine Convert(string text)
    {
        var segments = Segment(text);
        if (segments.Any(s => s.Lang == "zh")) throw new ChineseNotSupportedException(text);
        if (segments.Any(s => s.Lang != "en")) throw new NoSungTextException(text);

        var allPhonemes = new StringBuilder();
        var allTokens = new List<int>();
        for (int index = 0; index < segments.Count; index++)
        {
            var phonemes = EnglishToIpa(segments[index].Text);
            allPhonemes.Append(phonemes).Append('|');
            allTokens.AddRange(vocabulary.Tokens(phonemes));
            // the word break that english_to_ipa adds after a final consonant is not wanted at the end of the line
            if (index == segments.Count - 1 && allPhonemes[^2] == '_')
            {
                allPhonemes.Length -= 2;
                allTokens.RemoveAt(allTokens.Count - 1);
            }
        }
        return new G2pLine(allPhonemes.ToString(), allTokens);
    }

    /// <summary>The ids of a line as the sampler takes them (vocabulary index plus one).</summary>
    public int[] Encode(string text) => Convert(text).Ids;
}
