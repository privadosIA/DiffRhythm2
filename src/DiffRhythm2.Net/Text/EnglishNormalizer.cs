using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace DiffRhythm2.Text;

/// <summary>
/// The text normalization of the reference's <c>english.py</c> that runs before espeak: abbreviations are expanded, then numbers are spelled out
/// (thousands commas, pounds, dollars, fractions, decimals, percentages, ordinals, plain integers and years), in that order. Spacing is kept as
/// the reference leaves it (extra spaces around the spelled-out numbers), because the cleaning step that follows collapses it.
/// </summary>
public static class EnglishNormalizer
{
    // The reference writes \bword\b with re.IGNORECASE; Python's \b is a boundary against \w, so the boundaries are spelled with its \w.
    private static readonly (Regex Pattern, string Replacement)[] Abbreviations = new[]
    {
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"),
        ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"), ("etc", "et cetera"), ("btw", "by the way"),
    }.Select(a => (new Regex($"(?<![{PyText.WordClass}]){a.Item1}(?![{PyText.WordClass}])", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, RegexLimits.Match), a.Item2)).ToArray();

    private static readonly Regex CommaNumber = new("([0-9][0-9,]+[0-9])", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex DecimalNumber = new(@"([0-9]+\.[0-9]+)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex PercentNumber = new(@"([0-9.,]*[0-9]+%)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Pounds = new("£([0-9,]*[0-9]+)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Dollars = new(@"\$([0-9.,]*[0-9]+)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Fraction = new("([0-9]+)/([0-9]+)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Ordinal = new("[0-9]+(st|nd|rd|th)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Number = new("[0-9]+", RegexOptions.CultureInvariant, RegexLimits.Match);

    /// <summary>The reference's <c>_english_to_ipa</c> (despite its name, only text normalization): abbreviations, then numbers.</summary>
    public static string Normalize(string text) => NormalizeNumbers(ExpandAbbreviations(text));

    public static string ExpandAbbreviations(string text)
    {
        foreach (var (pattern, replacement) in Abbreviations) text = pattern.Replace(text, replacement.Replace("$", "$$"));
        return text;
    }

    public static string NormalizeNumbers(string text)
    {
        text = CommaNumber.Replace(text, m => m.Groups[1].Value.Replace(",", ""));
        text = Pounds.Replace(text, m => m.Groups[1].Value + " pounds");
        text = Dollars.Replace(text, m => ExpandDollars(m.Groups[1].Value));
        text = Fraction.Replace(text, m => ExpandFraction(m.Groups[1].Value, m.Groups[2].Value));
        text = DecimalNumber.Replace(text, m => m.Groups[1].Value.Replace(".", " point "));
        text = PercentNumber.Replace(text, m => m.Groups[1].Value.Replace("%", " percent "));
        text = Ordinal.Replace(text, m => " " + NumberWords.ToWords(m.Value) + " ");
        text = Number.Replace(text, m => ExpandNumber(m.Value));
        return text;
    }

    private static BigInteger ParseInt(string s)
    {
        // Python's int() on text such as ",5" raises ValueError, which aborts the song in the reference; so does this.
        if (s.Length == 0 || !s.All(c => c is >= '0' and <= '9')) throw new G2pException($"cannot read \"{s}\" as a whole number");
        return BigInteger.Parse(s, CultureInfo.InvariantCulture);
    }

    private static string ExpandDollars(string match)
    {
        var parts = match.Split('.');
        if (parts.Length > 2) return " " + match + " dollars ";
        var dollars = parts[0].Length > 0 ? ParseInt(parts[0]) : BigInteger.Zero;
        var cents = parts.Length > 1 && parts[1].Length > 0 ? ParseInt(parts[1]) : BigInteger.Zero;
        if (!dollars.IsZero && !cents.IsZero)
            return $" {dollars} {(dollars.IsOne ? "dollar" : "dollars")}, {cents} {(cents.IsOne ? "cent" : "cents")} ";
        if (!dollars.IsZero) return $" {dollars} {(dollars.IsOne ? "dollar" : "dollars")} ";
        if (!cents.IsZero) return $" {cents} {(cents.IsOne ? "cent" : "cents")} ";
        return " zero dollars ";
    }

    private static string ExpandFraction(string numeratorText, string denominatorText)
    {
        var numerator = ParseInt(numeratorText);
        var denominator = ParseInt(denominatorText);
        return FractionToWords(numerator, denominator);
    }

    /// <summary>The reference's <c>fraction_to_words</c>.</summary>
    public static string FractionToWords(BigInteger numerator, BigInteger denominator)
    {
        if (numerator.IsOne && denominator == 2) return " one half ";
        if (numerator.IsOne && denominator == 4) return " one quarter ";
        if (denominator == 2) return " " + NumberWords.ToWords(numerator) + " halves ";
        if (denominator == 4) return " " + NumberWords.ToWords(numerator) + " quarters ";
        return " " + NumberWords.ToWords(numerator) + " " + NumberWords.Ordinal(NumberWords.ToWords(denominator)) + " ";
    }

    private static string ExpandNumber(string digits)
    {
        var num = ParseInt(digits);
        if (num > 1000 && num < 3000)
        {
            if (num == 2000) return " two thousand ";
            if (num > 2000 && num < 2010) return " two thousand " + NumberWords.ToWords(num % 100) + " ";
            if (num % 100 == 0) return " " + NumberWords.ToWords(num / 100) + " hundred ";
            return " " + NumberWords.ToWords(num, andWord: "", zero: "oh", group: 2).Replace(", ", " ") + " ";
        }
        return " " + NumberWords.ToWords(num, andWord: "") + " ";
    }
}
