using System.Numerics;
using System.Text.RegularExpressions;

namespace DiffRhythm2.Text;

/// <summary>
/// A port of the part of the <c>inflect</c> library that the reference uses (<c>number_to_words</c> for integers, in the plain and the
/// group-of-two modes, with ordinal strings, and <c>ordinal</c> of spoken numbers). The reference's <c>inflect</c> is the copy vendored in
/// setuptools; this follows its algorithm step by step, including its regex-driven passes, because the exact commas, hyphens and spaces it
/// produces reach espeak and change the phonemes. The tests compare it with a table dumped from that copy.
/// </summary>
public static class NumberWords
{
    private static readonly string[] Unit = { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine" };
    private static readonly string[] Teen = { "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
    private static readonly string[] Ten = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
    private static readonly string[] Mill =
    {
        " ", " thousand", " million", " billion", " trillion", " quadrillion", " quintillion", " sextillion", " septillion", " octillion", " nonillion", " decillion",
    };

    private static readonly Dictionary<string, string> OrdinalWord = new()
    {
        ["ty"] = "tieth",
        ["one"] = "first",
        ["two"] = "second",
        ["three"] = "third",
        ["five"] = "fifth",
        ["eight"] = "eighth",
        ["nine"] = "ninth",
        ["twelve"] = "twelfth",
    };

    private static readonly Regex OrdinalSuffix = new(@"(ty|one|two|three|five|eight|nine|twelve)\z", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex DigitGroup = new("([0-9])", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex TwoDigits = new("([0-9])([0-9])", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex ThreeDigitsWord = new("([0-9])([0-9])([0-9])(?=[^0-9]*\\z)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex TwoDigitsWord = new("([0-9])([0-9])(?=[^0-9]*\\z)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex OneDigitWord = new("([0-9])(?=[^0-9]*\\z)", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex NonDigit = new("[^0-9]", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex WhitespacesComma = new(@"[" + PyText.SpaceClass + @"]+,", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex CommaWord = new(@", ([^" + PyText.SpaceClass + @"]+)[" + PyText.SpaceClass + @"]+\z", RegexOptions.CultureInvariant, RegexLimits.Match);
    private static readonly Regex Whitespaces = new(@"[" + PyText.SpaceClass + @"]+", RegexOptions.CultureInvariant, RegexLimits.Match);

    /// <summary>The most digits <c>inflect</c> can spell (up to "decillion"); a longer number raises an error there.</summary>
    public const int MaxDigits = 36;

    public static string ToWords(BigInteger n, string andWord = "and", string zero = "zero", int group = 0) => ToWords(n.ToString(), andWord, zero, group);

    /// <summary>
    /// <c>inflect.engine().number_to_words(num, andword=..., zero=..., group=...)</c> for a string of digits, optionally followed by an ordinal
    /// suffix (<c>st</c>, <c>nd</c>, <c>rd</c> or <c>th</c>), which makes the last word ordinal.
    /// </summary>
    public static string ToWords(string num, string andWord = "and", string zero = "zero", int group = 0)
    {
        string digits = num;
        bool ordinal = false;
        if (num is "th" or "st" or "nd" or "rd") digits = "";   // the reference turns a bare suffix into the word for zero
        else if (num.Length >= 2 && num[^2..] is "th" or "st" or "nd" or "rd") { ordinal = true; digits = num[..^2]; }
        if (digits.Any(c => c is < '0' or > '9')) throw new ArgumentException($"only digit strings are supported, got \"{num}\"");
        if (digits.TrimStart('0').Length > MaxDigits) throw new NumberOutOfRangeException(digits);

        string chunk = digits == "" ? "0" : digits;
        chunk = EnWord(chunk, group, andWord, zero);
        if (chunk.EndsWith(", ", StringComparison.Ordinal)) chunk = chunk[..^2];
        chunk = WhitespacesComma.Replace(chunk, ",");
        if (group == 0) chunk = CommaWord.Replace(chunk, m => $" {andWord} {m.Groups[1].Value}");
        chunk = Whitespaces.Replace(chunk, " ");
        chunk = PyText.Strip(chunk);

        var pieces = chunk.Split(", ");
        if (ordinal) pieces[^1] = SubOrdinal(pieces[^1]);
        return string.Join(", ", pieces);
    }

    /// <summary><c>inflect.engine().ordinal(words)</c> for spoken numbers ("eight" gives "eighth").</summary>
    public static string Ordinal(string words) => SubOrdinal(words);

    private static string SubOrdinal(string value)
    {
        var replaced = OrdinalSuffix.Replace(value, m => OrdinalWord[m.Groups[1].Value]);
        return replaced == value ? replaced + "th" : replaced;
    }

    private static string MillFn(int index)
    {
        if (index > Mill.Length - 1) throw new NumberOutOfRangeException("(too many digits)");
        return Mill[index];
    }

    private static string TenFn(int tens, int units, int mindex = 0)
    {
        if (tens != 1)
        {
            string hyphen = tens != 0 && units != 0 ? "-" : "";
            return $"{Ten[tens]}{hyphen}{Unit[units]}{MillFn(mindex)}";
        }
        if (mindex > Mill.Length - 1) throw new NumberOutOfRangeException("(too many digits)");
        return $"{Teen[units]}{Mill[mindex]}";
    }

    private static string HundFn(int hundreds, int tens, int units, int mindex, string andWord)
    {
        if (hundreds != 0)
        {
            string and = tens != 0 || units != 0 ? $" {andWord} " : "";
            return $"{Unit[hundreds]} hundred{and}{TenFn(tens, units)}{MillFn(mindex)}, ";
        }
        if (tens != 0 || units != 0) return $"{TenFn(tens, units)}{MillFn(mindex)}, ";
        return "";
    }

    private static string EnWord(string num, int group, string andWord, string zero)
    {
        const string one = "one";
        static int D(Group g) => int.Parse(g.Value);

        switch (group)
        {
            case 2:
                {
                    string text = TwoDigits.Replace(num, m =>
                    {
                        int tens = D(m.Groups[1]), units = D(m.Groups[2]);
                        if (tens != 0) return $"{TenFn(tens, units)}, ";
                        if (units != 0) return $" {zero} {Unit[units]}, ";
                        return $" {zero} {zero}, ";
                    });
                    return DigitGroup.Replace(text, m => D(m.Groups[1]) != 0 ? $"{Unit[D(m.Groups[1])]}, " : $" {zero}, ", 1);
                }
            case 0:
                {
                    string trimmed = num.TrimStart('0');
                    if (trimmed == "") return zero;
                    if (trimmed == "1") return one;
                    string text = trimmed;
                    int millCount = 0;
                    while (ThreeDigitsWord.IsMatch(text))
                    {
                        text = ThreeDigitsWord.Replace(text, m =>
                        {
                            string r = HundFn(D(m.Groups[1]), D(m.Groups[2]), D(m.Groups[3]), millCount, andWord);
                            millCount++;
                            return r;
                        }, 1);
                    }
                    text = TwoDigitsWord.Replace(text, m => $"{TenFn(D(m.Groups[1]), D(m.Groups[2]), millCount)}, ", 1);
                    text = OneDigitWord.Replace(text, m => $"{Unit[D(m.Groups[1])]}{MillFn(millCount)}, ", 1);
                    return text;
                }
            default:
                throw new NotSupportedException($"digit grouping {group} is not used by the reference");
        }
    }

    /// <summary>The "nonnumeric" check shared by callers: true when the string is only ASCII digits.</summary>
    internal static bool IsDigits(string s) => s.Length > 0 && s.All(c => c is >= '0' and <= '9');
}
