using System.Globalization;
using System.Text;

namespace DiffRhythm2.Text;

/// <summary>
/// The character classes of Python's <c>str</c> and <c>re</c> (the reference's), which differ from .NET's: <c>\w</c> is letters, the numeric
/// categories and underscore (but not combining marks), and whitespace is <c>str.isspace()</c>. Everything here works on code points, so a
/// surrogate pair counts once. The whitespace set is spelled with numbers because several of its members are invisible or line-breaking
/// characters that must not appear literally in source code.
/// </summary>
public static class PyText
{
    // Python's whitespace: 09-0D, 1C-1F, 20, 85, A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000 (inclusive ranges).
    private static readonly (int From, int To)[] SpaceRanges =
    {
        (0x09, 0x0D), (0x1C, 0x1F), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000),
    };

    /// <summary>The members of a regex character class that match Python's <c>\s</c> (all in the BMP), as regex escapes.</summary>
    public static readonly string SpaceClass = string.Concat(SpaceRanges.Select(r => r.From == r.To ? $"\\u{r.From:x4}" : $"\\u{r.From:x4}-\\u{r.To:x4}"));

    /// <summary>The members of a regex character class that match Python's <c>\w</c> for BMP characters.</summary>
    public const string WordClass = @"\p{L}\p{Nd}\p{Nl}\p{No}_";

    public static bool IsSpace(int c)
    {
        foreach (var (from, to) in SpaceRanges)
            if (c >= from && c <= to) return true;
        return false;
    }

    public static bool IsWordChar(Rune r)
    {
        if (r.Value == '_') return true;
        return Rune.GetUnicodeCategory(r) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter
            or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter or UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber
            or UnicodeCategory.OtherNumber;
    }

    /// <summary><c>str.strip()</c>.</summary>
    public static string Strip(string s)
    {
        int start = 0, end = s.Length;
        while (start < end && IsSpace(s[start])) start++;
        while (end > start && IsSpace(s[end - 1])) end--;
        return s[start..end];
    }

    // The characters at which Python's str.splitlines() breaks: LF, CR, VT, FF, 1C-1E, 85, 2028, 2029.
    private static bool IsLineBreak(int c) => c is 0x0A or 0x0D or 0x0B or 0x0C or 0x1C or 0x1D or 0x1E or 0x85 or 0x2028 or 0x2029;

    /// <summary><c>str.splitlines()</c> (without keeping the line ends): Python breaks lines at more characters than <c>\n</c>.</summary>
    public static List<string> SplitLines(string s)
    {
        var lines = new List<string>();
        int start = 0, i = 0;
        while (i < s.Length)
        {
            if (IsLineBreak(s[i]))
            {
                lines.Add(s[start..i]);
                if (s[i] == '\r' && i + 1 < s.Length && s[i + 1] == '\n') i++;
                start = i + 1;
            }
            i++;
        }
        if (start < s.Length) lines.Add(s[start..]);
        return lines;
    }

    private const string Marks = ",.?!;:'…";

    /// <summary>What the cleaning regex <c>[^\w\s_,\.\?!;:'…]</c> leaves when it deletes every other character.</summary>
    public static string KeepWordSpaceAndMarks(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var r in s.EnumerateRunes())
        {
            if (IsWordChar(r) || IsSpace(r.Value) || (r.Value <= 0xFFFF && Marks.Contains((char)r.Value)))
                sb.Append(r.ToString());
        }
        return sb.ToString();
    }
}
