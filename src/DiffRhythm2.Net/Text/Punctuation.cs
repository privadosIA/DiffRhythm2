using System.Text.RegularExpressions;

namespace DiffRhythm2.Text;

/// <summary>Where a punctuation mark sat in its line: begin, inside, end, or alone (a line of marks only).</summary>
internal enum MarkPosition { Begin, Inside, End, Alone }

internal readonly record struct PunctuationMark(int Index, string Text, MarkPosition Position);

/// <summary>
/// A port of <c>phonemizer.punctuation.Punctuation</c> as the reference uses it (marks <c>,.?!;:'…</c>, preserved): the marks are taken out of the
/// text so the backend never sees them, and put back, as separate items, after phonemization. A <c>,</c> or <c>.</c> between two digits is not a
/// mark. The word separator replaces spaces inside a mark.
/// </summary>
internal sealed class Punctuation
{
    private static readonly Regex MarksRegex;

    static Punctuation()
    {
        string space = $"[{PyText.SpaceClass}]";
        // alternatives: marks that are never decimal separators, then , and . when not touching a digit on the left or on the right
        MarksRegex = new Regex($@"({space}*(?:[?!;:'…]|(?<![0-9])[,.]|[,.](?![0-9]))+{space}*)+".Replace(@"…", "…"), RegexOptions.CultureInvariant, RegexLimits.Match);
    }

    /// <summary>Removes the marks from the lines and returns the non-empty chunks between them together with the marks to restore.</summary>
    public static (List<string> Chunks, List<PunctuationMark> Marks) Preserve(IReadOnlyList<string> lines)
    {
        var chunks = new List<string>();
        var marks = new List<PunctuationMark>();
        for (int num = 0; num < lines.Count; num++)
        {
            var (lineChunks, lineMarks) = PreserveLine(lines[num], num);
            chunks.AddRange(lineChunks);
            marks.AddRange(lineMarks);
        }
        return (chunks.Where(c => c.Length > 0).ToList(), marks);
    }

    private static (List<string> Chunks, List<PunctuationMark> Marks) PreserveLine(string line, int num)
    {
        var matches = MarksRegex.Matches(line);
        if (matches.Count == 0) return (new List<string> { line }, new List<PunctuationMark>());
        if (matches.Count == 1 && matches[0].Value == line) return (new List<string>(), new List<PunctuationMark> { new(num, line, MarkPosition.Alone) });

        var marks = new List<PunctuationMark>();
        for (int i = 0; i < matches.Count; i++)
        {
            var value = matches[i].Value;
            var position = MarkPosition.Inside;
            if (i == 0 && line.StartsWith(value, StringComparison.Ordinal)) position = MarkPosition.Begin;
            else if (i == matches.Count - 1 && line.EndsWith(value, StringComparison.Ordinal)) position = MarkPosition.End;
            marks.Add(new PunctuationMark(num, value, position));
        }

        var chunks = new List<string>();
        foreach (var mark in marks)
        {
            // split on the first occurrence of the mark's text, like str.split(mark)
            int at = line.IndexOf(mark.Text, StringComparison.Ordinal);
            chunks.Add(line[..at]);
            line = line[(at + mark.Text.Length)..];
        }
        chunks.Add(line);
        return (chunks, marks);
    }

    /// <summary>Puts the marks back into the phonemized chunks (the reference's <c>Punctuation.restore</c>).</summary>
    public static List<string> Restore(IReadOnlyList<string> phonemized, IReadOnlyList<PunctuationMark> markList, string wordSeparator, bool strip)
    {
        var text = new List<string>(phonemized);
        var marks = new List<PunctuationMark>(markList);
        var punctuated = new List<string>();
        int pos = 0;

        string Tail(string mark) => strip || mark.EndsWith(wordSeparator, StringComparison.Ordinal) ? "" : wordSeparator;

        while (text.Count > 0 || marks.Count > 0)
        {
            if (marks.Count == 0)
            {
                foreach (var line in text)
                {
                    var l = line;
                    if (!strip && wordSeparator.Length > 0 && !l.EndsWith(wordSeparator, StringComparison.Ordinal)) l += wordSeparator;
                    punctuated.Add(l);
                }
                text.Clear();
            }
            else if (text.Count == 0)
            {
                // nothing left to phonemize: the remaining marks alone, spaces inside them replaced by the word separator
                punctuated.Add(string.Concat(marks.Select(m => m.Text)).Replace(" ", wordSeparator));
                marks.Clear();
            }
            else
            {
                var current = marks[0];
                if (current.Index == pos)
                {
                    marks.RemoveAt(0);
                    string mark = current.Text.Replace(" ", wordSeparator);
                    if (wordSeparator.Length > 0 && text[0].EndsWith(wordSeparator, StringComparison.Ordinal)) text[0] = text[0][..^wordSeparator.Length];

                    switch (current.Position)
                    {
                        case MarkPosition.Begin:
                            text[0] = mark + text[0];
                            break;
                        case MarkPosition.End:
                            punctuated.Add(text[0] + mark + Tail(mark));
                            text.RemoveAt(0);
                            pos++;
                            break;
                        case MarkPosition.Alone:
                            punctuated.Add(mark + Tail(mark));
                            pos++;
                            break;
                        default:
                            if (text.Count == 1) text[0] += mark;
                            else
                            {
                                var first = text[0];
                                text.RemoveAt(0);
                                text[0] = first + mark + text[0];
                            }
                            break;
                    }
                }
                else
                {
                    punctuated.Add(text[0]);
                    text.RemoveAt(0);
                    pos++;
                }
            }
        }
        return punctuated;
    }
}
