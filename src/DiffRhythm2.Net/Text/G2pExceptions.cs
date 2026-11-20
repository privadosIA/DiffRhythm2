namespace DiffRhythm2.Text;

/// <summary>Base of the errors the lyrics front end raises on purpose (as opposed to bugs): unsupported or unspeakable input.</summary>
public class G2pException : Exception
{
    public G2pException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>A line with no ASCII letter and no Chinese character, which the reference refuses (<c>Unknown language: other</c>).</summary>
public sealed class NoSungTextException : G2pException
{
    public NoSungTextException(string text) : base($"the line has no sung text (it needs at least one ASCII letter): \"{text}\"") { }
}

/// <summary>A Chinese segment: the Mandarin front end is not part of this port.</summary>
public sealed class ChineseNotSupportedException : G2pException
{
    public ChineseNotSupportedException(string text) : base($"Chinese lyrics are not supported by this front end (the line contains Chinese characters): \"{text}\"") { }
}

/// <summary>A number the reference's <c>inflect</c> cannot spell (it stops at 36 digits) and so neither can the port.</summary>
public sealed class NumberOutOfRangeException : G2pException
{
    public NumberOutOfRangeException(string number) : base($"the number {number} is beyond what can be spelled out (36 digits at most)") { }
}

/// <summary>An error of the lyrics tokenizer, attributed to the line that caused it.</summary>
public sealed class LyricsLineException : G2pException
{
    public int Line { get; }
    public string Text { get; }

    public LyricsLineException(int line, string text, Exception inner)
        : base($"line {line} (\"{text}\") cannot be tokenized: {inner.Message}", inner)
    {
        Line = line;
        Text = text;
    }
}
