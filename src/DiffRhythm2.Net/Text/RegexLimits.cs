namespace DiffRhythm2.Text;

/// <summary>
/// The match timeout of every regular expression that reads user text (lyrics, style). The patterns are linear in practice, so this is a safety net: input that
/// somehow makes one run away ends the job with a <see cref="System.Text.RegularExpressions.RegexMatchTimeoutException"/> instead of hanging it.
/// </summary>
internal static class RegexLimits
{
    public static readonly TimeSpan Match = TimeSpan.FromSeconds(5);
}
