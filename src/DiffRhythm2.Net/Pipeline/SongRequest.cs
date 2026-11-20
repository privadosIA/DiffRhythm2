using System.Globalization;
using System.Text.RegularExpressions;
using DiffRhythm2.Text;

namespace DiffRhythm2.Pipeline;

/// <summary>
/// A request to make a song, with the limits and messages of the reference service (<c>Engine.validate</c>). Call <see cref="Validate"/> to check it and get
/// the normalized values the pipeline uses.
/// </summary>
/// <param name="Lyrics">The lyrics, with optional structure tags such as <c>[verse]</c> on their own lines. English only.</param>
/// <param name="Style">Comma-separated style words, for example <c>Pop, Piano, Female vocals</c>.</param>
/// <param name="Seconds">Length ceiling in seconds, 10 to 210 (the song ends by itself at the end of the lyrics).</param>
/// <param name="CfgStrength">Classifier-free guidance strength, 0.5 to 10.</param>
/// <param name="Steps">Euler steps per block, 4 to 64.</param>
/// <param name="Seed">Seed of the random numbers, 0 or more; <c>null</c> picks one, which is returned in the result.</param>
public sealed record SongRequest(string Lyrics, string Style, double Seconds = SongRequest.DefaultSeconds, double CfgStrength = SongRequest.DefaultCfgStrength, int Steps = SongRequest.DefaultSteps, long? Seed = null)
{
    /// <summary>Default length ceiling, 210 seconds.</summary>
    public const double DefaultSeconds = 210.0;
    /// <summary>Default guidance strength.</summary>
    public const double DefaultCfgStrength = 2.0;
    /// <summary>Default number of Euler steps per block.</summary>
    public const int DefaultSteps = 16;
    /// <summary>Most characters (Unicode code points) the lyrics may have.</summary>
    public const int MaxLyricsCharacters = 20_000;
    /// <summary>Most characters the normalized style may have.</summary>
    public const int MaxStyleCharacters = 500;

    /// <summary>The message for a seed that is not a whole number from 0 up to 2^63 - 1.</summary>
    public const string SeedMessage = "seed must be a whole number, 0 or more";

    /// <summary>Frames of latent per second of song.</summary>
    public const int FramesPerSecond = 5;

    private static readonly Regex CommaSpacing = new($"[{PyText.SpaceClass}]*,[{PyText.SpaceClass}]*", RegexOptions.CultureInvariant, RegexLimits.Match);

    /// <summary>A request whose style is given as a list of words (joined with a comma and a space first).</summary>
    public static SongRequest FromWords(string lyrics, IEnumerable<string> styleWords, double seconds = DefaultSeconds, double cfgStrength = DefaultCfgStrength, int steps = DefaultSteps, long? seed = null) =>
        new(lyrics, string.Join(", ", styleWords), seconds, cfgStrength, steps, seed);

    /// <summary>Reads a seed from text; false when it is not a whole number from 0 up to 2^63 - 1.</summary>
    public static bool TryParseSeed(string text, out long seed) =>
        long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out seed);

    /// <summary>
    /// Checks the request in the order the service does and returns the values to use: the lyrics prepared (known tags in lower case, unknown ones dropped,
    /// <c>[start]</c> added), the style normalized and the length in frames. Throws <see cref="SongRequestException"/> naming the field.
    /// </summary>
    public ValidatedSong Validate()
    {
        if (Lyrics is null || PyText.Strip(Lyrics).Length == 0) throw new SongRequestException("lyrics", "lyrics must be a non-empty text");
        if (CountCharacters(Lyrics) > MaxLyricsCharacters) throw new SongRequestException("lyrics", $"lyrics are too long ({MaxLyricsCharacters} characters at most)");

        var style = NormalizeStyle(Style ?? "");
        if (style.Length == 0) throw new SongRequestException("style", "style must name at least one style");
        if (CountCharacters(style) > MaxStyleCharacters) throw new SongRequestException("style", $"style is too long ({MaxStyleCharacters} characters at most)");

        string prepared;
        try { prepared = LyricsTokenizer.Prepare(Lyrics); }
        catch (G2pException) { throw new SongRequestException("lyrics", "lyrics must have at least one line of text besides the structure tags"); }

        if (!(Seconds >= 10.0 && Seconds <= 210.0)) throw new SongRequestException("max_secs", "max_secs must be between 10.0 and 210.0");
        if (!(CfgStrength >= 0.5 && CfgStrength <= 10.0)) throw new SongRequestException("cfg_strength", "cfg_strength must be between 0.5 and 10.0");
        if (Steps < 4 || Steps > 64) throw new SongRequestException("steps", "steps must be between 4 and 64");
        if (Seed is < 0) throw new SongRequestException("seed", SeedMessage);

        return new ValidatedSong(prepared, style, Seconds, (int)(Seconds * FramesPerSecond), CfgStrength, Steps, Seed);
    }

    /// <summary>The service's style normalization: exactly one comma and one space between words, no leading or trailing commas or spaces.</summary>
    public static string NormalizeStyle(string style) =>
        CommaSpacing.Replace(PyText.Strip(style), ", ").Trim(',', ' ');

    private static int CountCharacters(string text) => text.EnumerateRunes().Count();   // Python counts code points
}

/// <summary>A request that passed <see cref="SongRequest.Validate"/>.</summary>
/// <param name="Lyrics">Prepared lyrics (one tag or text line per line).</param>
/// <param name="Style">Normalized style.</param>
/// <param name="Seconds">Length ceiling in seconds.</param>
/// <param name="Frames">Latent frames requested, <c>int(seconds * 5)</c>; the sampler rounds up to whole blocks of 10.</param>
/// <param name="CfgStrength">Classifier-free guidance strength.</param>
/// <param name="Steps">Euler steps per block.</param>
/// <param name="Seed">The request's seed, or null when none was given.</param>
public sealed record ValidatedSong(string Lyrics, string Style, double Seconds, int Frames, double CfgStrength, int Steps, long? Seed);
