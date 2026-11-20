using System.Text.Json;
using System.Text.Json.Serialization;
using DiffRhythm2.Pipeline;

namespace DiffRhythm2.Service;

/// <summary>
/// What a worker process and the service say to each other, as JSON lines: the request on the worker's standard input, one progress line per
/// block on its standard output. Both sides use this class, so they cannot drift apart.
/// </summary>
internal static class WorkerProtocol
{
    /// <summary>The argument that makes the service executable act as a worker; the song's WAV path follows it.</summary>
    public const string Argument = "--worker";

    /// <summary>Exit codes of a worker: the song is written; the song could not be made (the cause is the last line of standard error); cancelled.</summary>
    public const int Done = 0, Failed = 1, Cancelled = 3;

    private sealed record RequestLine(
        [property: JsonPropertyName("lyrics")] string Lyrics, [property: JsonPropertyName("style")] string Style, [property: JsonPropertyName("seconds")] double Seconds,
        [property: JsonPropertyName("cfg_strength")] double CfgStrength, [property: JsonPropertyName("steps")] int Steps, [property: JsonPropertyName("seed")] long? Seed);

    private sealed record ProgressLine([property: JsonPropertyName("done")] int Done, [property: JsonPropertyName("total")] int Total);

    public static string EncodeRequest(SongRequest r) =>
        JsonSerializer.Serialize(new RequestLine(r.Lyrics, r.Style, r.Seconds, r.CfgStrength, r.Steps, r.Seed));

    public static SongRequest DecodeRequest(string line)
    {
        var r = JsonSerializer.Deserialize<RequestLine>(line) ?? throw new JsonException("the request line is empty");
        return new SongRequest(r.Lyrics, r.Style, r.Seconds, r.CfgStrength, r.Steps, r.Seed);
    }

    public static string EncodeProgress(int done, int total) => JsonSerializer.Serialize(new ProgressLine(done, total));

    /// <summary>Reads a progress line; false for anything else (a stray line from a native library, say), which the caller ignores.</summary>
    public static bool TryDecodeProgress(string line, out (int Done, int Total) progress)
    {
        progress = default;
        if (!line.StartsWith('{')) return false;
        try
        {
            var p = JsonSerializer.Deserialize<ProgressLine>(line);
            if (p is null) return false;
            progress = (p.Done, p.Total);
            return true;
        }
        catch (JsonException) { return false; }
    }
}
