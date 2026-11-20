using System.Text.Json;
using System.Text.RegularExpressions;
using DiffRhythm2.Pipeline;

namespace DiffRhythm2.Service;

/// <summary>A request body that cannot be served; the message is what goes in <c>{"error": ...}</c>.</summary>
internal sealed class BadRequestException : Exception
{
    public BadRequestException(string message) : base(message) { }
}

/// <summary>
/// Reads the JSON body of <c>POST /generate</c> the way the Python service's <c>validate</c> does: the same checks in the same order with the same
/// messages. The field checks and limits are the pipeline's (<see cref="SongRequest.Validate"/>); what is added here is the JSON typing (a number must be a
/// number, a seed a whole number...), the service's seed default of 0 and the truncation of a fractional <c>steps</c>.
/// </summary>
internal static partial class RequestParser
{
    // The order the Python service reports problems in: lyrics (type, blank, length), style (type, blank, length), the lyrics' tags, then the numbers.
    private static int Rank(SongRequestException e) => e.Field switch
    {
        "lyrics" => e.Message.StartsWith("lyrics must have at least one line", StringComparison.Ordinal) ? 2 : 0,
        "style" => 1,
        "max_secs" => 3,
        "cfg_strength" => 4,
        "steps" => 5,
        _ => 6,
    };

    public static SongRequest Parse(ReadOnlySpan<byte> body)
    {
        JsonDocument doc;
        try { doc = JsonDocument.Parse(body.IsEmpty ? "{}"u8.ToArray() : body.ToArray()); }
        catch (JsonException e) { throw new BadRequestException("the body is not valid JSON: " + e.Message); }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) throw new BadRequestException("the body must be a JSON object");

            (int Rank, string Message)? typeError = null;
            void Type(int rank, string message) { if (typeError is null || rank < typeError.Value.Rank) typeError = (rank, message); }

            // lyrics
            string lyrics = "";
            if (root.TryGetProperty("lyrics", out var l) && l.ValueKind == JsonValueKind.String) lyrics = l.GetString()!;
            else Type(0, "lyrics must be a non-empty text");

            // style: a text or a list of words
            string style = "";
            if (root.TryGetProperty("style", out var s))
            {
                if (s.ValueKind == JsonValueKind.Array) style = string.Join(", ", s.EnumerateArray().Select(PythonStr));
                else if (s.ValueKind == JsonValueKind.String) style = s.GetString()!;
                else Type(1, "style must be a text or a list of words");
            }
            if (typeError is { Rank: 0 }) throw new BadRequestException(typeError.Value.Message);

            double seconds = Number(root, "max_secs", SongRequest.DefaultSeconds, 3, Type, 10.0, 210.0, "10.0", "210.0");
            double cfg = Number(root, "cfg_strength", SongRequest.DefaultCfgStrength, 4, Type, 0.5, 10.0, "0.5", "10.0");
            double stepsValue = Number(root, "steps", SongRequest.DefaultSteps, 5, Type, 4, 64, "4", "64");
            long seed = 0;
            if (root.TryGetProperty("seed", out var sd))
            {
                if (sd.ValueKind == JsonValueKind.Number && WholeNumber().IsMatch(sd.GetRawText()) && sd.TryGetInt64(out var sv) && sv >= 0) seed = sv;
                else Type(6, SongRequest.SeedMessage);
            }

            // the pipeline's checks, with placeholders for the fields that already have a type error
            var request = new SongRequest(lyrics, style.Length == 0 && typeError is { Rank: 1 } ? "x" : style,
                typeError is { Rank: 3 } ? SongRequest.DefaultSeconds : seconds, typeError is { Rank: 4 } ? SongRequest.DefaultCfgStrength : cfg,
                typeError is { Rank: 5 } ? SongRequest.DefaultSteps : (int)Math.Truncate(Math.Clamp(stepsValue, 4, 64)), seed);
            try { request.Validate(); }
            catch (SongRequestException e)
            {
                if (typeError is null || Rank(e) < typeError.Value.Rank) throw new BadRequestException(e.Message);
            }
            if (typeError is not null) throw new BadRequestException(typeError.Value.Message);
            return request;
        }
    }

    [GeneratedRegex(@"^-?\d+$")]
    private static partial Regex WholeNumber();

    // A number field: absent gives the default; wrong type or out of range becomes a typed problem (the range message comes from the pipeline for the
    // fields it checks; here only the type, and the range of a value that would be clamped).
    private static double Number(JsonElement root, string name, double fallback, int rank, Action<int, string> type, double lo, double hi, string loText, string hiText)
    {
        if (!root.TryGetProperty(name, out var v)) return fallback;
        if (v.ValueKind != JsonValueKind.Number) { type(rank, $"{name} must be a number"); return fallback; }
        double value = v.GetDouble();
        if (!(value >= lo && value <= hi)) type(rank, $"{name} must be between {loText} and {hiText}");
        return value;
    }

    // str() of a JSON value as Python prints it, for the items of a style list.
    private static string PythonStr(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString()!,
        JsonValueKind.True => "True",
        JsonValueKind.False => "False",
        JsonValueKind.Null => "None",
        _ => e.GetRawText(),
    };
}
