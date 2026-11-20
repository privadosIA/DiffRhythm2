using System.Text.Json;

namespace DiffRhythm2.Weights;

/// <summary>One tensor listed in a safetensors header.</summary>
/// <param name="Name">Tensor name.</param>
/// <param name="DType">The safetensors type name (<c>F32</c>, ...).</param>
/// <param name="Shape">Dimensions.</param>
/// <param name="Start">Offset of the data from the start of the data section.</param>
/// <param name="End">End offset of the data (exclusive).</param>
public sealed record SafeTensorsHeaderEntry(string Name, string DType, IReadOnlyList<long> Shape, long Start, long End);

/// <summary>The header of a safetensors file: enough to list its tensors and locate their bytes, without reading the data.</summary>
public sealed class SafeTensorsHeader
{
    /// <summary>The tensors, in file order.</summary>
    public IReadOnlyList<SafeTensorsHeaderEntry> Entries { get; }

    /// <summary>Offset in the file where the data section starts.</summary>
    public long DataStart { get; }

    private SafeTensorsHeader(IReadOnlyList<SafeTensorsHeaderEntry> entries, long dataStart) { Entries = entries; DataStart = dataStart; }

    /// <summary>Reads and checks the header of a file; throws <see cref="InvalidDataException"/> if it is not a well-formed safetensors file.</summary>
    public static SafeTensorsHeader Read(string path)
    {
        using var f = File.OpenRead(path);
        if (f.Length < 8) throw new InvalidDataException($"{path} is too short to be a safetensors file");
        Span<byte> len = stackalloc byte[8];
        f.ReadExactly(len);
        ulong n = BitConverter.ToUInt64(len);
        if (n > 100_000_000 || 8 + (long)n > f.Length) throw new InvalidDataException($"{path} has an impossible header length ({n})");
        var json = new byte[n];
        f.ReadExactly(json);
        var entries = new List<SafeTensorsHeaderEntry>();
        try
        {
            using var doc = JsonDocument.Parse(json);
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (p.Name == "__metadata__") continue;
                var shape = p.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToList();
                var off = p.Value.GetProperty("data_offsets");
                entries.Add(new SafeTensorsHeaderEntry(p.Name, p.Value.GetProperty("dtype").GetString()!, shape, off[0].GetInt64(), off[1].GetInt64()));
            }
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            throw new InvalidDataException($"{path} has a malformed safetensors header: {e.Message}");
        }
        long dataStart = 8 + (long)n;
        if (entries.Count > 0 && dataStart + entries.Max(e => e.End) > f.Length) throw new InvalidDataException($"{path} is shorter than its header says");
        return new SafeTensorsHeader(entries, dataStart);
    }
}
