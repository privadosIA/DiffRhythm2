using System.Text;

namespace DiffRhythm2.Weights;

/// <summary>One tensor to be written by <see cref="SafeTensorsStreamWriter"/>.</summary>
/// <param name="Name">Tensor name.</param>
/// <param name="DType">Element type.</param>
/// <param name="Shape">Dimensions.</param>
/// <param name="ByteCount">Size of the data in bytes.</param>
/// <param name="WriteData">Writes exactly <paramref name="ByteCount"/> bytes of data to the stream it is given.</param>
public sealed record SafeTensorsEntry(string Name, TorchDType DType, IReadOnlyList<long> Shape, long ByteCount, Action<Stream> WriteData);

/// <summary>
/// Writes a safetensors file without holding the tensors in memory, with the layout of the Python <c>safetensors</c> library: a compact header
/// whose first key is <c>__metadata__</c> = <c>{"format":"pt"}</c>, tensors sorted by data type (the library puts the wider types first: int64,
/// then float32, and so on) and by name within a type, contiguous data in that order, and the header padded with spaces to a multiple of 8 bytes.
/// The result depends only on the entries, so it is reproducible.
/// </summary>
public static class SafeTensorsStreamWriter
{
    // The Rust library sorts by `right.dtype().cmp(&left.dtype())` and then by name: the order of its Dtype enum, widest first.
    private static readonly string[] DTypeOrder = { "U64", "I64", "F64", "F32", "U32", "I32", "BF16", "F16", "U16", "I16", "F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL" };

    private static int Rank(TorchDType t)
    {
        int i = Array.IndexOf(DTypeOrder, t.SafeTensorsName());
        return i < 0 ? DTypeOrder.Length : i;
    }

    /// <summary>Writes the file. Entries are sorted by data type (widest first) and then by name (ordinal); names must be distinct.</summary>
    public static void Write(string path, IEnumerable<SafeTensorsEntry> entries, CancellationToken cancellation = default)
    {
        var sorted = entries.OrderBy(e => Rank(e.DType)).ThenBy(e => e.Name, StringComparer.Ordinal).ToList();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var e in sorted)
            if (!names.Add(e.Name)) throw new ArgumentException($"tensor '{e.Name}' appears twice");

        var header = new StringBuilder("{\"__metadata__\":{\"format\":\"pt\"}");
        long offset = 0;
        foreach (var e in sorted)
        {
            header.Append(',');
            AppendString(header, e.Name);
            header.Append(":{\"dtype\":\"").Append(e.DType.SafeTensorsName()).Append("\",\"shape\":[").Append(string.Join(",", e.Shape)).Append("],\"data_offsets\":[")
                  .Append(offset).Append(',').Append(offset + e.ByteCount).Append("]}");
            offset += e.ByteCount;
        }
        header.Append('}');
        while (header.Length % 8 != 0) header.Append(' ');
        var headerBytes = Encoding.UTF8.GetBytes(header.ToString());
        if (headerBytes.Length % 8 != 0) throw new ArgumentException("the header is not a multiple of 8 bytes (non-ASCII tensor names are not supported)");

        using var file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
        file.Write(BitConverter.GetBytes((ulong)headerBytes.Length));
        file.Write(headerBytes);
        foreach (var e in sorted)
        {
            cancellation.ThrowIfCancellationRequested();
            long before = file.Position;
            e.WriteData(file);
            if (file.Position - before != e.ByteCount)
                throw new InvalidDataException($"tensor '{e.Name}': {file.Position - before} bytes were written, {e.ByteCount} expected");
        }
    }

    private static void AppendString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4")); else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }
}
