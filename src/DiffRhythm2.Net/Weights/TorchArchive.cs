using System.IO.Compression;
using System.Text;

namespace DiffRhythm2.Weights;

/// <summary>Element types a checkpoint tensor can have.</summary>
public enum TorchDType
{
    /// <summary>32-bit float.</summary>
    Float32,
    /// <summary>16-bit float.</summary>
    Float16,
    /// <summary>16-bit brain float.</summary>
    BFloat16,
    /// <summary>64-bit float.</summary>
    Float64,
    /// <summary>64-bit integer.</summary>
    Int64,
    /// <summary>32-bit integer.</summary>
    Int32,
    /// <summary>16-bit integer.</summary>
    Int16,
    /// <summary>8-bit integer.</summary>
    Int8,
    /// <summary>8-bit unsigned integer.</summary>
    UInt8,
    /// <summary>Boolean, one byte.</summary>
    Bool,
}

/// <summary>Helpers on <see cref="TorchDType"/>.</summary>
public static class TorchDTypeExtensions
{
    /// <summary>Size of one element in bytes.</summary>
    public static int ElementSize(this TorchDType t) => t switch
    {
        TorchDType.Float32 or TorchDType.Int32 => 4,
        TorchDType.Float16 or TorchDType.BFloat16 or TorchDType.Int16 => 2,
        TorchDType.Float64 or TorchDType.Int64 => 8,
        _ => 1,
    };

    /// <summary>The name safetensors uses for the type (<c>F32</c>, <c>BF16</c>, <c>I64</c>, ...).</summary>
    public static string SafeTensorsName(this TorchDType t) => t switch
    {
        TorchDType.Float32 => "F32",
        TorchDType.Float16 => "F16",
        TorchDType.BFloat16 => "BF16",
        TorchDType.Float64 => "F64",
        TorchDType.Int64 => "I64",
        TorchDType.Int32 => "I32",
        TorchDType.Int16 => "I16",
        TorchDType.Int8 => "I8",
        TorchDType.UInt8 => "U8",
        _ => "BOOL",
    };
}

/// <summary>A checkpoint that cannot be read: not a supported archive, damaged, or holding something the reader refuses.</summary>
public sealed class TorchArchiveException : IOException
{
    /// <summary>The tensor the problem is about, when there is one.</summary>
    public string? Tensor { get; }

    /// <summary>Creates the exception.</summary>
    public TorchArchiveException(string message, string? tensor = null) : base(tensor is null ? message : $"tensor '{tensor}': {message}") => Tensor = tensor;
}

/// <summary>One tensor of a checkpoint: what the pickle says about it. No data has been read.</summary>
public sealed class TorchTensorInfo
{
    /// <summary>Key in the state dictionary.</summary>
    public string Name { get; }
    /// <summary>Element type.</summary>
    public TorchDType DType { get; }
    /// <summary>Dimensions.</summary>
    public IReadOnlyList<long> Shape { get; }
    /// <summary>Size of the data in bytes.</summary>
    public long ByteCount { get; }

    internal long[] Strides { get; }
    internal string StorageKey { get; }
    internal long StorageOffsetBytes { get; }

    internal TorchTensorInfo(string name, TorchDType dtype, long[] shape, long[] strides, string storageKey, long storageOffsetElements)
    {
        Name = name;
        DType = dtype;
        Shape = shape;
        Strides = strides;
        StorageKey = storageKey;
        long n = 1;
        foreach (var d in shape) n = checked(n * d);
        ByteCount = n * dtype.ElementSize();
        StorageOffsetBytes = storageOffsetElements * dtype.ElementSize();
    }

    internal bool IsContiguous()
    {
        long expected = 1;
        for (int i = Shape.Count - 1; i >= 0; i--)
        {
            if (Shape[i] == 1) continue;
            if (Strides[i] != expected) return false;
            expected *= Shape[i];
        }
        return true;
    }
}

/// <summary>
/// A <c>torch.save</c> archive (the zip format of PyTorch 1.6 and later): lists the tensors of its state dictionary and streams the data of
/// one tensor at a time. The pickle inside is interpreted by <see cref="PickleReader"/>, which only knows the objects a checkpoint holds, so
/// nothing in the file is ever executed. Tensors must be contiguous.
/// </summary>
public sealed class TorchArchive : IDisposable
{
    private const int ChunkSize = 1 << 20;

    private readonly ZipArchive zip;
    private readonly string root;
    private readonly Dictionary<string, ZipArchiveEntry> storages = new(StringComparer.Ordinal);

    /// <summary>The path the archive was opened from.</summary>
    public string Path { get; }

    /// <summary>The tensors, in the order of the pickle.</summary>
    public IReadOnlyList<TorchTensorInfo> Tensors { get; }

    /// <summary>Opens an archive. <paramref name="stateKey"/> names the dictionary that wraps the state dictionary (<c>generator</c> for the decoder), or null if the top level is the state dictionary.</summary>
    public TorchArchive(string path, string? stateKey = null)
    {
        Path = path;
        using (var probe = File.OpenRead(path))
        {
            Span<byte> head = stackalloc byte[4];
            int n = probe.Read(head);
            if (n < 4 || head[0] != 'P' || head[1] != 'K')
                throw new TorchArchiveException(n >= 2 && head[0] == 0x80
                    ? $"{path} uses the legacy (non-zip) torch serialization, which is not supported; save it again with a current PyTorch"
                    : $"{path} is not a torch.save archive (it is not a zip file)");
        }
        try { zip = new ZipArchive(File.OpenRead(path), ZipArchiveMode.Read); }
        catch (InvalidDataException e) { throw new TorchArchiveException($"{path} is damaged or incomplete: {e.Message}"); }

        try
        {
            var pickle = zip.Entries.FirstOrDefault(e => e.FullName == "data.pkl" || e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal))
                ?? throw new TorchArchiveException($"{path} has no data.pkl: it is not a torch.save archive");
            root = pickle.FullName[..^"data.pkl".Length];
            foreach (var e in zip.Entries)
                if (e.FullName.StartsWith(root + "data/", StringComparison.Ordinal)) storages[e.FullName[(root.Length + "data/".Length)..]] = e;

            var order = zip.GetEntry(root + "byteorder");
            if (order is not null)
            {
                using var r = new StreamReader(order.Open(), Encoding.ASCII);
                var text = r.ReadToEnd().Trim();
                if (text != "little") throw new TorchArchiveException($"{path} has byte order '{text}'; only little endian is supported");
            }

            byte[] bytes;
            using (var s = pickle.Open())
            {
                var ms = new MemoryStream();
                s.CopyTo(ms);
                bytes = ms.ToArray();
            }
            var top = new PickleReader(bytes).Read();
            if (stateKey is not null)
            {
                var found = top.Items.FirstOrDefault(kv => kv.Key is string k && k == stateKey);
                if (found.Key is null || found.Value is not PickleReader.Dict inner)
                    throw new TorchArchiveException($"{path} has no state dictionary under the key '{stateKey}'");
                top = inner;
            }
            var list = new List<TorchTensorInfo>();
            foreach (var kv in top.Items)
            {
                if (kv.Key is not string name) throw new TorchArchiveException("a key of the state dictionary is not a string");
                if (kv.Value is not PickleReader.Tensor t) throw new TorchArchiveException("the entry is not a tensor", name);
                list.Add(new TorchTensorInfo(name, t.Storage.DType, t.Shape, t.Strides, t.Storage.Key, t.Offset));
            }
            Tensors = list;
        }
        catch
        {
            zip.Dispose();
            throw;
        }
    }

    /// <summary>Throws a <see cref="TorchArchiveException"/> naming the tensor if its data cannot be read: non-contiguous, missing file, or shorter than needed.</summary>
    public void EnsureReadable(TorchTensorInfo t)
    {
        if (!t.IsContiguous())
            throw new TorchArchiveException($"its strides ({string.Join(", ", t.Strides)}) are not those of a contiguous tensor of shape ({string.Join(", ", t.Shape)}); save it with .contiguous()", t.Name);
        if (!storages.TryGetValue(t.StorageKey, out var entry))
            throw new TorchArchiveException($"its data file data/{t.StorageKey} is missing from the archive", t.Name);
        if (entry.Length < t.StorageOffsetBytes + t.ByteCount)
            throw new TorchArchiveException($"its data is cut short: the archive holds {entry.Length} bytes of it and {t.StorageOffsetBytes + t.ByteCount} are needed", t.Name);
    }

    /// <summary>Copies the raw little-endian bytes of one tensor to <paramref name="destination"/>, in chunks of 1 MiB.</summary>
    public void CopyTo(TorchTensorInfo t, Stream destination, CancellationToken cancellation = default)
    {
        EnsureReadable(t);
        using var s = storages[t.StorageKey].Open();
        Skip(s, t.StorageOffsetBytes, t);
        var buffer = new byte[(int)Math.Min(ChunkSize, Math.Max(1, t.ByteCount))];
        long left = t.ByteCount;
        while (left > 0)
        {
            cancellation.ThrowIfCancellationRequested();
            int n = s.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
            if (n <= 0) throw new TorchArchiveException("its data ends before the end of the tensor", t.Name);
            destination.Write(buffer, 0, n);
            left -= n;
        }
    }

    private static void Skip(Stream s, long count, TorchTensorInfo t)
    {
        if (count == 0) return;
        if (s.CanSeek) { s.Seek(count, SeekOrigin.Current); return; }
        var scratch = new byte[(int)Math.Min(ChunkSize, count)];
        while (count > 0)
        {
            int n = s.Read(scratch, 0, (int)Math.Min(scratch.Length, count));
            if (n <= 0) throw new TorchArchiveException("its data ends before the end of the tensor", t.Name);
            count -= n;
        }
    }

    /// <summary>Releases the file.</summary>
    public void Dispose() => zip.Dispose();
}
