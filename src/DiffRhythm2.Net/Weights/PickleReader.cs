using System.Text;

namespace DiffRhythm2.Weights;

/// <summary>
/// A pickle interpreter that knows only what a <c>torch.save</c> state dictionary contains. It builds plain descriptions (dictionaries,
/// tuples, tensor descriptors) and never calls anything: the only "objects" it can create are an ordered dictionary and a tensor
/// descriptor, selected from a closed list of allowed globals. Anything else (another global, another opcode) is refused by name.
/// </summary>
internal sealed class PickleReader
{
    internal sealed class Dict
    {
        public List<KeyValuePair<object, object>> Items { get; } = new();
    }

    internal sealed record GlobalRef(string Module, string Name);

    internal sealed record Storage(TorchDType DType, string Key);

    internal sealed record Tensor(Storage Storage, long Offset, long[] Shape, long[] Strides);

    private sealed class Mark { }

    private static readonly Dictionary<string, TorchDType> StorageTypes = new()
    {
        ["FloatStorage"] = TorchDType.Float32,
        ["HalfStorage"] = TorchDType.Float16,
        ["BFloat16Storage"] = TorchDType.BFloat16,
        ["DoubleStorage"] = TorchDType.Float64,
        ["LongStorage"] = TorchDType.Int64,
        ["IntStorage"] = TorchDType.Int32,
        ["ShortStorage"] = TorchDType.Int16,
        ["CharStorage"] = TorchDType.Int8,
        ["ByteStorage"] = TorchDType.UInt8,
        ["BoolStorage"] = TorchDType.Bool,
    };

    private readonly byte[] data;
    private int pos;
    private readonly List<object?> stack = new();
    private readonly Dictionary<int, object?> memo = new();

    public PickleReader(byte[] data) => this.data = data;

    private TorchArchiveException Error(string message) => new($"the pickle {message} (at offset {pos})");

    private byte Byte() => pos < data.Length ? data[pos++] : throw Error("ends early");

    private int Int32() { Need(4); int v = BitConverter.ToInt32(data, pos); pos += 4; return v; }

    private void Need(int n) { if (pos + n > data.Length) throw Error("ends early"); }

    private string Line()
    {
        int end = Array.IndexOf(data, (byte)'\n', pos);
        if (end < 0) throw Error("ends early");
        var s = Encoding.ASCII.GetString(data, pos, end - pos);
        pos = end + 1;
        return s;
    }

    private object? Pop() { if (stack.Count == 0) throw Error("is malformed (empty stack)"); var v = stack[^1]; stack.RemoveAt(stack.Count - 1); return v; }

    private List<object?> PopToMark()
    {
        int m = stack.FindLastIndex(x => x is Mark);
        if (m < 0) throw Error("is malformed (no mark)");
        var items = stack.GetRange(m + 1, stack.Count - m - 1);
        stack.RemoveRange(m, stack.Count - m);
        return items;
    }

    private static bool Allowed(string module, string name) =>
        (module == "collections" && name == "OrderedDict") || (module == "torch._utils" && name == "_rebuild_tensor_v2")
        || (module == "torch" && StorageTypes.ContainsKey(name));

    /// <summary>Interprets the pickle and returns the top-level dictionary.</summary>
    public Dict Read()
    {
        while (true)
        {
            int at = pos;
            byte op = Byte();
            switch (op)
            {
                case 0x80: { var v = Byte(); if (v > 4) throw Error($"has protocol {v}, which is not supported"); break; }
                case (byte)'.':
                    if (Pop() is Dict top && stack.Count == 0) return top;
                    throw Error("does not hold a dictionary");
                case (byte)'}': stack.Add(new Dict()); break;
                case (byte)')': stack.Add(Array.Empty<object?>()); break;
                case (byte)'(': stack.Add(new Mark()); break;
                case (byte)'N': stack.Add(null); break;
                case 0x88: stack.Add(true); break;
                case 0x89: stack.Add(false); break;
                case (byte)'X':
                    {
                        int len = Int32();
                        if (len < 0) throw Error("is malformed");
                        Need(len);
                        stack.Add(Encoding.UTF8.GetString(data, pos, len));
                        pos += len;
                        break;
                    }
                case (byte)'c':
                    {
                        string module = Line(), name = Line();
                        if (!Allowed(module, name))
                            throw new TorchArchiveException($"the pickle asks for '{module}.{name}', which a checkpoint must not contain; the file was not trusted and nothing was run (at offset {at})");
                        stack.Add(new GlobalRef(module, name));
                        break;
                    }
                case (byte)'J': stack.Add((long)Int32()); break;
                case (byte)'K': stack.Add((long)Byte()); break;
                case (byte)'M': { Need(2); stack.Add((long)BitConverter.ToUInt16(data, pos)); pos += 2; break; }
                case (byte)'t': stack.Add(PopToMark().ToArray()); break;
                case 0x85: { var a = Pop(); stack.Add(new[] { a }); break; }
                case 0x86: { var b = Pop(); var a = Pop(); stack.Add(new[] { a, b }); break; }
                case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); stack.Add(new[] { a, b, c }); break; }
                case (byte)'q': { var i = Byte(); memo[i] = stack.Count > 0 ? stack[^1] : throw Error("is malformed"); break; }
                case (byte)'r': { var i = Int32(); memo[i] = stack.Count > 0 ? stack[^1] : throw Error("is malformed"); break; }
                case (byte)'h': { var i = Byte(); stack.Add(memo.TryGetValue(i, out var v) ? v : throw Error("is malformed (unknown memo entry)")); break; }
                case (byte)'j': { var i = Int32(); stack.Add(memo.TryGetValue(i, out var v) ? v : throw Error("is malformed (unknown memo entry)")); break; }
                case (byte)'s': { var value = Pop(); var key = Pop(); AsDict(Pop(), out var d); d.Items.Add(new(key!, value!)); stack.Add(d); break; }
                case (byte)'u':
                    {
                        var items = PopToMark();
                        AsDict(Pop(), out var d);
                        if (items.Count % 2 != 0) throw Error("is malformed (odd number of dictionary items)");
                        for (int i = 0; i < items.Count; i += 2) d.Items.Add(new(items[i]!, items[i + 1]!));
                        stack.Add(d);
                        break;
                    }
                case (byte)'b':
                    {
                        var state = Pop();
                        if (stack.Count == 0 || stack[^1] is not Dict || state is not Dict) throw Error("sets attributes on something other than a dictionary");
                        break;   // the state of a state_dict is its _metadata (versions of the modules): not needed
                    }
                case (byte)'Q': stack.Add(PersistentId(Pop())); break;
                case (byte)'R': Reduce(); break;
                default:
                    throw new TorchArchiveException($"the pickle uses the opcode 0x{op:x2} ('{(op >= 32 && op < 127 ? (char)op : '?')}') at offset {at}, which this reader does not implement");
            }
        }
    }

    private void AsDict(object? v, out Dict d) => d = v as Dict ?? throw Error("is malformed (a dictionary was expected)");

    private Storage PersistentId(object? id)
    {
        if (id is object?[] t && t.Length == 5 && t[0] is "storage" && t[1] is GlobalRef g && g.Module == "torch" && StorageTypes.TryGetValue(g.Name, out var dtype) && t[2] is string key)
            return new Storage(dtype, key);
        throw Error("has a persistent id that is not a tensor storage");
    }

    private void Reduce()
    {
        var args = Pop() as object?[] ?? throw Error("is malformed (call arguments)");
        if (Pop() is not GlobalRef f) throw Error("calls something that is not an allowed global");
        if (f.Module == "collections")
        {
            if (args.Length != 0) throw Error("builds an ordered dictionary from a sequence, which is not supported");
            stack.Add(new Dict());
            return;
        }
        if (f.Name != "_rebuild_tensor_v2" || args.Length < 5 || args[0] is not Storage storage || args[1] is not long offset || args[2] is not object?[] size || args[3] is not object?[] stride)
            throw Error("builds a tensor in a way that is not supported");
        stack.Add(new Tensor(storage, offset, ToLongs(size), ToLongs(stride)));
    }

    private long[] ToLongs(object?[] a)
    {
        var r = new long[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] is long l && l >= 0 ? l : throw Error("has a shape or stride that is not a non-negative integer");
        return r;
    }
}
