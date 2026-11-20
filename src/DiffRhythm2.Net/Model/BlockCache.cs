using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Model;

/// <summary>How an attention call uses the <see cref="BlockCache"/>.</summary>
public enum CacheUse
{
    /// <summary>The keys/values are stored as the text section; attention sees only them (the text prefill).</summary>
    TextPrefill,

    /// <summary>The keys/values are appended to the clean section; attention sees <c>[text | clean + these]</c> (a finished block written back).</summary>
    CleanWrite,

    /// <summary>The keys/values are NOT stored; attention sees <c>[text | clean | these]</c> (a noisy block during integration).</summary>
    Transient,
}

/// <summary>
/// Per-layer key/value cache of the block-wise sampler. Keys are stored after q/k normalization and RoPE, as in the reference. A text section
/// (written once) comes first, then the clean blocks in generation order. There is deliberately no history window: the reference never enables
/// <c>num_history_block</c> (see design.md). Tensors are detached from any dispose scope; <see cref="Dispose"/> frees them.
/// </summary>
public sealed class BlockCache : IDisposable
{
    private readonly Tensor?[] textK, textV, cleanK, cleanV;

    public int Layers { get; }

    public BlockCache(int layers)
    {
        Layers = layers;
        textK = new Tensor?[layers];
        textV = new Tensor?[layers];
        cleanK = new Tensor?[layers];
        cleanV = new Tensor?[layers];
    }

    /// <summary>Length of the text section of a layer (0 when nothing was prefilled).</summary>
    public long TextLength(int layer) => textK[layer]?.shape[2] ?? 0;

    /// <summary>Length of the clean section of a layer.</summary>
    public long CleanLength(int layer) => cleanK[layer]?.shape[2] ?? 0;

    public Tensor? TextKeys(int layer) => textK[layer];
    public Tensor? TextValues(int layer) => textV[layer];
    public Tensor? CleanKeys(int layer) => cleanK[layer];
    public Tensor? CleanValues(int layer) => cleanV[layer];

    /// <summary>Applies <paramref name="use"/> for one layer and returns the keys/values to attend over (<c>[b, heads, tokens, headDim]</c>).</summary>
    public (Tensor Keys, Tensor Values) Update(int layer, Tensor k, Tensor v, CacheUse use)
    {
        switch (use)
        {
            case CacheUse.TextPrefill:
                Replace(textK, layer, k.detach().clone());
                Replace(textV, layer, v.detach().clone());
                return (textK[layer]!, textV[layer]!);

            case CacheUse.CleanWrite:
                Replace(cleanK, layer, Append(cleanK[layer], k));
                Replace(cleanV, layer, Append(cleanV[layer], v));
                return (Join(textK[layer], cleanK[layer]!), Join(textV[layer], cleanV[layer]!));

            default:
                return (Join(Join(textK[layer], cleanK[layer]), k), Join(Join(textV[layer], cleanV[layer]), v));
        }
    }

    /// <summary>A new tensor owned by the cache: <paramref name="existing"/> followed by <paramref name="added"/> (a copy of it when nothing is stored yet).</summary>
    private static Tensor Append(Tensor? existing, Tensor added) =>
        existing is null ? added.detach().clone() : cat(new[] { existing, added }, dim: 2);

    /// <summary>Concatenation along the sequence dimension for reading; a missing part is skipped and the other returned as is (not copied).</summary>
    private static Tensor Join(Tensor? a, Tensor? b)
    {
        if (a is null) return b!;
        if (b is null) return a;
        return cat(new[] { a, b }, dim: 2);
    }

    private static void Replace(Tensor?[] slots, int layer, Tensor value)
    {
        slots[layer]?.Dispose();
        slots[layer] = value.DetachFromDisposeScope();
    }

    public void Dispose()
    {
        foreach (var slots in new[] { textK, textV, cleanK, cleanV })
            for (int i = 0; i < slots.Length; i++) { slots[i]?.Dispose(); slots[i] = null; }
    }
}
