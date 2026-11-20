using TorchSharp;
using TorchSharp.Modules;
using MusicGen.Core.Layers;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.Model;

/// <summary>
/// DiffRhythm2's Llama attention: bias-free q/k/v/o projections, <c>q_norm</c>/<c>k_norm</c> (<see cref="LlamaRmsNorm"/> over the head dimension) applied
/// per head BEFORE the rotary embedding, no causal restriction and no mask. Keys/values go through a <see cref="BlockCache"/> after normalization and RoPE.
/// </summary>
public sealed class DitAttention : Module<Tensor, Tensor>
{
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly LlamaRmsNorm q_norm;
    private readonly LlamaRmsNorm k_norm;
    private readonly int heads;
    private readonly int headDim;
    private readonly RotaryEmbedding rope;

    public DitAttention(int dim, int heads, RotaryEmbedding rope) : base(nameof(DitAttention))
    {
        this.heads = heads;
        headDim = dim / heads;
        this.rope = rope;
        q_proj = Linear(dim, dim, hasBias: false);
        k_proj = Linear(dim, dim, hasBias: false);
        v_proj = Linear(dim, dim, hasBias: false);
        o_proj = Linear(dim, dim, hasBias: false);
        q_norm = new LlamaRmsNorm(headDim);
        k_norm = new LlamaRmsNorm(headDim);
        RegisterComponents();
    }

    /// <summary>Self-attention over the block alone (no cache, positions 0..n-1).</summary>
    public override Tensor forward(Tensor x) => Forward(x, null, null, 0, CacheUse.Transient);

    /// <param name="x"><c>[b, n, dim]</c>.</param>
    /// <param name="positions"><c>[b, n]</c> long position ids; null means 0..n-1.</param>
    /// <param name="cache">Null attends to this call's own keys/values only.</param>
    /// <param name="layer">Index of this layer, the slot of <paramref name="cache"/> that holds its keys and values.</param>
    /// <param name="use">How the call reads and writes <paramref name="cache"/>.</param>
    public Tensor Forward(Tensor x, Tensor? positions, BlockCache? cache, int layer, CacheUse use)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], n = x.shape[1];
        Tensor Heads(Tensor y) => y.view(b, n, heads, headDim).transpose(1, 2);   // [b, heads, n, headDim]
        var q = q_norm.forward(Heads(q_proj.forward(x)));
        var k = k_norm.forward(Heads(k_proj.forward(x)));
        var v = Heads(v_proj.forward(x));

        var pos = positions ?? arange(n, dtype: ScalarType.Int64, device: x.device).unsqueeze(0).expand(b, n);
        q = rope.Apply(q, pos, seqDim: 2);
        k = rope.Apply(k, pos, seqDim: 2);

        Tensor attn;
        if (cache is null)
        {
            attn = F.scaled_dot_product_attention(q, k, v);
        }
        else
        {
            var (keys, values) = cache.Update(layer, k, v, use);
            attn = F.scaled_dot_product_attention(q, keys, values);
        }
        return o_proj.forward(attn.transpose(1, 2).contiguous().view(b, n, heads * headDim)).MoveToOuterDisposeScope();
    }
}

/// <summary>One decoder layer: <c>x + attn(input_layernorm(x))</c> then <c>x + mlp(post_attention_layernorm(x))</c>. Field names are the checkpoint's.</summary>
public sealed class DitLayer : Module<Tensor, Tensor>
{
    private readonly LlamaRmsNorm input_layernorm;
    private readonly DitAttention self_attn;
    private readonly LlamaRmsNorm post_attention_layernorm;
    private readonly LlamaMlp mlp;
    private readonly int index;

    public DitLayer(int dim, int heads, int ffMult, int index, RotaryEmbedding rope) : base(nameof(DitLayer))
    {
        this.index = index;
        input_layernorm = new LlamaRmsNorm(dim);
        self_attn = new DitAttention(dim, heads, rope);
        post_attention_layernorm = new LlamaRmsNorm(dim);
        mlp = new LlamaMlp(dim, dim * ffMult);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => Forward(x, null, null, CacheUse.Transient);

    public Tensor Forward(Tensor x, Tensor? positions, BlockCache? cache, CacheUse use)
    {
        using var scope = NewDisposeScope();
        var h = x + self_attn.Forward(input_layernorm.forward(x), positions, cache, index, use);
        return (h + mlp.forward(post_attention_layernorm.forward(h))).MoveToOuterDisposeScope();
    }
}
