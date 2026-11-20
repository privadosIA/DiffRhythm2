using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace DiffRhythm2.AudioTower;

/// <summary>A feed-forward block of the Conformer: up-projection, swish, down-projection. Field names are the checkpoint's.</summary>
public sealed class ConformerFeedForward : Module<Tensor, Tensor>
{
    private readonly Linear intermediate_dense;
    private readonly Linear output_dense;

    /// <summary>Creates the block.</summary>
    public ConformerFeedForward(MuQConfig c) : base(nameof(ConformerFeedForward))
    {
        intermediate_dense = Linear(c.Hidden, c.Intermediate);
        output_dense = Linear(c.Intermediate, c.Hidden);
        RegisterComponents();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return output_dense.forward(F.silu(intermediate_dense.forward(x))).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// The convolution module: layer norm, pointwise convolution to twice the width, gated linear unit over the channels, depthwise convolution (kernel 31, same
/// padding), batch normalization (stored statistics), swish, pointwise convolution. Field names are the checkpoint's.
/// </summary>
public sealed class ConformerConvModule : Module<Tensor, Tensor>
{
    private readonly LayerNorm layer_norm;
    private readonly Conv1d pointwise_conv1;
    private readonly Conv1d depthwise_conv;
    private readonly BatchNorm1d batch_norm;
    private readonly Conv1d pointwise_conv2;

    /// <summary>Creates the module.</summary>
    public ConformerConvModule(MuQConfig c) : base(nameof(ConformerConvModule))
    {
        layer_norm = LayerNorm(c.Hidden);
        pointwise_conv1 = Conv1d(c.Hidden, 2 * c.Hidden, 1, bias: false);
        depthwise_conv = Conv1d(c.Hidden, c.Hidden, c.ConvKernel, padding: (c.ConvKernel - 1) / 2, groups: c.Hidden, bias: false);
        batch_norm = BatchNorm1d(c.Hidden);
        pointwise_conv2 = Conv1d(c.Hidden, c.Hidden, 1, bias: false);
        RegisterComponents();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = layer_norm.forward(x).transpose(1, 2);                              // [B, C, T]
        h = F.glu(pointwise_conv1.forward(h), 1);
        h = F.silu(batch_norm.forward(depthwise_conv.forward(h)));
        return pointwise_conv2.forward(h).transpose(1, 2).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Multi-head self-attention with rotary positions as the reference's wav2vec2-conformer computes it: the rotation is applied to the <b>input</b> of the query and key
/// projections (not to the projected queries and keys), the value projection sees the unrotated input, the scores are divided by the square root of the head width and
/// there is no mask. Field names are the checkpoint's.
/// </summary>
public sealed class ConformerAttention : Module<Tensor, Tensor, Tensor>
{
    private readonly Linear linear_q;
    private readonly Linear linear_k;
    private readonly Linear linear_v;
    private readonly Linear linear_out;
    private readonly int heads;
    private readonly int headSize;

    /// <summary>Creates the attention.</summary>
    public ConformerAttention(MuQConfig c) : base(nameof(ConformerAttention))
    {
        heads = c.Heads;
        headSize = c.HeadSize;
        linear_q = Linear(c.Hidden, c.Hidden);
        linear_k = Linear(c.Hidden, c.Hidden);
        linear_v = Linear(c.Hidden, c.Hidden);
        linear_out = Linear(c.Hidden, c.Hidden);
        RegisterComponents();
    }

    /// <summary><paramref name="x"/> is <c>[B, T, 1024]</c> and <paramref name="rotary"/> the <c>[2, T, 1, 1, 64]</c> cosine and sine table of <see cref="ConformerRotary"/>.</summary>
    public override Tensor forward(Tensor x, Tensor rotary)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], t = x.shape[1];
        var rotated = Rotate(x, rotary);
        Tensor Split(Tensor y) => y.view(b, t, heads, headSize).transpose(1, 2);   // 'b t (h d) -> b h t d'
        var q = Split(linear_q.forward(rotated));
        var k = Split(linear_k.forward(rotated));
        var v = Split(linear_v.forward(x));
        var scores = matmul(q, k.transpose(-2, -1)) / Math.Sqrt(headSize);
        var probs = scores.softmax(-1);
        var merged = matmul(probs, v).transpose(1, 2).reshape(b, t, heads * headSize);
        return linear_out.forward(merged).MoveToOuterDisposeScope();
    }

    /// <summary><c>x * cos + rotate_half(x) * sin</c> within each head, the way <c>_apply_rotary_embedding</c> does it.</summary>
    private Tensor Rotate(Tensor x, Tensor rotary)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], t = x.shape[1];
        var h = x.view(b, t, heads, headSize).transpose(0, 1);                      // [T, B, H, D]
        var cos = rotary[0];
        var sin = rotary[1];
        long half = headSize / 2;
        var begin = h[.., .., .., ..(int)half];
        var end = h[.., .., .., (int)half..];
        var rot = cat(new[] { -end, begin }, -1);
        var outH = (h * cos + rot * sin).transpose(0, 1);
        return outH.reshape(b, t, heads * headSize).MoveToOuterDisposeScope();
    }
}

/// <summary>The rotary table of the Conformer: <c>inv_freq</c> is a buffer of the checkpoint (32 values for 64-wide heads).</summary>
public sealed class ConformerRotary : Module<Tensor, Tensor>
{
    private readonly Tensor inv_freq;

    /// <summary>Creates the table with the frequencies the reference computes (they are then overwritten by the checkpoint's, which are equal).</summary>
    public ConformerRotary(MuQConfig c) : base(nameof(ConformerRotary))
    {
        int dim = c.HeadSize;
        using var exponent = arange(0, dim, 2, dtype: ScalarType.Int64).to(ScalarType.Float32) / dim;
        inv_freq = full_like(exponent, c.RotaryBase).pow(exponent).reciprocal();
        register_buffer("inv_freq", inv_freq);
    }

    /// <summary>The table for a sequence of <paramref name="length"/> steps: <c>[2, length, 1, 1, 64]</c> (cosines, then sines).</summary>
    public Tensor Table(int length)
    {
        using var scope = NewDisposeScope();
        var stamps = arange(length, dtype: ScalarType.Float32, device: inv_freq.device);
        var freqs = stamps.unsqueeze(1) * inv_freq.unsqueeze(0);                    // outer product, 'i,j->ij'
        var emb = cat(new[] { freqs, freqs }, -1);
        var cos = emb.cos().unsqueeze(1).unsqueeze(1);
        var sin = emb.sin().unsqueeze(1).unsqueeze(1);
        return stack(new[] { cos, sin }).MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor x) => Table((int)x.shape[1]);
}

/// <summary>One Conformer layer: half feed-forward, attention, convolution module, half feed-forward, final layer norm. Field names are the checkpoint's.</summary>
public sealed class ConformerLayer : Module<Tensor, Tensor, Tensor>
{
    private readonly LayerNorm ffn1_layer_norm;
    private readonly ConformerFeedForward ffn1;
    private readonly LayerNorm self_attn_layer_norm;
    private readonly ConformerAttention self_attn;
    private readonly ConformerConvModule conv_module;
    private readonly LayerNorm ffn2_layer_norm;
    private readonly ConformerFeedForward ffn2;
    private readonly LayerNorm final_layer_norm;

    /// <summary>Creates the layer.</summary>
    public ConformerLayer(MuQConfig c) : base(nameof(ConformerLayer))
    {
        ffn1_layer_norm = LayerNorm(c.Hidden);
        ffn1 = new ConformerFeedForward(c);
        self_attn_layer_norm = LayerNorm(c.Hidden);
        self_attn = new ConformerAttention(c);
        conv_module = new ConformerConvModule(c);
        ffn2_layer_norm = LayerNorm(c.Hidden);
        ffn2 = new ConformerFeedForward(c);
        final_layer_norm = LayerNorm(c.Hidden);
        RegisterComponents();
    }

    /// <inheritdoc/>
    public override Tensor forward(Tensor x, Tensor rotary)
    {
        using var scope = NewDisposeScope();
        var h = ffn1.forward(ffn1_layer_norm.forward(x)) * 0.5 + x;
        h = self_attn.forward(self_attn_layer_norm.forward(h), rotary) + h;
        h = conv_module.forward(h) + h;
        h = ffn2.forward(ffn2_layer_norm.forward(h)) * 0.5 + h;
        return final_layer_norm.forward(h).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// The Conformer encoder: the rotary table, the layers and a final layer norm. The convolutional positional embedding whose tensors are in the checkpoint is
/// constructed by the reference but never called, so it does not exist here. Field names are the checkpoint's.
/// </summary>
public sealed class ConformerEncoder : Module<Tensor, Tensor>
{
    private readonly ConformerRotary embed_positions;
    private readonly LayerNorm layer_norm;
    private readonly ModuleList<ConformerLayer> layers;

    /// <summary>Sizes of the encoder.</summary>
    public MuQConfig Config { get; }

    /// <summary>Creates the encoder.</summary>
    public ConformerEncoder(MuQConfig c) : base(nameof(ConformerEncoder))
    {
        Config = c;
        embed_positions = new ConformerRotary(c);
        layer_norm = LayerNorm(c.Hidden);
        layers = ModuleList(Enumerable.Range(0, c.Layers).Select(_ => new ConformerLayer(c)).ToArray());
        RegisterComponents();
    }

    /// <summary>The rotary table for a sequence of <paramref name="length"/> steps.</summary>
    public Tensor Rotary(int length) => embed_positions.Table(length);

    /// <summary>Layer <paramref name="index"/> (0 based) applied to a hidden state, with the table of that sequence length (stage fixture).</summary>
    public Tensor Layer(int index, Tensor x, Tensor rotary) => layers[index].forward(x, rotary);

    /// <summary>The encoder's last layer norm (stage fixture).</summary>
    public Tensor FinalNorm(Tensor x) => layer_norm.forward(x);

    /// <summary>The sequence <c>[B, T, 1024]</c> through every layer and the final layer norm: the last hidden state of the reference.</summary>
    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var rotary = embed_positions.Table((int)x.shape[1]);
        for (int i = 0; i < layers.Count; i++) x = layers[i].forward(x, rotary);
        return layer_norm.forward(x).MoveToOuterDisposeScope();
    }
}
