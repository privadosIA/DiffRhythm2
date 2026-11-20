using System.Text.Json;

namespace DiffRhythm2.Model;

/// <summary>
/// Dimensions of the DiT, read from the checkpoint's <c>config.json</c>. Not in the file because the reference hard-codes it: the conditioning width
/// (<see cref="CondDim"/> = 512, <c>cond_dim</c> in <c>DiT.__init__</c>; the <c>text_dim</c> key of the config is not used by the model). Keys that
/// only matter for training or for another attention backend (<c>use_flex_attn</c>, <c>repa_*</c>, <c>conv_layers</c>) are ignored.
/// </summary>
public sealed record DitConfig(int Dim, int Depth, int Heads, int FfMult, int MelDim, int TextNumEmbeds, int BlockSize)
{
    public const int CondDim = 512;
    public const int FreqEmbedDim = 256;

    /// <summary>The reference builds <c>LlamaConfig(max_position_embeddings=4096)</c> and a rotary table with <c>rope_theta</c> 10000.</summary>
    public const int MaxPositions = 4096;
    public const double RopeTheta = 10000;

    public int HeadDim => Dim / Heads;

    public static DitConfig Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        int Get(string name) => root.TryGetProperty(name, out var v) && v.TryGetInt32(out var i)
            ? i
            : throw new InvalidDataException($"{path}: the key '{name}' is missing or not an integer");
        var c = new DitConfig(Get("dim"), Get("depth"), Get("heads"), Get("ff_mult"), Get("mel_dim"), Get("text_num_embeds"), Get("block_size"));
        if (c.Dim % c.Heads != 0 || c.HeadDim % 2 != 0)
            throw new InvalidDataException($"{path}: dim {c.Dim} must split into {c.Heads} heads of an even size");
        return c;
    }
}
