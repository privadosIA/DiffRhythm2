using TorchSharp;
using static TorchSharp.torch;
using MusicGen.Core.Weights;

namespace DiffRhythm2.StyleText;

/// <summary>
/// A style text to the 512-dimensional embedding that conditions the DiT: the XLM-RoBERTa tokenizer and the text tower of MuQ-MuLan, in the
/// precision of the model (fp32, as the service runs it). One text per call. A caller that runs the sampler in half precision casts the result,
/// as the service does.
/// </summary>
public sealed class StyleEncoder : IDisposable
{
    private readonly XlmrTokenizer tokenizer;
    private readonly MuLanTextTower tower;

    public XlmrTokenizer Tokenizer => tokenizer;
    public MuLanTextTower Tower => tower;
    public MuLanLoadReport? Report { get; }

    public StyleEncoder(XlmrTokenizer tokenizer, MuLanTextTower tower, MuLanLoadReport? report = null)
    {
        this.tokenizer = tokenizer;
        this.tower = tower;
        Report = report;
    }

    /// <summary>Loads <c>mulan_text.safetensors</c> and <c>mulan_text_tokenizer.json</c> from a directory.</summary>
    public static StyleEncoder LoadDirectory(string dir, Device device)
    {
        var tokenizer = XlmrTokenizer.Load(Path.Combine(dir, MuLanTextWeights.TokenizerFile));
        var tower = new MuLanTextTower();
        var report = MuLanTextWeights.Load(tower, CheckpointIndex.Open(Path.Combine(dir, MuLanTextWeights.CheckpointFile)), device);
        return new StyleEncoder(tokenizer, tower, report);
    }

    /// <summary>The embedding <c>[1, 512]</c> (float32, unit length) of one text.</summary>
    public Tensor Embed(string text)
    {
        var ids = tokenizer.Encode(text);
        int limit = tower.Text.Encoder.Config.MaxTokens;
        if (ids.Length > limit) throw new ArgumentException($"the text has {ids.Length} positions (the markers included) but the encoder supports at most {limit}");
        using var _ = no_grad();
        using var tensor = torch.tensor(ids.Select(i => (long)i).ToArray(), new long[] { 1, ids.Length }, ScalarType.Int64, tower.parameters().First().device);
        return tower.forward(tensor);
    }

    /// <summary>Several texts at once are not supported: the reference averages over padded positions, so a batch would give different results.</summary>
    public Tensor Embed(IReadOnlyList<string> texts) =>
        texts.Count == 1 ? Embed(texts[0]) : throw new NotSupportedException($"only one text per call is supported (got {texts.Count}); the reference averages over padded positions, so a batch would give different results");

    public void Dispose() => tower.Dispose();
}
