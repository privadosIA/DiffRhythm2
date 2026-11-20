using DiffRhythm2.StyleText;
using MusicGen.Core.Testing;
using MusicGen.Core.Weights;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The text tower on a tiny synthetic checkpoint: names, strict loading, positions, batch refusal. No checkpoint needed.</summary>
public class MuLanTextModelTests : IDisposable
{
    private static readonly XlmRobertaConfig TinyEncoder = new(Vocabulary: 30, Hidden: 16, Layers: 2, Heads: 2, Intermediate: 32, MaxPositions: 12);
    private static readonly MuLanTransformerConfig TinyTransformer = new(Dim: 16, Depth: 2, Heads: 2, DimHead: 4, FfMult: 4);
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-mulan-").FullName;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    private static MuLanTextTower Tiny() => new(TinyEncoder, TinyTransformer, latentDim: 8);

    private string WriteCheckpoint(Action<Dictionary<string, Tensor>>? edit = null)
    {
        using var _ = no_grad();
        manual_seed(9);
        var source = Tiny();
        var file = source.state_dict().ToDictionary(kv => kv.Key, kv => kv.Value.detach().clone());
        foreach (var t in file.Values) t.normal_(0, 0.3);
        file["text.model.pooler.dense.weight"] = randn(16, 16);
        file["text.model.pooler.dense.bias"] = randn(16);
        edit?.Invoke(file);
        var path = Path.Combine(tmp, MuLanTextWeights.CheckpointFile);
        SafeTensorsWriter.Save(path, file);
        return path;
    }

    private Exception LoadFails(Action<Dictionary<string, Tensor>> edit) =>
        Assert.Throws<InvalidDataException>(() => MuLanTextWeights.Load(Tiny(), CheckpointIndex.Open(WriteCheckpoint(edit)), CPU));

    [Fact]
    public void ModelKeys_AreTheCheckpointKeys()
    {
        var keys = Tiny().state_dict().Keys.ToHashSet();
        foreach (var expected in new[]
        {
            "text.model.embeddings.word_embeddings.weight", "text.model.embeddings.position_embeddings.weight", "text.model.embeddings.token_type_embeddings.weight",
            "text.model.embeddings.LayerNorm.weight", "text.model.embeddings.LayerNorm.bias",
            "text.model.encoder.layer.1.attention.self.query.weight", "text.model.encoder.layer.1.attention.self.value.bias",
            "text.model.encoder.layer.0.attention.output.dense.weight", "text.model.encoder.layer.0.attention.output.LayerNorm.bias",
            "text.model.encoder.layer.0.intermediate.dense.weight", "text.model.encoder.layer.0.output.dense.bias", "text.model.encoder.layer.1.output.LayerNorm.weight",
            "text.transformer.layers.0.0.q_scale", "text.transformer.layers.0.0.k_scale", "text.transformer.layers.0.0.norm.learned_gamma",
            "text.transformer.layers.0.0.to_q.weight", "text.transformer.layers.1.0.to_kv.weight", "text.transformer.layers.1.0.to_out.0.weight",
            "text.transformer.layers.0.1.0.learned_gamma", "text.transformer.layers.0.1.1.weight", "text.transformer.layers.1.1.4.weight",
            "text_to_latents.weight", "text_to_latents.bias",
        })
            Assert.Contains(expected, keys);
        Assert.DoesNotContain(keys, k => k.Contains("pooler"));
    }

    [Fact]
    public void Load_MatchingCheckpoint_AssignsEveryTensor_AndReportsThePooler()
    {
        var path = WriteCheckpoint();
        var model = Tiny();
        var report = MuLanTextWeights.Load(model, CheckpointIndex.Open(path), CPU);
        Assert.Equal(new[] { "text.model.pooler.dense.bias", "text.model.pooler.dense.weight" }, report.IgnoredKeys);
        Assert.Equal(model.state_dict().Count, report.Result.TensorsRead);
        var file = SafeTensors.LoadFile(path);
        foreach (var (name, value) in model.state_dict())
            Assert.True((value - file[name]).abs().max().item<float>() == 0f, $"{name} was not loaded");
    }

    [Fact]
    public void Load_MissingTensor_NamesIt()
    {
        var e = LoadFails(f => f.Remove("text.transformer.layers.1.1.4.weight"));
        Assert.Contains("text.transformer.layers.1.1.4.weight", e.Message);
        Assert.Contains("missing", e.Message);
    }

    [Fact]
    public void Load_ShapeMismatch_NamesTheTensorWithBothShapes()
    {
        var e = LoadFails(f => f["text_to_latents.weight"] = zeros(3, 5));
        Assert.Contains("text_to_latents.weight", e.Message);
        Assert.Contains("[8, 16]", e.Message);
        Assert.Contains("[3, 5]", e.Message);
    }

    [Fact]
    public void Load_UnexpectedTensor_IsRejected()
    {
        var e = LoadFails(f => f["text.model.surprise.weight"] = zeros(2));
        Assert.Contains("surprise.weight", e.Message);
    }

    [Fact]
    public void PositionIds_StartAfterThePaddingIndex_AndPaddingKeepsIt()
    {
        var emb = new XlmRobertaModel(TinyEncoder);
        // the embedding block is private to the model; check the rule through the module that owns it
        var embeddings = new XlmRobertaEmbeddings(TinyEncoder);
        using var plain = embeddings.PositionIds(tensor(new long[] { 0, 5, 6, 2 }).unsqueeze(0));
        Assert.Equal(new long[] { 2, 3, 4, 5 }, plain.data<long>().ToArray());
        using var padded = embeddings.PositionIds(tensor(new long[] { 0, 5, 1, 1 }).unsqueeze(0));
        Assert.Equal(new long[] { 2, 3, 1, 1 }, padded.data<long>().ToArray());
        Assert.Equal(10, TinyEncoder.MaxTokens);   // 12 position rows minus the padding offset
        Assert.Equal(512, new XlmRobertaConfig().MaxTokens);
    }

    [Fact]
    public void Forward_GivesAUnitEmbedding_AndRefusesABatch()
    {
        using var _ = no_grad();
        var model = Tiny();
        foreach (var p in model.parameters()) p.normal_(0, 0.3);
        using var one = model.forward(tensor(new long[] { 0, 7, 9, 2 }).unsqueeze(0));
        Assert.Equal(new long[] { 1, 8 }, one.shape);
        Assert.Equal(1.0, one.norm().item<float>(), 5);

        var e = Assert.Throws<NotSupportedException>(() => model.forward(tensor(new long[] { 0, 7, 9, 2, 0, 7, 9, 2 }).view(2, 4)));
        Assert.Contains("only one text", e.Message);
        Assert.Throws<ArgumentException>(() => model.forward(zeros(1, 11, dtype: ScalarType.Int64)));   // over the tiny encoder's 10 positions
    }
}

/// <summary>The real tower against the reference: strict load of the converted checkpoint, every stage, the embeddings, the limits and the hand-off to the sampler.</summary>
[Collection(RealMulanCollection.Name)]
public class MuLanTextParityTests
{
    private readonly RealMulanFixture fx;
    private readonly ITestOutputHelper log;

    public MuLanTextParityTests(RealMulanFixture fx, ITestOutputHelper log)
    {
        this.fx = fx;
        this.log = log;
    }

    private StyleEncoder Encoder => fx.Encoder!;

    private static Tensor Stage(int prompt, string name) => MulanTextReference.Tower[$"p{prompt}_{name}"];

    private static void AssertClose(ITestOutputHelper log, string what, Tensor got, Tensor expected, double relTol)
    {
        var r = Parity.Compare(got, expected);
        double scale = expected.abs().max().item<float>();
        double mean = (got.to(ScalarType.Float64) - expected.to(ScalarType.Float64)).abs().mean().item<double>();
        log.WriteLine($"{what}: max_abs={r.MaxAbs:E3} mean_abs={mean:E3} scale={scale:E3} rel={r.MaxAbs / scale:E3} cos={r.CosSim:F9}");
        Assert.Equal(expected.shape, got.shape);
        Assert.True(r.MaxAbs <= relTol * scale, $"{what}: max_abs {r.MaxAbs:E3} exceeds {relTol:E1} x scale {scale:E3}");
    }

    [RequiresMulanText]
    public void Load_ReadsEveryTowerTensor_AndIgnoresThePooler()
    {
        var report = Encoder.Report!;
        log.WriteLine($"read {report.Result.TensorsRead} tensors, ignored {string.Join(", ", report.IgnoredKeys)}");
        Assert.Equal(271, report.Result.TensorsRead);
        Assert.Equal(new[] { "text.model.pooler.dense.bias", "text.model.pooler.dense.weight" }, report.IgnoredKeys);
    }

    public static IEnumerable<object[]> PromptIndexes => Enumerable.Range(0, 7).Select(i => new object[] { i });

    [RequiresMulanText]
    public void Stages_MatchTheReference_OnEveryPrompt()
    {
        using var _ = no_grad();
        var tower = Encoder.Tower;
        var encoder = tower.Text.Encoder;
        var transformer = tower.Text.Transformer;
        var prompts = MulanTextReference.Prompts;
        for (int i = 0; i < prompts.Length; i++)
        {
            var ids = Encoder.Tokenizer.Encode(prompts[i]);
            Assert.Equal(Stage(i, "ids").data<long>().ToArray(), ids.Select(x => (long)x).ToArray());
            using var idTensor = tensor(ids.Select(x => (long)x).ToArray()).unsqueeze(0);

            // each stage is fed the reference's previous stage, so an error is attributed to the stage that introduced it
            using var emb = encoder.Embed(idTensor);
            AssertClose(log, $"prompt {i} embeddings", emb[0], Stage(i, "embeddings"), Tolerances.Tower);
            using var l0 = encoder.Layer(0, Stage(i, "embeddings").unsqueeze(0));
            AssertClose(log, $"prompt {i} encoder layer 0", l0[0], Stage(i, "l0"), Tolerances.Tower);
            using var enc = encoder.forward(idTensor);
            AssertClose(log, $"prompt {i} encoder output", enc[0], Stage(i, "enc"), Tolerances.Tower);
            using var t0 = transformer.Layer(0, Stage(i, "enc").unsqueeze(0));
            AssertClose(log, $"prompt {i} transformer layer 0", t0[0], Stage(i, "t0"), Tolerances.Tower);
            using var tr = transformer.forward(Stage(i, "enc").unsqueeze(0));
            AssertClose(log, $"prompt {i} transformer output", tr[0], Stage(i, "tr"), Tolerances.Tower);
            using var mean = MuLanTextTower.Mean(Stage(i, "tr").unsqueeze(0));
            AssertClose(log, $"prompt {i} mean", mean[0], Stage(i, "mean"), Tolerances.Tower);
            using var final = tower.Project(Stage(i, "mean").unsqueeze(0));
            AssertClose(log, $"prompt {i} projection", final[0], Stage(i, "emb"), Tolerances.Tower);
        }
    }

    [RequiresMulanText]
    public void Embeddings_MatchTheReference_EndToEnd()
    {
        var prompts = MulanTextReference.Prompts;
        for (int i = 0; i < prompts.Length; i++)
        {
            using var emb = Encoder.Embed(prompts[i]);
            Assert.Equal(new long[] { 1, 512 }, emb.shape);
            Assert.Equal(ScalarType.Float32, emb.dtype);
            Assert.Equal(1.0, emb.norm().item<float>(), 5);
            AssertClose(log, $"prompt {i} embedding", emb[0], Stage(i, "emb"), Tolerances.Embedding);
            double cos = (emb[0].to(ScalarType.Float64) * Stage(i, "emb").to(ScalarType.Float64)).sum().item<double>();   // in double: a float32 sum of 512 products cannot show better than 1 - 1.2e-7
            Assert.True(cos > Tolerances.EmbeddingCosine, $"prompt {i}: cosine {cos:F9}");
        }
    }

    [RequiresMulanText]
    public void LongestText_512Positions_MatchesTheReference()
    {
        var reference = Directory.GetFiles(MulanTextReference.Directory.Dir, "max_length.safetensors").Single();
        var max = SafeTensors.LoadFile(reference);
        using var emb = Encoder.Embed(string.Concat(Enumerable.Repeat("a ", 510)));
        Assert.Equal(512, max["ids"].shape[0]);
        AssertClose(log, "512-position embedding", emb[0], max["emb"], Tolerances.Embedding);
    }

    [RequiresMulanText]
    public void TooLongText_IsRefusedWithBothNumbers()
    {
        var e = Assert.Throws<ArgumentException>(() => Encoder.Embed(string.Join(" ", Enumerable.Repeat("a", 598))));
        Assert.Contains("600", e.Message);
        Assert.Contains("512", e.Message);
    }

    [RequiresMulanText]
    public void SeveralTexts_AreRefused()
    {
        var e = Assert.Throws<NotSupportedException>(() => Encoder.Embed(new[] { "Pop", "Rock" }));
        Assert.Contains("only one text", e.Message);
        using var one = Encoder.Embed(new[] { "Pop" });
        Assert.Equal(new long[] { 1, 512 }, one.shape);
    }

    [RequiresMulanText]
    public void Performance_IsReported()
    {
        using var process = System.Diagnostics.Process.GetCurrentProcess();
        double before = process.WorkingSet64 / (1024.0 * 1024 * 1024);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using (var second = StyleEncoder.LoadDirectory(Checkpoint.Dir!, CPU))
        {
            double loadSeconds = sw.Elapsed.TotalSeconds;
            process.Refresh();
            double after = process.WorkingSet64 / (1024.0 * 1024 * 1024);
            using var warm = second.Embed("Pop, Piano, Female vocals, Happy, Bright");
            sw.Restart();
            for (int i = 0; i < 20; i++) { using var e = second.Embed("Pop, Piano, Female vocals, Happy, Bright"); }
            double perEmbedMs = sw.Elapsed.TotalMilliseconds / 20;
            log.WriteLine($"load {loadSeconds:F1} s; resident {before:F2} -> {after:F2} GiB (+{after - before:F2}); one 13-token embedding {perEmbedMs:F0} ms (CPU, fp32)");
            Assert.True(perEmbedMs > 0, "the timing was not measured");   // the figures are only reported
        }
    }

    [RequiresMulanText]
    public void Embedding_FeedsTheSampler_AsItsStyle()
    {
        manual_seed(1);
        using var model = new DiffRhythm2.Model.Dit(new DiffRhythm2.Model.DitConfig(Dim: 32, Depth: 2, Heads: 2, FfMult: 2, MelDim: 8, TextNumEmbeds: 1000, BlockSize: 10));
        using var style = Encoder.Embed("Pop, Piano, Female vocals, Happy, Bright");
        using var tokens = tensor(new long[] { 500, 511, 503, 511 });
        var noise = new List<Tensor> { randn(1, 10, 8) };
        var sampler = new DiffRhythm2.Sampling.BlockSampler(model);
        using var latent = sampler.Sample(tokens, style, noise, frames: 10, new DiffRhythm2.Sampling.SamplerOptions(Steps: 2, CfgStrength: 2.0));
        Assert.Equal(new long[] { 1, 10, 8 }, latent.shape);
        Assert.True(latent.isfinite().all().item<bool>());
        using var half = style.to(ScalarType.Float16);       // what the service does before a half-precision run
        Assert.Equal(new long[] { 1, 512 }, half.shape);
    }
}
