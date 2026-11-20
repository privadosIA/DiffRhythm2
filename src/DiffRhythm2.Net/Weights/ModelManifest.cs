namespace DiffRhythm2.Weights;

/// <summary>One file the tools download: where it comes from, what it must be, and where it goes.</summary>
/// <param name="Repository">HuggingFace repository (<c>owner/name</c>).</param>
/// <param name="Revision">The commit it is pinned to.</param>
/// <param name="File">Path of the file in the repository.</param>
/// <param name="Target">Path inside the model directory (checkpoints to convert go under <c>sources/</c>).</param>
/// <param name="Size">Size in bytes.</param>
/// <param name="Sha256">Lower-case hex SHA-256.</param>
public sealed record ManifestFile(string Repository, string Revision, string File, string Target, long Size, string Sha256);

/// <summary>The license of a repository.</summary>
/// <param name="Repository">HuggingFace repository.</param>
/// <param name="License">SPDX-style identifier.</param>
/// <param name="NonCommercial">Whether the license forbids commercial use.</param>
public sealed record RepositoryLicense(string Repository, string License, bool NonCommercial);

/// <summary>The files a complete model directory is made from, pinned to the revisions this port was validated against.</summary>
public static class ModelManifest
{
    /// <summary>DiffRhythm2 itself (the diffusion transformer and the audio decoder).</summary>
    public const string DiffRhythm2Repository = "ASLP-lab/DiffRhythm2";
    /// <summary>MuQ-MuLan, whose text tower is the style encoder.</summary>
    public const string MuLanRepository = "OpenMuQ/MuQ-MuLan-large";
    /// <summary>XLM-RoBERTa, whose tokenizer the style encoder uses.</summary>
    public const string XlmRobertaRepository = "FacebookAI/xlm-roberta-base";

    private const string DiffRhythm2Revision = "9aa15742e4889c0eb2e198db6fdab1facf1b6761";

    /// <summary>The seven files, in download order.</summary>
    public static readonly IReadOnlyList<ManifestFile> Files = new[]
    {
        new ManifestFile(DiffRhythm2Repository, DiffRhythm2Revision, "config.json", "config.json", 262, "606d6392fdd00398d36373574b4990199580ec7dcc0bd82903a1ff00de82c249"),
        new ManifestFile(DiffRhythm2Repository, DiffRhythm2Revision, "decoder.json", "decoder.json", 544, "3d96454939a75879b2a8911826ab7371f91e7245d01de858250e92c90520a3b5"),
        new ManifestFile(DiffRhythm2Repository, DiffRhythm2Revision, "model.safetensors", "model.safetensors", 4545024856, "a3745f77199eef0615e3ec21288663f9c6b3234078b62c9be8ad040061fc921e"),
        new ManifestFile(DiffRhythm2Repository, DiffRhythm2Revision, "decoder.bin", "sources/decoder.bin", 523071584, "9ec403c3e8b9ef59f4adbb6d4e4a930f2a1c2d88ff488329fed74e04f15d1cfb"),
        new ManifestFile(MuLanRepository, "2e01c796b71dca71b45251384c04cd7b237c9020", "pytorch_model.bin", "sources/pytorch_model.bin", 2653954401, "d42ae3f7cb9b66759ee0089ddc70e2f28b130c2d8ba621457358272d32dd0444"),
        new ManifestFile(XlmRobertaRepository, "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", "tokenizer.json", "sources/tokenizer.json", 9096718, "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c"),
    };

    /// <summary>The licenses of the three repositories.</summary>
    public static readonly IReadOnlyList<RepositoryLicense> Licenses = new[]
    {
        new RepositoryLicense(DiffRhythm2Repository, "Apache-2.0", false),
        new RepositoryLicense(XlmRobertaRepository, "MIT", false),
        new RepositoryLicense(MuLanRepository, "CC-BY-NC-4.0", true),
    };

    /// <summary>Total size of <see cref="Files"/> in bytes.</summary>
    public static long TotalBytes => Files.Sum(f => f.Size);

    /// <summary>
    /// The files still worth fetching for a model directory: the three model files always (when missing), and each checkpoint to convert only
    /// if its converted output is absent (the audio tower's too, when <paramref name="audio"/> is set) and the checkpoint cannot be found already (in the directory, in <c>sources</c> or in a HuggingFace cache layout).
    /// </summary>
    public static IReadOnlyList<ManifestFile> Needed(string modelDir, bool audio = false) => Files.Where(f => f.Target switch
    {
        "sources/decoder.bin" => !File.Exists(Path.Combine(modelDir, "decoder.safetensors")) && WeightConverter.FindSource(modelDir, "decoder.bin") is null,
        "sources/pytorch_model.bin" => (!File.Exists(Path.Combine(modelDir, "mulan_text.safetensors")) || (audio && !File.Exists(Path.Combine(modelDir, WeightConverter.AudioFile)))) && WeightConverter.FindSource(modelDir, "pytorch_model.bin") is null,
        "sources/tokenizer.json" => !File.Exists(Path.Combine(modelDir, "mulan_text_tokenizer.json")) && WeightConverter.FindSource(modelDir, "tokenizer.json") is null,
        _ => true,
    }).ToList();
}
