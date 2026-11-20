namespace DiffRhythm2.Pipeline;

/// <summary>One expected file of the model directory.</summary>
/// <param name="Name">File name.</param>
/// <param name="Length">Size in bytes, or null when the file is absent.</param>
public sealed record ModelFile(string Name, long? Length)
{
    /// <summary>Whether the file exists.</summary>
    public bool Present => Length is not null;
}

/// <summary>
/// The directory the pipeline reads its models from: <c>config.json</c> and <c>model.safetensors</c> (the diffusion transformer),
/// <c>decoder.json</c> and <c>decoder.safetensors</c> (the audio decoder), <c>mulan_text.safetensors</c> and <c>mulan_text_tokenizer.json</c> (the style encoder).
/// </summary>
public sealed class ModelDirectory
{
    /// <summary>Configuration of the diffusion transformer.</summary>
    public const string Config = "config.json";
    /// <summary>Weights of the diffusion transformer.</summary>
    public const string Dit = "model.safetensors";
    /// <summary>Configuration of the audio decoder.</summary>
    public const string DecoderConfig = "decoder.json";
    /// <summary>Weights of the audio decoder.</summary>
    public const string Decoder = "decoder.safetensors";
    /// <summary>Weights of the style text encoder.</summary>
    public const string StyleWeights = "mulan_text.safetensors";
    /// <summary>Tokenizer of the style text encoder.</summary>
    public const string StyleTokenizer = "mulan_text_tokenizer.json";

    /// <summary>The names of the six files a model directory must hold.</summary>
    public static readonly IReadOnlyList<string> RequiredFiles = new[] { Config, Dit, DecoderConfig, Decoder, StyleWeights, StyleTokenizer };

    /// <summary>The directory.</summary>
    public string Path { get; }

    /// <summary>Points at a directory; nothing is read until a member is used.</summary>
    public ModelDirectory(string path) => Path = path;

    /// <summary>Every expected file with its size, present or not.</summary>
    public IReadOnlyList<ModelFile> Files => RequiredFiles
        .Select(name => new ModelFile(name, File.Exists(System.IO.Path.Combine(Path, name)) ? new FileInfo(System.IO.Path.Combine(Path, name)).Length : null))
        .ToList();

    /// <summary>The names of the expected files that are not there.</summary>
    public IReadOnlyList<string> Missing => Files.Where(f => !f.Present).Select(f => f.Name).ToList();

    /// <summary>The full path of an expected file.</summary>
    public string Of(string name) => System.IO.Path.Combine(Path, name);

    /// <summary>Throws one <see cref="ModelDirectoryException"/> listing every missing file (or saying the directory does not exist).</summary>
    public void EnsureComplete()
    {
        if (!Directory.Exists(Path)) throw new ModelDirectoryException(Path, RequiredFiles);
        var missing = Missing;
        if (missing.Count > 0) throw new ModelDirectoryException(Path, missing);
    }
}
