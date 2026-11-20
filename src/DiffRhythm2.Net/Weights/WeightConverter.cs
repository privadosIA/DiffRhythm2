using System.Security.Cryptography;

namespace DiffRhythm2.Weights;

/// <summary>What a conversion did with one output file.</summary>
public enum ConversionStatus
{
    /// <summary>The file was written.</summary>
    Written,
    /// <summary>A valid file was already there and nothing was rewritten.</summary>
    UpToDate,
}

/// <summary>The result for one output file of a conversion.</summary>
/// <param name="Output">File name inside the model directory.</param>
/// <param name="Status">Written or already up to date.</param>
/// <param name="Source">The source file it was made from, or null when it was up to date and the source was not needed.</param>
public sealed record ConversionResult(string Output, ConversionStatus Status, string? Source);

/// <summary>Options of <see cref="WeightConverter.Convert"/>.</summary>
/// <param name="From">A directory to look for the sources in before the usual places, or null.</param>
/// <param name="Force">Rewrite outputs even if valid ones exist.</param>
/// <param name="DeleteSources">Delete the converted source checkpoints once every output is written and verified.</param>
/// <param name="Audio">Also convert the audio tower of MuQ-MuLan (<c>mulan_audio.safetensors</c>, 1.3 GB), which making songs does not need.</param>
public sealed record ConversionOptions(string? From = null, bool Force = false, bool DeleteSources = false, bool Audio = false);

/// <summary>A conversion that cannot proceed: a source is missing, or has not the expected content.</summary>
public sealed class ConversionException : IOException
{
    /// <summary>Creates the exception.</summary>
    public ConversionException(string message) : base(message) { }
}

/// <summary>
/// Turns the published checkpoints (pickled <c>torch.save</c> files) into the safetensors files the pipeline loads, without Python:
/// <c>decoder.bin</c> to <c>decoder.safetensors</c>, and MuQ-MuLan's <c>pytorch_model.bin</c> to <c>mulan_text.safetensors</c> plus the tokenizer copy and,
/// on request, <c>mulan_audio.safetensors</c>. The outputs are byte for byte what the reference Python tools write.
/// </summary>
public static class WeightConverter
{
    /// <param name="Output">File name of the converted checkpoint.</param>
    /// <param name="Source">File name of the pickled checkpoint it is made from.</param>
    /// <param name="StateKey">Key of the state dictionary inside the pickle, or null when the pickle is the dictionary.</param>
    /// <param name="Map">Output name of a source tensor, or null when the tensor is not kept.</param>
    /// <param name="Expected">Exactly how many tensors are kept.</param>
    /// <param name="Describe">What the file is, for messages.</param>
    /// <param name="RequiredDType">The safetensors dtype a tensor must have, by its output name; null means float32 for every tensor.</param>
    /// <param name="ExpectedInt64">How many tensors <paramref name="RequiredDType"/> asks to be int64 (the batch-norm counters of the audio tower).</param>
    internal sealed record Profile(string Output, string Source, string? StateKey, Func<string, string?> Map, int Expected, string Describe,
        Func<string, string>? RequiredDType = null, int ExpectedInt64 = 0)
    {
        internal string DTypeOf(string name) => RequiredDType?.Invoke(name) ?? "F32";
    }

    internal static readonly Profile Decoder = new(ModelDirectory_Decoder, "decoder.bin", "generator", Rename, 783, "the decoder");

    internal static readonly Profile Mulan = new(ModelDirectory_Style, "pytorch_model.bin", null, name =>
        name.StartsWith("mulan.text.", StringComparison.Ordinal) || name.StartsWith("mulan.text_to_latents.", StringComparison.Ordinal) ? name["mulan.".Length..] : null,
        273, "the style encoder");

    /// <summary>
    /// The audio tower of MuQ-MuLan: the 525 tensors of MuQ-large (<c>audio.model.model.*</c>, of which 18 are the int64 <c>num_batches_tracked</c> counters of the
    /// batch normalizations), the audio projection and layer aggregator (<c>audio.proj.*</c>, <c>audio.aggregator.*</c>) and the projection to the latent space.
    /// They come from the MuLan pickle, not from <c>MuQ-large-msd-iter</c>: MuLan fine-tuned MuQ-large, and only 69 of its 525 tensors are still equal to that repository's.
    /// </summary>
    internal static readonly Profile Audio = new(AudioFile, "pytorch_model.bin", null, name =>
        name.StartsWith("mulan.audio.", StringComparison.Ordinal) || name.StartsWith("mulan.audio_to_latents.", StringComparison.Ordinal) ? name["mulan.".Length..] : null,
        531, "the audio tower", name => name.EndsWith(".num_batches_tracked", StringComparison.Ordinal) ? "I64" : "F32", ExpectedInt64: 18);

    /// <summary>Name of the converted audio tower inside the model directory.</summary>
    public const string AudioFile = "mulan_audio.safetensors";

    private const string ModelDirectory_Decoder = "decoder.safetensors";
    private const string ModelDirectory_Style = "mulan_text.safetensors";
    private const string ModelDirectory_Tokenizer = "mulan_text_tokenizer.json";
    private const string TokenizerSource = "tokenizer.json";

    private static string? Rename(string name)
    {
        foreach (var (old, @new) in new[] { (".weight_g", ".parametrizations.weight.original0"), (".weight_v", ".parametrizations.weight.original1") })
            if (name.EndsWith(old, StringComparison.Ordinal)) return name[..^old.Length] + @new;
        return name;
    }

    /// <summary>The places looked at for a source, in order, for the error message and for the search itself.</summary>
    public static IReadOnlyList<string> SearchPlaces(string modelDir, string file, string? from) => Candidates(modelDir, file, from).ToList();

    private static readonly Dictionary<string, string[]> CacheRepos = new()
    {
        ["decoder.bin"] = new[] { "models--ASLP-lab--DiffRhythm2" },
        ["pytorch_model.bin"] = new[] { "models--OpenMuQ--MuQ-MuLan-large" },
        [TokenizerSource] = new[] { "models--xlm-roberta-base", "models--FacebookAI--xlm-roberta-base" },
    };

    private static IEnumerable<string> Candidates(string modelDir, string file, string? from)
    {
        var roots = new List<string>();
        if (from is not null) roots.Add(from);
        roots.Add(modelDir);
        roots.Add(Path.Combine(modelDir, "sources"));
        foreach (var r in roots) yield return Path.Combine(r, file);
        foreach (var r in roots.Take(from is null ? 1 : 2))
            foreach (var repo in CacheRepos.GetValueOrDefault(file) ?? Array.Empty<string>())
                yield return Path.Combine(r, repo, "snapshots", "*", file);
    }

    /// <summary>Finds a source file, or returns null.</summary>
    public static string? FindSource(string modelDir, string file, string? from = null)
    {
        foreach (var c in Candidates(modelDir, file, from))
        {
            if (c.Contains('*'))
            {
                var snapshots = Path.GetDirectoryName(Path.GetDirectoryName(c))!;
                if (!Directory.Exists(snapshots)) continue;
                foreach (var dir in Directory.GetDirectories(snapshots).OrderBy(d => d, StringComparer.Ordinal))
                    if (File.Exists(Path.Combine(dir, file))) return Path.Combine(dir, file);
            }
            else if (File.Exists(c)) return c;
        }
        return null;
    }

    private static string Locate(string modelDir, string file, string? from) =>
        FindSource(modelDir, file, from)
        ?? throw new ConversionException($"{file} was not found; looked in:\n  " + string.Join("\n  ", Candidates(modelDir, file, from)));

    /// <summary>
    /// Produces the three converted files in <paramref name="modelDir"/> (and the audio tower when <see cref="ConversionOptions.Audio"/> is set). <paramref name="step"/> receives the name of each output as work on it starts.
    /// Sources are only read, unless <see cref="ConversionOptions.DeleteSources"/> is set. Outputs that are already valid are left alone unless forced.
    /// </summary>
    public static IReadOnlyList<ConversionResult> Convert(string modelDir, ConversionOptions? options = null, Action<string>? step = null, CancellationToken cancellation = default)
    {
        options ??= new ConversionOptions();
        Directory.CreateDirectory(modelDir);
        var results = new List<ConversionResult>();
        var used = new List<string>();

        foreach (var p in new[] { Decoder, Mulan })
        {
            step?.Invoke(p.Output);
            var r = ConvertTensors(modelDir, p, options, used, cancellation);
            results.Add(r);
        }

        step?.Invoke(ModelDirectoryTokenizerName);
        results.Add(CopyTokenizer(modelDir, options, used));

        if (options.Audio)
        {
            step?.Invoke(Audio.Output);
            results.Add(ConvertTensors(modelDir, Audio, options, used, cancellation));
        }

        if (options.DeleteSources) foreach (var f in used.Distinct()) File.Delete(f);
        return results;
    }

    private static string ModelDirectoryTokenizerName => ModelDirectory_Tokenizer;

    internal static ConversionResult ConvertTensors(string modelDir, Profile p, ConversionOptions options, List<string> used, CancellationToken cancellation)
    {
        var output = Path.Combine(modelDir, p.Output);
        bool upToDate = !options.Force && IsValid(output, p);
        if (upToDate && !options.DeleteSources) return new ConversionResult(p.Output, ConversionStatus.UpToDate, null);

        var source = Locate(modelDir, p.Source, options.From);
        using var archive = new TorchArchive(source, p.StateKey);
        var plan = new List<(TorchTensorInfo Tensor, string Name)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var t in archive.Tensors)
        {
            var name = p.Map(t.Name);
            if (name is null) continue;
            if (t.DType.SafeTensorsName() != p.DTypeOf(name)) throw new ConversionException($"{p.Describe}: tensor '{t.Name}' is {t.DType.SafeTensorsName()}, expected {p.DTypeOf(name)}; nothing was written");
            if (!seen.Add(name)) throw new ConversionException($"{p.Describe}: two tensors would be called '{name}' after renaming; nothing was written");
            plan.Add((t, name));
        }
        if (plan.Count != p.Expected)
            throw new ConversionException($"{p.Describe}: {source} holds {plan.Count} tensors to convert, {p.Expected} were expected; it is not the checkpoint this tool was written for. Nothing was written");
        int counters = plan.Count(x => p.DTypeOf(x.Name) == "I64");
        if (counters != p.ExpectedInt64)
            throw new ConversionException($"{p.Describe}: {source} holds {counters} int64 counters, {p.ExpectedInt64} were expected; it is not the checkpoint this tool was written for. Nothing was written");
        foreach (var (t, _) in plan) archive.EnsureReadable(t);

        if (!upToDate)
        {
            var tmp = output + ".tmp";
            try
            {
                SafeTensorsStreamWriter.Write(tmp, plan.Select(x => new SafeTensorsEntry(x.Name, x.Tensor.DType, x.Tensor.Shape, x.Tensor.ByteCount, s => archive.CopyTo(x.Tensor, s, cancellation))), cancellation);
                Verify(tmp, archive, plan, p.Describe);
                File.Move(tmp, output, overwrite: true);
            }
            finally { if (File.Exists(tmp)) File.Delete(tmp); }
        }
        else Verify(output, archive, plan, p.Describe + " (existing output; use --force to rewrite)");

        used.Add(source);
        return new ConversionResult(p.Output, upToDate ? ConversionStatus.UpToDate : ConversionStatus.Written, source);
    }

    private static ConversionResult CopyTokenizer(string modelDir, ConversionOptions options, List<string> used)
    {
        var output = Path.Combine(modelDir, ModelDirectory_Tokenizer);
        bool exists = File.Exists(output);
        if (exists && !options.Force && !options.DeleteSources) return new ConversionResult(ModelDirectory_Tokenizer, ConversionStatus.UpToDate, null);
        var source = Locate(modelDir, TokenizerSource, options.From);
        if (exists && !options.Force)
        {
            if (Sha(output) != Sha(source)) throw new ConversionException($"{output} differs from {source}; use --force to rewrite it");
            used.Add(source);
            return new ConversionResult(ModelDirectory_Tokenizer, ConversionStatus.UpToDate, source);
        }
        var tmp = output + ".tmp";
        try
        {
            File.Copy(source, tmp, overwrite: true);
            if (Sha(tmp) != Sha(source)) throw new ConversionException($"the copy of {source} does not match it");
            File.Move(tmp, output, overwrite: true);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
        used.Add(source);
        return new ConversionResult(ModelDirectory_Tokenizer, ConversionStatus.Written, source);
    }

    private static bool IsValid(string path, Profile p)
    {
        if (!File.Exists(path)) return false;
        try
        {
            var h = SafeTensorsHeader.Read(path);
            return h.Entries.Count == p.Expected && h.Entries.All(e => e.DType == p.DTypeOf(e.Name));
        }
        catch (InvalidDataException) { return false; }
    }

    private static string Sha(string path)
    {
        using var f = File.OpenRead(path);
        return System.Convert.ToHexString(SHA256.HashData(f));
    }

    /// <summary>Re-reads an output and compares every tensor (name, type, shape, bytes) with the source.</summary>
    private static void Verify(string path, TorchArchive archive, List<(TorchTensorInfo Tensor, string Name)> plan, string what)
    {
        var header = SafeTensorsHeader.Read(path);
        if (header.Entries.Count != plan.Count) throw new ConversionException($"{what}: the output has {header.Entries.Count} tensors, {plan.Count} expected");
        var byName = header.Entries.ToDictionary(e => e.Name, StringComparer.Ordinal);
        using var file = File.OpenRead(path);
        var buffer = new byte[1 << 20];
        foreach (var (t, name) in plan)
        {
            if (!byName.TryGetValue(name, out var e)) throw new ConversionException($"{what}: the output lacks '{name}'");
            if (e.DType != t.DType.SafeTensorsName() || !e.Shape.SequenceEqual(t.Shape) || e.End - e.Start != t.ByteCount)
                throw new ConversionException($"{what}: '{name}' differs from the source in type or shape");
            using var expected = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            using (var sink = new HashSink(expected)) archive.CopyTo(t, sink);
            using var actual = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            file.Position = header.DataStart + e.Start;
            for (long left = t.ByteCount; left > 0;)
            {
                int n = file.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
                if (n <= 0) throw new ConversionException($"{what}: the output ends inside '{name}'");
                actual.AppendData(buffer, 0, n);
                left -= n;
            }
            if (!expected.GetHashAndReset().AsSpan().SequenceEqual(actual.GetHashAndReset())) throw new ConversionException($"{what}: the bytes of '{name}' differ from the source");
        }
    }

    private sealed class HashSink : Stream
    {
        private readonly IncrementalHash hash;
        public HashSink(IncrementalHash hash) => this.hash = hash;
        public override void Write(byte[] buffer, int offset, int count) => hash.AppendData(buffer, offset, count);
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
    }
}
