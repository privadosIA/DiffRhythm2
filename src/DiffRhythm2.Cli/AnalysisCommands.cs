using System.Globalization;
using System.Text;
using System.Text.Json;
using DiffRhythm2.AudioTower;
using DiffRhythm2.Pipeline;
using DiffRhythm2.StyleText;
using TorchSharp;

namespace DiffRhythm2.Cli;

// embed and classify: recognizing and comparing songs with MuQ-MuLan (docs/mulan-audio.md)
public static partial class Cli
{
    private static readonly string[] AnalysisFlags = { "--json" };
    private static readonly string[] EmbedOptions = { "--model", "--audio", "--text", "--out", "--device" };
    private static readonly string[] ClassifyOptions = { "--model", "--audio", "--vocab", "--top", "--device", "--json" };

    /// <summary>One category of a tag vocabulary: the tags, and the text each one is embedded as.</summary>
    internal sealed record VocabularyCategory(string Name, IReadOnlyList<string> Tags, IReadOnlyList<string> Texts);

    /// <summary>
    /// Reads a vocabulary: a JSON object whose keys are categories and whose values are a list of tags or <c>{"template": "...{}...", "tags": [...]}</c>.
    /// Throws <see cref="ArgumentError"/> naming the category for anything else.
    /// </summary>
    internal static IReadOnlyList<VocabularyCategory> ParseVocabulary(string json, string source)
    {
        JsonDocument doc;
        try { doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = false, CommentHandling = JsonCommentHandling.Disallow }); }
        catch (JsonException e) { throw new ArgumentError($"--vocab: {source} is not valid JSON ({e.Message})"); }
        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object) throw new ArgumentError($"--vocab: {source} must be a JSON object whose keys are categories");
            var categories = new List<VocabularyCategory>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                string name = prop.Name;
                if (name.Length == 0) throw new ArgumentError($"--vocab: {source} has a category with an empty name");
                if (!seen.Add(name)) throw new ArgumentError($"--vocab: the category '{name}' appears twice");
                string? template = null;
                JsonElement tagsElement;
                if (prop.Value.ValueKind == JsonValueKind.Array) tagsElement = prop.Value;
                else if (prop.Value.ValueKind == JsonValueKind.Object)
                {
                    if (!prop.Value.TryGetProperty("tags", out tagsElement) || tagsElement.ValueKind != JsonValueKind.Array)
                        throw new ArgumentError($"--vocab: the category '{name}' needs a \"tags\" list");
                    if (prop.Value.TryGetProperty("template", out var t))
                    {
                        if (t.ValueKind != JsonValueKind.String || !t.GetString()!.Contains("{}", StringComparison.Ordinal))
                            throw new ArgumentError($"--vocab: the template of the category '{name}' must be a text containing {{}}");
                        template = t.GetString();
                    }
                    foreach (var other in prop.Value.EnumerateObject())
                        if (other.Name is not ("tags" or "template")) throw new ArgumentError($"--vocab: the category '{name}' has an unknown key '{other.Name}' (only \"template\" and \"tags\" are allowed)");
                }
                else throw new ArgumentError($"--vocab: the category '{name}' must be a list of tags or an object with \"template\" and \"tags\"");

                var tags = new List<string>();
                var tagSet = new HashSet<string>(StringComparer.Ordinal);
                foreach (var tag in tagsElement.EnumerateArray())
                {
                    if (tag.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(tag.GetString()))
                        throw new ArgumentError($"--vocab: the category '{name}' has a tag that is not a non-empty text");
                    if (!tagSet.Add(tag.GetString()!)) throw new ArgumentError($"--vocab: the category '{name}' lists the tag '{tag.GetString()}' twice");
                    tags.Add(tag.GetString()!);
                }
                if (tags.Count == 0) throw new ArgumentError($"--vocab: the category '{name}' has no tags");
                var texts = tags.Select(tag => template is null ? tag : template.Replace("{}", tag, StringComparison.Ordinal)).ToList();
                if (texts.Distinct(StringComparer.Ordinal).Count() != texts.Count) throw new ArgumentError($"--vocab: two tags of the category '{name}' give the same text");
                categories.Add(new VocabularyCategory(name, tags, texts));
            }
            if (categories.Count == 0) throw new ArgumentError($"--vocab: {source} has no categories");
            return categories;
        }
    }

    /// <summary>The audio file, read and checked before any model is loaded. Throws <see cref="IOException"/> or <see cref="UnsupportedSampleRateException"/>.</summary>
    private static WavSignal ReadAudioForAnalysis(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"--audio: the file '{path}' does not exist", path);
        var wav = WavReader.Read(path);
        if (wav.Samples.Length < AudioClips.MinimumSignalSamples)
            throw new InvalidDataException($"{Path.GetFileName(path)}: the audio has {wav.Samples.Length} samples; at least {AudioClips.MinimumSignalSamples} are needed");
        return wav;
    }

    /// <summary>The converted files the command needs and the directory lacks, each with the command that creates it.</summary>
    private static List<string> MissingAnalysisFiles(string modelDir, bool audio, bool text)
    {
        var missing = new List<string>();
        if (audio && !File.Exists(Path.Combine(modelDir, MuLanAudioWeights.CheckpointFile)))
            missing.Add($"{MuLanAudioWeights.CheckpointFile} (create it with: diffrhythm2 convert --model {modelDir} --audio)");
        if (text)
            foreach (var f in new[] { MuLanTextWeights.CheckpointFile, MuLanTextWeights.TokenizerFile })
                if (!File.Exists(Path.Combine(modelDir, f))) missing.Add($"{f} (create it with: diffrhythm2 convert --model {modelDir})");
        return missing;
    }

    private static string MissingMessage(string modelDir, List<string> missing) =>
        Directory.Exists(modelDir) ? $"the model directory {modelDir} lacks: " + string.Join("; ", missing) : $"the model directory {modelDir} does not exist";

    private static void WriteJsonNumbers(Utf8JsonWriter w, string name, IEnumerable<float> values)
    {
        w.WriteStartArray(name);
        foreach (var v in values) w.WriteNumberValue(v);
        w.WriteEndArray();
    }

    private static torch.Device DeviceOf(string name) => SongGenerator.ResolveDevice(name);

    // ---- embed -----------------------------------------------------------------------------------------------------------------------

    private static int Embed(string[] args, TextWriter stdout, TextWriter stderr, CancellationToken cancellation)
    {
        Dictionary<string, string> v;
        string model, device;
        try
        {
            v = Parse(args, EmbedOptions, AnalysisFlags);
            model = Required(v, "--model");
            bool hasAudio = v.ContainsKey("--audio"), hasText = v.ContainsKey("--text");
            if (hasAudio == hasText) throw new ArgumentError("give exactly one of --audio and --text");
            if (hasText && string.IsNullOrWhiteSpace(v["--text"])) throw new ArgumentError("--text must not be empty");
            device = v.TryGetValue("--device", out var d) ? d : "cpu";
            if (!IsDeviceName(device)) throw new ArgumentError($"--device must be cpu, cuda or cuda:N, got '{device}'");
        }
        catch (ArgumentError e) { return Fail(stderr, Usage, e.Message); }

        bool audio = v.ContainsKey("--audio");
        try
        {
            WavSignal? wav = audio ? ReadAudioForAnalysis(v["--audio"]) : null;
            var missing = MissingAnalysisFiles(model, audio: audio, text: !audio);
            if (missing.Count > 0) return Fail(stderr, Failed, MissingMessage(model, missing));
            var dev = DeviceOf(device);

            string json;
            if (audio)
            {
                stderr.WriteLine($"audio: {wav!.Seconds:F1} s, {AudioClips.Split(wav.Samples).Count} clip(s)");
                using var encoder = MuLanAudioEncoder.LoadDirectory(model, dev);
                var result = encoder.Embed(wav, (done, total) => stderr.WriteLine($"clip {done}/{total}"), cancellation);
                json = EmbeddingJson("audio", result.Values, result.Clips, result.Seconds, result.Length);
            }
            else
            {
                using var style = StyleEncoder.LoadDirectory(model, dev);
                using var e = style.Embed(v["--text"]);
                var values = e.cpu().data<float>().ToArray();
                json = EmbeddingJson("text", values, null, null, null);
            }

            if (v.TryGetValue("--out", out var outPath)) WriteAtomically(outPath, json + "\n");
            else stdout.WriteLine(json);
            return Ok;
        }
        catch (OperationCanceledException) { return Fail(stderr, Cancelled, "cancelled; no output file was written"); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException or ArgumentException or UnsupportedSampleRateException or SongException)
        {
            return Fail(stderr, Failed, e.Message);
        }
    }

    private static string EmbeddingJson(string kind, float[] values, int? clips, double? seconds, double? length)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteString("kind", kind);
            w.WriteNumber("dim", values.Length);
            if (clips is { } c) w.WriteNumber("clips", c);
            if (seconds is { } s) w.WriteNumber("seconds", Math.Round(s, 3));
            if (length is { } l) w.WriteNumber("length", Math.Round(l, 6));
            WriteJsonNumbers(w, "embedding", values);
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    private static void WriteAtomically(string path, string content)
    {
        var tmp = path + ".tmp";
        try
        {
            File.WriteAllText(tmp, content, new UTF8Encoding(false));
            File.Move(tmp, path, overwrite: true);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    // ---- classify --------------------------------------------------------------------------------------------------------------------

    private static int Classify(string[] args, TextWriter stdout, TextWriter stderr, CancellationToken cancellation)
    {
        Dictionary<string, string> v;
        string model, device, audioPath;
        int top;
        IReadOnlyList<VocabularyCategory> vocabulary;
        try
        {
            v = Parse(args, ClassifyOptions, AnalysisFlags);
            model = Required(v, "--model");
            audioPath = Required(v, "--audio");
            var vocabPath = Required(v, "--vocab");
            top = 3;
            if (v.TryGetValue("--top", out var topText) && !(int.TryParse(topText, NumberStyles.None, CultureInfo.InvariantCulture, out top) && top >= 1))
                throw new ArgumentError($"--top must be a whole number of at least 1, got '{topText}'");
            device = v.TryGetValue("--device", out var d) ? d : "cpu";
            if (!IsDeviceName(device)) throw new ArgumentError($"--device must be cpu, cuda or cuda:N, got '{device}'");
            if (!File.Exists(vocabPath)) throw new ArgumentError($"--vocab: the file '{vocabPath}' does not exist");
            vocabulary = ParseVocabulary(File.ReadAllText(vocabPath), Path.GetFileName(vocabPath));
        }
        catch (ArgumentError e) { return Fail(stderr, Usage, e.Message); }
        catch (IOException e) { return Fail(stderr, Usage, $"--vocab: {e.Message}"); }

        try
        {
            var wav = ReadAudioForAnalysis(audioPath);
            var missing = MissingAnalysisFiles(model, audio: true, text: true);
            if (missing.Count > 0) return Fail(stderr, Failed, MissingMessage(model, missing));
            var dev = DeviceOf(device);

            stderr.WriteLine($"audio: {wav.Seconds:F1} s, {AudioClips.Split(wav.Samples).Count} clip(s)");
            AudioEmbedding embedding;
            using (var encoder = MuLanAudioEncoder.LoadDirectory(model, dev))
                embedding = encoder.Embed(wav, (done, total) => stderr.WriteLine($"clip {done}/{total}"), cancellation);

            using var style = StyleEncoder.LoadDirectory(model, dev);
            var rankings = new List<(VocabularyCategory Category, IReadOnlyList<(string Tag, string Text, float Score)> Best)>();
            foreach (var category in vocabulary)
            {
                cancellation.ThrowIfCancellationRequested();
                stderr.WriteLine($"category {category.Name}: {category.Tags.Count} tag(s)");
                var ranked = TagRanker.Rank(embedding.Values, category.Texts, style, top);
                var byText = category.Texts.Select((text, i) => (text, tag: category.Tags[i])).ToDictionary(x => x.text, x => x.tag, StringComparer.Ordinal);
                rankings.Add((category, ranked.Select(r => (byText[r.Text], r.Text, r.Score)).ToList()));
            }

            if (v.ContainsKey("--json")) stdout.WriteLine(ClassificationJson(embedding, rankings));
            else
                foreach (var (category, best) in rankings)
                    stdout.WriteLine($"{category.Name}: " + string.Join(", ", best.Select(b => $"{b.Tag} {b.Score.ToString("F4", CultureInfo.InvariantCulture)}")));
            return Ok;
        }
        catch (OperationCanceledException) { return Fail(stderr, Cancelled, "cancelled"); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException or ArgumentException or UnsupportedSampleRateException or SongException)
        {
            return Fail(stderr, Failed, e.Message);
        }
    }

    private static string ClassificationJson(AudioEmbedding embedding, List<(VocabularyCategory Category, IReadOnlyList<(string Tag, string Text, float Score)> Best)> rankings)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteNumber("clips", embedding.Clips);
            w.WriteNumber("seconds", Math.Round(embedding.Seconds, 3));
            w.WriteStartObject("categories");
            foreach (var (category, best) in rankings)
            {
                w.WriteStartArray(category.Name);
                foreach (var (tag, text, score) in best)
                {
                    w.WriteStartObject();
                    w.WriteString("tag", tag);
                    w.WriteString("text", text);
                    w.WriteNumber("score", score);
                    w.WriteEndObject();
                }
                w.WriteEndArray();
            }
            w.WriteEndObject();
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }
}
