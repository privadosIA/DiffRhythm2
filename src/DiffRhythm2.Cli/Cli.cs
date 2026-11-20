using System.Globalization;
using System.Net.Http;
using System.Text;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Text;
using DiffRhythm2.Weights;
using TorchSharp;

namespace DiffRhythm2.Cli;

/// <summary>The command line of the song pipeline. Separated from <c>Main</c> so it can be run in-process with captured output.</summary>
public static partial class Cli
{
    public const int Ok = 0, Failed = 1, Usage = 2, Cancelled = 130;

    public const string UsageText = """
        diffrhythm2 - make a song from lyrics and a style text (DiffRhythm2, C# / TorchSharp)

        Usage:
          diffrhythm2 generate --model DIR --lyrics FILE|- --style TEXT --out FILE.wav [options]
          diffrhythm2 check    --model DIR [--device DEVICE]
          diffrhythm2 install  --model DIR --accept-licenses [--audio]
          diffrhythm2 fetch    --model DIR --accept-licenses
          diffrhythm2 convert  --model DIR [--from DIR] [--force] [--delete-sources] [--audio]
          diffrhythm2 embed    --model DIR (--audio FILE.wav | --text TEXT) [--out FILE.json] [--device D]
          diffrhythm2 classify --model DIR --audio FILE.wav --vocab FILE.json [--top N] [--json] [--device D]
          diffrhythm2 --help

        generate options:
          --model DIR      directory with config.json, model.safetensors, decoder.json, decoder.safetensors,
                           mulan_text.safetensors and mulan_text_tokenizer.json
          --lyrics FILE    lyrics, English, with optional [verse]/[chorus]/... lines; "-" reads standard input
          --style TEXT     comma-separated style words, e.g. "Pop, Piano, Female vocals" (500 characters at most)
          --out FILE       output WAV (16 bit, 48 kHz); written only if the song succeeds
          --seconds N      length ceiling in seconds, 10 to 210 (default 210); the song ends by itself at the end of the lyrics
          --seed N         seed, 0 or more (default: a random one, always printed as seed=N); the same seed repeats the song
          --cfg X          guidance strength, 0.5 to 10 (default 2.0)
          --steps N        sampling steps per block, 4 to 64 (default 16)
          --device D       cpu (default), cuda or cuda:N
          --dtype T        fp32 or fp16 (default: fp32 on cpu, fp16 on cuda)
          --mono           write one channel instead of the stereo of the service
          --espeak-lib F   path of libespeak-ng (default: DIFFRHYTHM2_ESPEAK_LIB or the system library)
          --espeak-data D  espeak-ng-data directory (default: DIFFRHYTHM2_ESPEAK_DATA or the library's own)

        check options:
          --model DIR      model directory to inspect
          --device D       device the song would run on (default cpu)

        install, fetch and convert (see docs/weights-tools.md) build the model directory without Python:
          install          fetch what is missing, convert it and check the result (the usual way to start)
          fetch            download the six files (about 7.7 GB) from pinned HuggingFace revisions, verified and resumable;
                           the checkpoints to convert go to DIR/sources
          convert          turn the downloaded checkpoints into decoder.safetensors, mulan_text.safetensors and
                           mulan_text_tokenizer.json; outputs that are already valid are left alone
          --accept-licenses  required to download: DiffRhythm2 is Apache-2.0, XLM-RoBERTa is MIT, MuQ-MuLan is
                           CC-BY-NC-4.0 (non-commercial use only)
          --from DIR       look for the checkpoints in DIR first (convert)
          --force          rewrite outputs even if valid ones exist (convert)
          --delete-sources delete the checkpoints once the conversion is written and verified (convert)
          --audio          also produce mulan_audio.safetensors (1.3 GB), the audio tower of MuQ-MuLan that only the song
                           analysis needs; making songs does not (convert, install). Nothing more is downloaded
          Environment: HF_ENDPOINT (mirror of huggingface.co), HTTPS_PROXY / HTTP_PROXY.

        embed and classify (see docs/mulan-audio.md) recognize and compare songs with MuQ-MuLan; they need
        mulan_audio.safetensors (diffrhythm2 convert --audio) and, for text, mulan_text.safetensors and
        mulan_text_tokenizer.json. The audio must be a 24 kHz WAV (ffmpeg -i IN -ac 1 -ar 24000 OUT.wav). Non-commercial use only.
          embed            print the 512-value embedding of an audio file or of a text, as one JSON object on standard output
                           (kind, dim, embedding and, for audio, clips, seconds and length); --out writes it to a file instead
          classify         rank the tags of a vocabulary for an audio file. --vocab is a JSON object of categories, each a list of
                           tags or {"template": "This is a {} song.", "tags": [...]}; one line per category:
                           "category: tag score, tag score, ..." (the best --top, default 3), or one JSON object with --json
          --audio FILE     the audio (embed: exclusive with --text; classify: required)
          --text TEXT      the text to embed (embed)
          --vocab FILE     the tag vocabulary (classify)
          --top N          how many tags to keep per category (classify, default 3)
          --json           print JSON instead of lines (classify)
          --out FILE       write the JSON here instead of standard output (embed)

        Exit codes: 0 done, 1 failure while working, 2 wrong usage, 130 cancelled.
        """;

    public static int Run(string[] args, TextWriter stdout, TextWriter stderr, TextReader stdin, CancellationToken cancellation = default)
    {
        if (args.Length == 0)
        {
            stderr.WriteLine(UsageText);
            return Usage;
        }
        var command = args[0];
        if (command is "--help" or "-h" or "help")
        {
            stdout.WriteLine(UsageText);
            return Ok;
        }
        return command switch
        {
            "generate" => Generate(args[1..], stdout, stderr, stdin, cancellation),
            "check" => Check(args[1..], stdout, stderr),
            "fetch" => Fetch(args[1..], stdout, stderr, install: false, cancellation),
            "install" => Fetch(args[1..], stdout, stderr, install: true, cancellation),
            "convert" => ConvertWeights(args[1..], stdout, stderr, cancellation),
            "embed" => Embed(args[1..], stdout, stderr, cancellation),
            "classify" => Classify(args[1..], stdout, stderr, cancellation),
            _ => Fail(stderr, Usage, $"unknown command '{command}'; the commands are: generate, check, install, fetch, convert, embed, classify (try --help)"),
        };
    }

    private static int Fail(TextWriter stderr, int code, string message)
    {
        stderr.WriteLine($"error: {message}");
        return code;
    }

    // ---- argument parsing ------------------------------------------------------------------------------------------------------------

    internal sealed class ArgumentError : Exception
    {
        public ArgumentError(string message) : base(message) { }
    }

    private static readonly string[] Flags = { "--mono", "--accept-licenses", "--force", "--delete-sources", "--audio" };
    private static readonly string[] InstallOptions = { "--model", "--accept-licenses", "--audio" };
    private static readonly string[] InstallOptionsWithoutAudio = { "--model", "--accept-licenses" };
    private static readonly string[] ConvertOptions = { "--model", "--from", "--force", "--delete-sources", "--audio" };

    /// <summary>Parses <c>--option value</c> pairs and flags. <paramref name="flags"/> says which options take no value (the default list is that of generate, check and the weight commands).</summary>
    private static Dictionary<string, string> Parse(string[] args, string[] known, string[]? flags = null)
    {
        flags ??= Flags;
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (!a.StartsWith("--", StringComparison.Ordinal)) throw new ArgumentError($"unexpected argument '{a}'");
            if (!known.Contains(a)) throw new ArgumentError($"unknown option '{a}'");
            if (flags.Contains(a)) { values[a] = "true"; continue; }
            if (i + 1 >= args.Length) throw new ArgumentError($"option {a} needs a value");
            if (values.ContainsKey(a)) throw new ArgumentError($"option {a} was given twice");
            values[a] = args[++i];
        }
        return values;
    }

    private static string Required(Dictionary<string, string> v, string option) =>
        v.TryGetValue(option, out var value) ? value : throw new ArgumentError($"option {option} is required");

    private static double Number(Dictionary<string, string> v, string option, double fallback) =>
        v.TryGetValue(option, out var text)
            ? double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var n) && double.IsFinite(n) ? n : throw new ArgumentError($"{option} must be a number, got '{text}'")
            : fallback;

    private static readonly Dictionary<string, string> OptionOfField = new()
    {
        ["lyrics"] = "--lyrics",
        ["style"] = "--style",
        ["max_secs"] = "--seconds",
        ["cfg_strength"] = "--cfg",
        ["steps"] = "--steps",
        ["seed"] = "--seed",
    };

    private static bool IsDeviceName(string d) =>
        d.Equals("cpu", StringComparison.OrdinalIgnoreCase) || d.Equals("cuda", StringComparison.OrdinalIgnoreCase)
        || (d.StartsWith("cuda:", StringComparison.OrdinalIgnoreCase) && int.TryParse(d.AsSpan(5), NumberStyles.None, CultureInfo.InvariantCulture, out _));

    // ---- generate --------------------------------------------------------------------------------------------------------------------

    private static int Generate(string[] args, TextWriter stdout, TextWriter stderr, TextReader stdin, CancellationToken cancellation)
    {
        SongRequest request;
        SongOptions options;
        string outPath;
        bool mono;
        try
        {
            var v = Parse(args, new[] { "--model", "--lyrics", "--style", "--out", "--seconds", "--seed", "--cfg", "--steps", "--device", "--dtype", "--mono", "--espeak-lib", "--espeak-data" });
            var model = Required(v, "--model");
            var lyricsSource = Required(v, "--lyrics");
            var style = Required(v, "--style");
            outPath = Required(v, "--out");
            mono = v.ContainsKey("--mono");

            string lyrics;
            if (lyricsSource == "-") lyrics = stdin.ReadToEnd();
            else
            {
                if (!File.Exists(lyricsSource)) throw new ArgumentError($"--lyrics: the file '{lyricsSource}' does not exist");
                lyrics = File.ReadAllText(lyricsSource, Encoding.UTF8);
            }

            long? seed = null;
            if (v.TryGetValue("--seed", out var seedText))
            {
                if (!SongRequest.TryParseSeed(seedText, out var s)) throw new ArgumentError($"--seed: {SongRequest.SeedMessage} (got '{seedText}')");
                seed = s;
            }
            double steps = Number(v, "--steps", SongRequest.DefaultSteps);
            if (!double.IsInteger(steps) || steps < int.MinValue || steps > int.MaxValue) throw new ArgumentError($"--steps must be a whole number, got '{v["--steps"]}'");

            var device = v.TryGetValue("--device", out var d) ? d : "cpu";
            if (!IsDeviceName(device)) throw new ArgumentError($"--device must be cpu, cuda or cuda:N, got '{device}'");
            SongPrecision? precision = null;
            if (v.TryGetValue("--dtype", out var dtype))
                precision = dtype.ToLowerInvariant() switch { "fp32" => SongPrecision.Float32, "fp16" => SongPrecision.Float16, _ => throw new ArgumentError($"--dtype must be fp32 or fp16, got '{dtype}'") };

            if (v.TryGetValue("--espeak-lib", out var lib)) Environment.SetEnvironmentVariable(EspeakEngine.LibraryVariable, lib);
            if (v.TryGetValue("--espeak-data", out var data)) Environment.SetEnvironmentVariable(EspeakEngine.DataVariable, data);

            request = new SongRequest(lyrics, style, Number(v, "--seconds", SongRequest.DefaultSeconds), Number(v, "--cfg", SongRequest.DefaultCfgStrength), (int)steps, seed);
            try { request.Validate(); }
            catch (SongRequestException e) { throw new ArgumentError($"{OptionOfField[e.Field]}: {e.Message}"); }

            options = new SongOptions(model, device, precision);
        }
        catch (ArgumentError e)
        {
            return Fail(stderr, Usage, e.Message);
        }

        request = request with { Seed = request.Seed ?? NoiseSource.NewSeed() };
        stdout.WriteLine($"seed={request.Seed}");
        stdout.Flush();

        try
        {
            var progress = new SyncProgress(p => stderr.WriteLine(Describe(p)));
            var result = new SongGenerator(options).Generate(request, progress, cancellation);
            result.WriteWav(outPath, stereo: !mono);
            stdout.WriteLine($"wrote {outPath} ({result.Audio.Length / (double)result.SampleRate:F1} s, {result.SampleRate} Hz, {(mono ? "mono" : "stereo")}, " +
                             $"{result.Frames} frames{(result.EndedEarly ? ", ended at the end of the lyrics" : "")}, {result.Elapsed.TotalSeconds:F0} s to make)");
            return Ok;
        }
        catch (OperationCanceledException)
        {
            return Fail(stderr, Cancelled, "cancelled; no file was written");
        }
        catch (Exception e) when (e is SongException or G2pException or IOException or NotSupportedException or InvalidDataException or ArgumentException)
        {
            return Fail(stderr, Failed, e.Message);
        }
    }

    private sealed class SyncProgress : IProgress<SongProgress>
    {
        private readonly Action<SongProgress> handler;
        public SyncProgress(Action<SongProgress> handler) => this.handler = handler;
        public void Report(SongProgress value) => handler(value);
    }

    private static string Describe(SongProgress p) => p.Event switch
    {
        SongEvent.StageStarted => $"[{p.Stage}] started",
        SongEvent.ModelLoaded => $"[{p.Stage}] model loaded",
        SongEvent.BlockDone => $"[{p.Stage}] block {p.Done}/{p.Total}",
        SongEvent.ModelReleased => $"[{p.Stage}] model released",
        _ => $"[{p.Stage}] done",
    };

    // ---- fetch / install / convert -----------------------------------------------------------------------------------------------------

    /// <summary>Replaces the downloader (for tests that serve the files from a local server). Null uses the real hub.</summary>
    public static ModelDownloader? DownloaderOverride { get; set; }

    /// <summary>Replaces the list of files to download (for tests). Null uses <see cref="ModelManifest.Files"/>.</summary>
    public static IReadOnlyList<ManifestFile>? ManifestOverride { get; set; }

    private static int Fetch(string[] args, TextWriter stdout, TextWriter stderr, bool install, CancellationToken cancellation)
    {
        Dictionary<string, string> v;
        try { v = Parse(args, install ? InstallOptions : InstallOptionsWithoutAudio); Required(v, "--model"); }
        catch (ArgumentError e) { return Fail(stderr, Usage, e.Message); }
        var modelDir = v["--model"];
        bool accepted = v.ContainsKey("--accept-licenses");
        bool audio = v.ContainsKey("--audio");

        var manifest = ManifestOverride ?? ModelManifest.Files;
        var wanted = install ? (ManifestOverride is null ? ModelManifest.Needed(modelDir, audio) : manifest.Where(f => !File.Exists(Path.Combine(modelDir, f.Target))).ToList()) : manifest;
        var licenses = ModelDownloader.LicensesOf(wanted);
        foreach (var l in licenses) stdout.WriteLine($"license: {l.Repository} is {l.License}{(l.NonCommercial ? " (non-commercial use only)" : "")}");
        if (!accepted && licenses.Any(l => l.NonCommercial))
            return Fail(stderr, Usage, "the licenses above must be accepted before anything is downloaded; add --accept-licenses to accept them (nothing was downloaded)");

        if (wanted.Count > 0)
        {
            var downloader = DownloaderOverride ?? new ModelDownloader(files: manifest);
            var progress = new SyncDownloadProgress(e => WriteDownload(stdout, e));
            try
            {
                downloader.EnsureAsync(modelDir, accepted, wanted, progress, cancellation).GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                return Fail(stderr, Cancelled, "cancelled; partial downloads were kept and the next run resumes them");
            }
            catch (LicenseNotAcceptedException e)
            {
                return Fail(stderr, Usage, e.Message + "; add --accept-licenses");
            }
            catch (Exception e) when (e is IOException or HttpRequestException or UnauthorizedAccessException)
            {
                stdout.WriteLine("missing: " + string.Join(", ", wanted.Where(f => !File.Exists(Path.Combine(modelDir, f.Target))).Select(f => f.Target)));
                return Fail(stderr, Failed, e.Message);
            }
        }

        if (!install)
        {
            stdout.WriteLine("done");
            return Ok;
        }

        var code = RunConversion(modelDir, new ConversionOptions(Audio: audio), stdout, stderr, cancellation);
        if (code != Ok) return code;
        var missing = new ModelDirectory(modelDir).Missing.ToList();
        if (audio && !File.Exists(Path.Combine(modelDir, WeightConverter.AudioFile))) missing.Add(WeightConverter.AudioFile);
        if (missing.Count > 0)
        {
            stdout.WriteLine("missing: " + string.Join(", ", missing));
            return Failed;
        }
        stdout.WriteLine("ready");
        return Ok;
    }

    private static void WriteDownload(TextWriter stdout, DownloadEvent e)
    {
        switch (e.Kind)
        {
            case DownloadEventKind.Step: stdout.WriteLine($"step: {e.File.Repository}/{e.File.File}"); break;
            case DownloadEventKind.Present: stdout.WriteLine($"present: {e.File.Target}"); break;
            case DownloadEventKind.Bytes: stdout.WriteLine($"  {e.Done:N0} of {e.File.Size:N0} bytes"); break;
            case DownloadEventKind.Retry: stdout.WriteLine($"  retry: {e.Message}"); break;
        }
        stdout.Flush();
    }

    private sealed class SyncDownloadProgress : IProgress<DownloadEvent>
    {
        private readonly Action<DownloadEvent> handler;
        public SyncDownloadProgress(Action<DownloadEvent> handler) => this.handler = handler;
        public void Report(DownloadEvent value) => handler(value);
    }

    private static int ConvertWeights(string[] args, TextWriter stdout, TextWriter stderr, CancellationToken cancellation)
    {
        Dictionary<string, string> v;
        try { v = Parse(args, ConvertOptions); Required(v, "--model"); }
        catch (ArgumentError e) { return Fail(stderr, Usage, e.Message); }
        var options = new ConversionOptions(v.GetValueOrDefault("--from"), v.ContainsKey("--force"), v.ContainsKey("--delete-sources"), v.ContainsKey("--audio"));
        if (options.From is not null && !Directory.Exists(options.From)) return Fail(stderr, Usage, $"--from: the directory '{options.From}' does not exist");
        var code = RunConversion(v["--model"], options, stdout, stderr, cancellation);
        if (code == Ok) stdout.WriteLine("done");
        return code;
    }

    private static int RunConversion(string modelDir, ConversionOptions options, TextWriter stdout, TextWriter stderr, CancellationToken cancellation)
    {
        try
        {
            var results = WeightConverter.Convert(modelDir, options, name => { stdout.WriteLine($"step: {name}"); stdout.Flush(); }, cancellation);
            foreach (var r in results) stdout.WriteLine($"  {r.Output}: {(r.Status == ConversionStatus.Written ? "written" : "up to date")}");
            if (options.DeleteSources) stdout.WriteLine("  the converted checkpoints were deleted");
            return Ok;
        }
        catch (OperationCanceledException)
        {
            return Fail(stderr, Cancelled, "cancelled; no incomplete file was left under its final name");
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException)
        {
            return Fail(stderr, Failed, e.Message);
        }
    }

    // ---- check -----------------------------------------------------------------------------------------------------------------------

    private static int Check(string[] args, TextWriter stdout, TextWriter stderr)
    {
        Dictionary<string, string> v;
        try { v = Parse(args, new[] { "--model", "--device", "--espeak-lib", "--espeak-data" }); Required(v, "--model"); }
        catch (ArgumentError e) { return Fail(stderr, Usage, e.Message); }
        var device = v.TryGetValue("--device", out var d) ? d : "cpu";
        if (!IsDeviceName(device)) return Fail(stderr, Usage, $"--device must be cpu, cuda or cuda:N, got '{device}'");
        if (v.TryGetValue("--espeak-lib", out var lib)) Environment.SetEnvironmentVariable(EspeakEngine.LibraryVariable, lib);
        if (v.TryGetValue("--espeak-data", out var data)) Environment.SetEnvironmentVariable(EspeakEngine.DataVariable, data);

        var problems = new List<string>();
        var dir = new ModelDirectory(v["--model"]);
        stdout.WriteLine($"model directory: {dir.Path}");
        if (!Directory.Exists(dir.Path)) problems.Add($"the model directory {dir.Path} does not exist");
        foreach (var f in dir.Files)
        {
            stdout.WriteLine(f.Present ? $"  [ok]      {f.Name} ({f.Length:N0} bytes)" : $"  [missing] {f.Name}");
            if (!f.Present && Directory.Exists(dir.Path)) problems.Add($"{f.Name} is missing from the model directory");
        }

        if (EspeakEngine.TryGetShared(out var engine, out var reason))
        {
            var note = engine!.VersionText.StartsWith("1.51", StringComparison.Ordinal) ? "the version of the reference" : "the reference uses 1.51; pronunciations may differ";
            stdout.WriteLine($"phonemizer: espeak-ng {engine.VersionText} at {engine.LibraryPath} ({note})");
        }
        else
        {
            stdout.WriteLine($"phonemizer: NOT FOUND - {reason}");
            problems.Add("the phonemizer library (libespeak-ng) cannot be loaded");
        }

        stdout.WriteLine("devices:");
        stdout.WriteLine("  cpu: available");
        bool cuda = torch.cuda.is_available();
        stdout.WriteLine(cuda ? $"  cuda: {torch.cuda.device_count()} device(s) available" : "  cuda: not available (this build of libtorch has no CUDA support or no device was found)");
        try { SongGenerator.ResolveDevice(device); }
        catch (DeviceUnavailableException e) { problems.Add(e.Message); }

        if (problems.Count == 0)
        {
            stdout.WriteLine($"ready: a song can be made on {device}");
            return Ok;
        }
        stdout.WriteLine("not ready:");
        foreach (var p in problems) stdout.WriteLine($"  - {p}");
        if (dir.Missing.Count > 0) stdout.WriteLine($"to download and convert the missing files run: diffrhythm2 install --model {dir.Path} --accept-licenses");
        return Failed;
    }
}
