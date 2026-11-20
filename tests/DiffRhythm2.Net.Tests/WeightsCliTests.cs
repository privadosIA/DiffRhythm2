using System.Net;
using System.Security.Cryptography;
using DiffRhythm2.Weights;

namespace DiffRhythm2.Tests;

/// <summary>The fetch, convert and install commands, run in-process against a local server and, where noted, the real checkpoints.</summary>
public class WeightsCliTests : IDisposable
{
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-wcli-").FullName;
    private readonly Dictionary<string, byte[]> served = new();
    private readonly FileServer server;
    private readonly List<ManifestFile> manifest = new();

    public WeightsCliTests()
    {
        server = new FileServer("/unused", Array.Empty<byte>());
        server.Hook = (ctx, n) =>
        {
            if (!served.TryGetValue(ctx.Request.Url!.AbsolutePath, out var bytes)) { ctx.Response.StatusCode = 404; ctx.Response.Close(); return true; }
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes);
            ctx.Response.Close();
            return true;
        };
        Add(ModelManifest.DiffRhythm2Repository, "config.json", "config.json", 2000);
        Add(ModelManifest.MuLanRepository, "pytorch_model.bin", "sources/pytorch_model.bin", 150_000);
        Add(ModelManifest.XlmRobertaRepository, "tokenizer.json", "sources/tokenizer.json", 30_000);
        Cli.Cli.ManifestOverride = manifest;
        Cli.Cli.DownloaderOverride = new ModelDownloader(null, new DownloaderOptions { Endpoint = server.Endpoint, MaxAttempts = 2, BaseDelay = TimeSpan.FromMilliseconds(5) }, manifest);
    }

    private void Add(string repo, string file, string target, int size)
    {
        var bytes = new byte[size];
        new Random(size).NextBytes(bytes);
        served[$"/{repo}/resolve/0123456789012345678901234567890123456789/{file}"] = bytes;
        manifest.Add(new ManifestFile(repo, "0123456789012345678901234567890123456789", file, target, size, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant()));
    }

    public void Dispose()
    {
        Cli.Cli.ManifestOverride = null;
        Cli.Cli.DownloaderOverride = null;
        server.Dispose();
        Directory.Delete(tmp, recursive: true);
    }

    private static (int Code, string Out, string Err) Run(string[] args, CancellationToken ct = default)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Cli.Cli.Run(args, o, e, new StringReader(""), ct);
        return (code, o.ToString(), e.ToString());
    }

    [Fact]
    public void Help_ListsTheNewCommandsAndOptions()
    {
        var (code, output, _) = Run(new[] { "--help" });
        Assert.Equal(0, code);
        foreach (var word in new[] { "install", "fetch", "convert", "--accept-licenses", "--from", "--force", "--delete-sources", "HF_ENDPOINT", "CC-BY-NC-4.0" })
            Assert.Contains(word, output);
    }

    [Fact]
    public void Fetch_WithoutAcceptance_ShowsTheLicenses_ExitsWith2_AndDownloadsNothing()
    {
        var (code, output, err) = Run(new[] { "fetch", "--model", tmp });
        Assert.Equal(2, code);
        Assert.Equal(0, server.Requests);
        Assert.Contains("Apache-2.0", output);
        Assert.Contains("MIT", output);
        Assert.Contains("CC-BY-NC-4.0", output);
        Assert.Contains("--accept-licenses", err);
        Assert.Empty(Directory.GetFileSystemEntries(tmp));
    }

    [Fact]
    public void Fetch_FirstRun_PrintsAStepPerFile_AndEndsWithDone_SecondRunReportsThemPresent()
    {
        var (code, output, _) = Run(new[] { "fetch", "--model", tmp, "--accept-licenses" });
        Assert.Equal(0, code);
        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.TrimEnd('\r')).ToList();
        Assert.Contains("step: ASLP-lab/DiffRhythm2/config.json", lines);
        Assert.Contains("step: OpenMuQ/MuQ-MuLan-large/pytorch_model.bin", lines);
        Assert.Equal("done", lines[^1]);
        Assert.All(manifest, f => Assert.True(File.Exists(Path.Combine(tmp, f.Target))));

        int before = server.Requests;
        var (code2, output2, _) = Run(new[] { "fetch", "--model", tmp, "--accept-licenses" });
        Assert.Equal(0, code2);
        Assert.Equal(before, server.Requests);
        Assert.Equal(manifest.Count, output2.Split('\n').Count(l => l.StartsWith("present: ")));
    }

    [Fact]
    public void Fetch_WhenAFileCannotBeObtained_ListsWhatIsMissing_AndExitsWith1()
    {
        served.Remove(served.Keys.First(k => k.EndsWith("tokenizer.json")));
        var (code, output, err) = Run(new[] { "fetch", "--model", tmp, "--accept-licenses" });
        Assert.Equal(1, code);
        Assert.Contains("missing: sources/tokenizer.json", output);
        Assert.Contains("404", err);
    }

    [Fact]
    public void Fetch_Cancelled_ExitsWith130()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var (code, _, err) = Run(new[] { "fetch", "--model", tmp, "--accept-licenses" }, cts.Token);
        Assert.Equal(130, code);
        Assert.Contains("cancelled", err);
    }

    [Fact]
    public void Install_WithoutAcceptance_ExitsWith2()
    {
        Assert.Equal(2, Run(new[] { "install", "--model", tmp }).Code);
        Assert.Equal(0, server.Requests);
    }

    [Fact]
    public void Convert_WithoutSources_FailsListingTheSearchedPlaces()
    {
        var (code, _, err) = Run(new[] { "convert", "--model", tmp });
        Assert.Equal(1, code);
        Assert.Contains("decoder.bin was not found", err);
        Assert.Contains("sources", err);
    }

    [Fact]
    public void BadOptions_AreReportedWithExitCode2()
    {
        Assert.Equal(2, Run(new[] { "fetch" }).Code);
        Assert.Contains("--model", Run(new[] { "convert" }).Err);
        Assert.Contains("--from", Run(new[] { "convert", "--model", tmp, "--from", Path.Combine(tmp, "nope") }).Err);
        Assert.Contains("unknown option", Run(new[] { "install", "--model", tmp, "--verbose" }).Err);
    }

    [Fact]
    public void Check_OnAnEmptyDirectory_SuggestsInstall()
    {
        var (code, output, _) = Run(new[] { "check", "--model", tmp });
        Assert.Equal(1, code);
        Assert.Contains("diffrhythm2 install --model", output);
    }

    // ---- with the real checkpoints --------------------------------------------------------------------------------------------------

    private static readonly string[] ModelFiles = { "config.json", "model.safetensors", "decoder.json", "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json" };

    private string? LinkModelDirectory(string[] files)
    {
        var ckpt = Checkpoint.Dir;
        if (ckpt is null || files.Any(f => !File.Exists(Path.Combine(ckpt, f)))) return null;
        foreach (var f in files) File.CreateSymbolicLink(Path.Combine(tmp, f), Path.Combine(ckpt, f));
        return tmp;
    }

    [RequiresCheckpoint]
    public void Install_OnACompleteDirectory_DownloadsAndRewritesNothing_AndEndsReady()
    {
        if (LinkModelDirectory(ModelFiles) is null) return;
        manifest.Clear();   // nothing is missing, so nothing may be requested
        var (code, output, _) = Run(new[] { "install", "--model", tmp, "--accept-licenses" });
        Assert.Equal(0, code);
        Assert.Equal(0, server.Requests);
        Assert.EndsWith("ready", output.TrimEnd());
        Assert.Contains("decoder.safetensors: up to date", output);
    }

    [RequiresCheckpoint]
    public void Convert_WithDeleteSources_ConvertsFirst_ThenRemovesOnlyTheCheckpoints()
    {
        var ckpt = Checkpoint.Dir!;
        string?[] sources = { WeightConverter.FindSource(ckpt, "decoder.bin"), WeightConverter.FindSource(ckpt, "pytorch_model.bin"), WeightConverter.FindSource(ckpt, "tokenizer.json") };
        if (sources.Any(s => s is null)) return;
        Directory.CreateDirectory(Path.Combine(tmp, "sources"));
        var names = new[] { "decoder.bin", "pytorch_model.bin", "tokenizer.json" };
        for (int i = 0; i < 3; i++) File.CreateSymbolicLink(Path.Combine(tmp, "sources", names[i]), sources[i]!);   // links: deleting them leaves the real files alone

        var (code, output, _) = Run(new[] { "convert", "--model", tmp, "--delete-sources" });
        Assert.Equal(0, code);
        Assert.EndsWith("done", output.TrimEnd());
        Assert.Equal(new[] { "step: decoder.safetensors", "step: mulan_text.safetensors", "step: mulan_text_tokenizer.json" }, output.Split('\n').Where(l => l.StartsWith("step:")).Select(l => l.TrimEnd('\r')));
        foreach (var n in new[] { "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json" }) Assert.True(File.Exists(Path.Combine(tmp, n)));
        foreach (var n in names) Assert.False(File.Exists(Path.Combine(tmp, "sources", n)));
        foreach (var s in sources) Assert.True(File.Exists(s));
    }
}
