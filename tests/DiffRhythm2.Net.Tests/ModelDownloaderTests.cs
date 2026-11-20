using System.Net;
using System.Security.Cryptography;
using DiffRhythm2.Weights;

namespace DiffRhythm2.Tests;

/// <summary>A small HTTP server on localhost that serves one path, with hooks to misbehave.</summary>
internal sealed class FileServer : IDisposable
{
    private readonly HttpListener listener = new();
    private readonly Task loop;
    public string Endpoint { get; }
    public List<string> Ranges { get; } = new();
    public int Requests;
    public Func<HttpListenerContext, int, bool>? Hook;
    public bool IgnoreRange;
    public byte[] Content;
    public string Path;

    public FileServer(string path, byte[] content)
    {
        Path = path;
        Content = content;
        for (int port = Random.Shared.Next(20000, 40000); ; port++)
        {
            try { listener.Prefixes.Clear(); listener.Prefixes.Add($"http://127.0.0.1:{port}/"); listener.Start(); Endpoint = $"http://127.0.0.1:{port}"; break; }
            catch (HttpListenerException) { }
        }
        loop = Task.Run(Loop);
    }

    private async Task Loop()
    {
        while (listener.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = await listener.GetContextAsync(); } catch (Exception) { return; }
            _ = Task.Run(() => Handle(ctx));
        }
    }

    private void Handle(HttpListenerContext ctx)
    {
        int n = Interlocked.Increment(ref Requests);
        try
        {
            var range = ctx.Request.Headers["Range"];
            lock (Ranges) Ranges.Add(range ?? "");
            if (Hook is not null && Hook(ctx, n)) return;
            if (ctx.Request.Url!.AbsolutePath != Path) { ctx.Response.StatusCode = 404; ctx.Response.Close(); return; }
            long start = 0;
            if (range is not null && !IgnoreRange && range.StartsWith("bytes=") && range.EndsWith("-")) start = long.Parse(range[6..^1]);
            ctx.Response.StatusCode = start > 0 ? 206 : 200;
            ctx.Response.ContentLength64 = Content.Length - start;
            ctx.Response.OutputStream.Write(Content, (int)start, Content.Length - (int)start);
            ctx.Response.Close();
        }
        catch (Exception) { try { ctx.Response.Abort(); } catch (Exception) { } }
    }

    public void Dispose() { listener.Stop(); listener.Close(); }
}

/// <summary>The downloader against a local server: verification, resume, retries, refusals, cancellation and the license gate.</summary>
public class ModelDownloaderTests : IDisposable
{
    private readonly string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "dr2-dl-" + Guid.NewGuid().ToString("N"));
    private readonly byte[] content = new byte[3_000_000];
    private readonly FileServer server;
    private readonly ManifestFile file;

    public ModelDownloaderTests()
    {
        Directory.CreateDirectory(dir);
        new Random(5).NextBytes(content);
        server = new FileServer("/owner/repo/resolve/rev1/data.bin", content);
        file = new ManifestFile("owner/repo", "rev1", "data.bin", "sources/data.bin", content.Length, Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant());
    }

    public void Dispose() { server.Dispose(); try { Directory.Delete(dir, true); } catch (IOException) { } }

    private ModelDownloader Downloader(int attempts = 3, TimeSpan? stall = null) =>
        new(null, new DownloaderOptions { Endpoint = server.Endpoint, MaxAttempts = attempts, BaseDelay = TimeSpan.FromMilliseconds(10), StallTimeout = stall ?? TimeSpan.FromSeconds(10) }, new[] { file });

    private string Target => System.IO.Path.Combine(dir, "sources", "data.bin");

    [Fact]
    public async Task Download_WritesTheVerifiedFile_AndNothingElse()
    {
        var r = await Downloader().EnsureAsync(dir, true);
        Assert.Equal(FetchStatus.Downloaded, Assert.Single(r).Status);
        Assert.Equal(content, File.ReadAllBytes(Target));
        Assert.False(File.Exists(Target + ".part"));
    }

    [Fact]
    public async Task ACompleteFile_IsNotDownloadedAgain()
    {
        await Downloader().EnsureAsync(dir, true);
        int before = server.Requests;
        var r = await Downloader().EnsureAsync(dir, true);
        Assert.Equal(FetchStatus.Present, Assert.Single(r).Status);
        Assert.Equal(before, server.Requests);
    }

    [Fact]
    public async Task CorruptedBytes_AreDeleted_AndBothHashesReported()
    {
        var bad = (byte[])content.Clone();
        bad[1234] ^= 0xFF;
        server.Content = bad;
        var e = await Assert.ThrowsAsync<DownloadException>(() => Downloader().EnsureAsync(dir, true));
        Assert.Contains(file.Sha256, e.Message);
        Assert.Contains(Convert.ToHexString(SHA256.HashData(bad)).ToLowerInvariant(), e.Message);
        Assert.False(File.Exists(Target));
        Assert.False(File.Exists(Target + ".part"));
    }

    [Fact]
    public async Task APartialFile_IsResumedWithARangeRequest()
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Target)!);
        File.WriteAllBytes(Target + ".part", content[..1_000_000]);
        await Downloader().EnsureAsync(dir, true);
        Assert.Equal("bytes=1000000-", Assert.Single(server.Ranges));
        Assert.Equal(content, File.ReadAllBytes(Target));
    }

    [Fact]
    public async Task AServerWithoutRanges_MakesTheDownloadStartOver()
    {
        server.IgnoreRange = true;
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Target)!);
        File.WriteAllBytes(Target + ".part", content[..1_000_000]);
        await Downloader().EnsureAsync(dir, true);
        Assert.Equal(content, File.ReadAllBytes(Target));
    }

    [Fact]
    public async Task ATransientServerError_IsRetried()
    {
        server.Hook = (ctx, n) => { if (n == 1) { ctx.Response.StatusCode = 503; ctx.Response.Close(); return true; } return false; };
        var events = new List<DownloadEvent>();
        await Downloader().EnsureAsync(dir, true, progress: new Progress(events.Add));
        Assert.Equal(2, server.Requests);
        Assert.Contains(events, x => x.Kind == DownloadEventKind.Retry);
        Assert.Equal(content, File.ReadAllBytes(Target));
    }

    [Fact]
    public async Task AConnectionCutHalfWay_IsResumedByTheRetry()
    {
        server.Hook = (ctx, n) =>
        {
            if (n != 1) return false;
            ctx.Response.ContentLength64 = content.Length;
            ctx.Response.OutputStream.Write(content, 0, 1_500_000);
            ctx.Response.OutputStream.Flush();
            ctx.Response.Abort();
            return true;
        };
        await Downloader().EnsureAsync(dir, true);
        Assert.Equal(2, server.Requests);
        Assert.StartsWith("bytes=", server.Ranges[1]);
        Assert.Equal(content, File.ReadAllBytes(Target));
    }

    [Fact]
    public async Task NotFound_FailsAtOnce_NamingTheAddressAndTheStatus()
    {
        server.Path = "/elsewhere";
        var e = await Assert.ThrowsAsync<DownloadException>(() => Downloader(attempts: 5).EnsureAsync(dir, true));
        Assert.Equal(1, server.Requests);
        Assert.Contains("404", e.Message);
        Assert.Contains("/owner/repo/resolve/rev1/data.bin", e.Message);
    }

    [Fact]
    public async Task GivingUp_AfterTheAttempts_SaysSo()
    {
        server.Hook = (ctx, n) => { ctx.Response.StatusCode = 500; ctx.Response.Close(); return true; };
        var e = await Assert.ThrowsAsync<DownloadException>(() => Downloader(attempts: 2).EnsureAsync(dir, true));
        Assert.Equal(2, server.Requests);
        Assert.Contains("giving up after 2 attempts", e.Message);
    }

    [Fact]
    public async Task Cancelling_KeepsThePartialFileForTheNextRun()
    {
        server.Hook = (ctx, n) =>
        {
            if (n != 1) return false;
            ctx.Response.ContentLength64 = content.Length;
            ctx.Response.OutputStream.Write(content, 0, 200_000);
            ctx.Response.OutputStream.Flush();
            Thread.Sleep(3000);
            return true;
        };
        using var cts = new CancellationTokenSource();
        var task = Downloader().EnsureAsync(dir, true, cancellation: cts.Token);
        for (int i = 0; i < 100 && !(File.Exists(Target + ".part") && new FileInfo(Target + ".part").Length >= 200_000); i++) await Task.Delay(50);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.False(File.Exists(Target));
        Assert.True(new FileInfo(Target + ".part").Length >= 200_000);
        server.Hook = null;
        await Downloader().EnsureAsync(dir, true);   // the next run resumes
        Assert.Equal(content, File.ReadAllBytes(Target));
    }

    [Fact]
    public async Task NonCommercialRepository_IsRefusedWithoutAcceptance_AndNoRequestIsMade()
    {
        var mulan = file with { Repository = ModelManifest.MuLanRepository };
        var d = new ModelDownloader(null, new DownloaderOptions { Endpoint = server.Endpoint }, new[] { mulan });
        var e = await Assert.ThrowsAsync<LicenseNotAcceptedException>(() => d.EnsureAsync(dir, false));
        Assert.Equal(0, server.Requests);
        Assert.Contains("CC-BY-NC-4.0", e.Message);
        Assert.Contains(e.Licenses, l => l.NonCommercial);
    }

    [Fact]
    public async Task OpenRepositories_NeedNoAcceptance()
    {
        var open = file with { Repository = ModelManifest.XlmRobertaRepository };
        server.Path = $"/{open.Repository}/resolve/rev1/data.bin";
        var d = new ModelDownloader(null, new DownloaderOptions { Endpoint = server.Endpoint }, new[] { open });
        Assert.Equal(FetchStatus.Downloaded, Assert.Single(await d.EnsureAsync(dir, false)).Status);
    }

    [Fact]
    public void TheManifest_HasTheSevenFilesAndTheirLicenses()
    {
        Assert.Equal(6, ModelManifest.Files.Count);   // config, decoder.json, model.safetensors and the three checkpoints to convert
        Assert.InRange(ModelManifest.TotalBytes / 1e9, 7.7, 7.8);
        Assert.Equal(new[] { "Apache-2.0", "CC-BY-NC-4.0", "MIT" }, ModelManifest.Licenses.Select(l => l.License).OrderBy(x => x));
        Assert.All(ModelManifest.Files, f => { Assert.Equal(64, f.Sha256.Length); Assert.Equal(40, f.Revision.Length); });
    }

    /// <summary>Optional: the real hub, the smallest file (262 bytes). Runs only with DIFFRHYTHM2_NETWORK_TESTS=1.</summary>
    [Fact]
    public async Task RealHub_ServesTheSmallestPinnedFile()
    {
        if (Environment.GetEnvironmentVariable("DIFFRHYTHM2_NETWORK_TESTS") != "1") { Console.WriteLine("skipped: set DIFFRHYTHM2_NETWORK_TESTS=1 to reach huggingface.co"); return; }
        var config = ModelManifest.Files.Single(f => f.File == "config.json");
        var r = await new ModelDownloader(files: new[] { config }).EnsureAsync(dir, licensesAccepted: false);
        Assert.Equal(FetchStatus.Downloaded, Assert.Single(r).Status);
        Assert.Equal(config.Size, new FileInfo(System.IO.Path.Combine(dir, "config.json")).Length);
    }

    private sealed class Progress : IProgress<DownloadEvent>
    {
        private readonly Action<DownloadEvent> a;
        public Progress(Action<DownloadEvent> a) => this.a = a;
        public void Report(DownloadEvent value) => a(value);
    }
}
