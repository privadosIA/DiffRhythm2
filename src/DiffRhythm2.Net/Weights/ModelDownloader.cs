using System.Net;
using System.Security.Cryptography;

namespace DiffRhythm2.Weights;

/// <summary>What happened to a file.</summary>
public enum FetchStatus
{
    /// <summary>The file was downloaded and verified.</summary>
    Downloaded,
    /// <summary>A file with the right size and hash was already there.</summary>
    Present,
}

/// <summary>The outcome for one file of a fetch.</summary>
/// <param name="File">The manifest entry.</param>
/// <param name="Status">Downloaded or already present.</param>
public sealed record FetchResult(ManifestFile File, FetchStatus Status);

/// <summary>The kinds of <see cref="DownloadEvent"/>.</summary>
public enum DownloadEventKind
{
    /// <summary>Work on a file starts.</summary>
    Step,
    /// <summary>The file is already complete.</summary>
    Present,
    /// <summary>Bytes received so far.</summary>
    Bytes,
    /// <summary>A transient failure; the download will be retried.</summary>
    Retry,
}

/// <summary>Progress of a fetch.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="File">The file concerned.</param>
/// <param name="Done">Bytes so far (<see cref="DownloadEventKind.Bytes"/>).</param>
/// <param name="Message">Detail for a retry.</param>
public sealed record DownloadEvent(DownloadEventKind Kind, ManifestFile File, long Done = 0, string? Message = null);

/// <summary>A download that failed for good: the server refused it, or the bytes are not the expected ones.</summary>
public sealed class DownloadException : IOException
{
    /// <summary>Creates the exception.</summary>
    public DownloadException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>Files of a non-commercial repository were requested without the licenses being accepted.</summary>
public sealed class LicenseNotAcceptedException : InvalidOperationException
{
    /// <summary>The licenses of the repositories that were requested.</summary>
    public IReadOnlyList<RepositoryLicense> Licenses { get; }

    /// <summary>Creates the exception.</summary>
    public LicenseNotAcceptedException(IReadOnlyList<RepositoryLicense> licenses) : base(
        "the licenses must be accepted before downloading: " + string.Join("; ", licenses.Select(l => $"{l.Repository} is {l.License}{(l.NonCommercial ? " (non-commercial use only)" : "")}")))
        => Licenses = licenses;
}

/// <summary>Settings of <see cref="ModelDownloader"/>.</summary>
public sealed class DownloaderOptions
{
    /// <summary>The default settings.</summary>
    public DownloaderOptions() { }

    /// <summary>Base address of the hub; defaults to <c>HF_ENDPOINT</c> or <c>https://huggingface.co</c>.</summary>
    public string Endpoint { get; init; } = Environment.GetEnvironmentVariable("HF_ENDPOINT") is { Length: > 0 } e ? e.TrimEnd('/') : "https://huggingface.co";
    /// <summary>Attempts per file for transient failures.</summary>
    public int MaxAttempts { get; init; } = 5;
    /// <summary>Delay before the first retry; it doubles each time.</summary>
    public TimeSpan BaseDelay { get; init; } = TimeSpan.FromSeconds(1);
    /// <summary>How long the connection may stay silent before the attempt is abandoned.</summary>
    public TimeSpan StallTimeout { get; init; } = TimeSpan.FromSeconds(60);
}

/// <summary>
/// Downloads the model files from pinned revisions. Each file is written to <c>NAME.part</c>, resumed with a range request if it exists, checked
/// against the recorded size and SHA-256 and only then given its name. Transient failures are retried with increasing delays; a refusal by the
/// server is not. Files of a non-commercial repository are not fetched unless the licenses are accepted.
/// </summary>
public sealed class ModelDownloader
{
    private const int BufferSize = 1 << 20;

    private readonly HttpClient client;
    private readonly DownloaderOptions options;
    private readonly IReadOnlyList<ManifestFile> files;

    /// <summary>Creates a downloader. <paramref name="client"/> defaults to a client with the system proxy and no overall timeout; <paramref name="files"/> to <see cref="ModelManifest.Files"/>.</summary>
    public ModelDownloader(HttpClient? client = null, DownloaderOptions? options = null, IReadOnlyList<ManifestFile>? files = null)
    {
        this.client = client ?? new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        this.options = options ?? new DownloaderOptions();
        this.files = files ?? ModelManifest.Files;
    }

    /// <summary>The licenses of the repositories the given files come from.</summary>
    public static IReadOnlyList<RepositoryLicense> LicensesOf(IEnumerable<ManifestFile> files)
    {
        var repos = files.Select(f => f.Repository).Distinct().ToHashSet();
        return ModelManifest.Licenses.Where(l => repos.Contains(l.Repository)).ToList();
    }

    /// <summary>
    /// Makes sure the files are in <paramref name="modelDir"/>. Throws <see cref="LicenseNotAcceptedException"/>, before any request,
    /// if a non-commercial repository is involved and <paramref name="licensesAccepted"/> is false.
    /// </summary>
    public async Task<IReadOnlyList<FetchResult>> EnsureAsync(string modelDir, bool licensesAccepted, IReadOnlyList<ManifestFile>? only = null, IProgress<DownloadEvent>? progress = null, CancellationToken cancellation = default)
    {
        var wanted = only ?? files;
        var licenses = LicensesOf(wanted);
        if (!licensesAccepted && licenses.Any(l => l.NonCommercial)) throw new LicenseNotAcceptedException(licenses);

        var results = new List<FetchResult>();
        foreach (var file in wanted)
        {
            cancellation.ThrowIfCancellationRequested();
            progress?.Report(new DownloadEvent(DownloadEventKind.Step, file));
            var target = Path.Combine(modelDir, file.Target);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            if (new FileInfo(target) is { Exists: true } info && info.Length == file.Size && await HashAsync(target, cancellation) == file.Sha256)
            {
                progress?.Report(new DownloadEvent(DownloadEventKind.Present, file));
                results.Add(new FetchResult(file, FetchStatus.Present));
                continue;
            }
            await DownloadAsync(file, target, progress, cancellation);
            results.Add(new FetchResult(file, FetchStatus.Downloaded));
        }
        return results;
    }

    private static async Task<string> HashAsync(string path, CancellationToken ct)
    {
        await using var f = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(f, ct)).ToLowerInvariant();
    }

    private Uri UrlOf(ManifestFile f) => new($"{options.Endpoint}/{f.Repository}/resolve/{f.Revision}/{f.File}");

    private async Task DownloadAsync(ManifestFile file, string target, IProgress<DownloadEvent>? progress, CancellationToken ct)
    {
        var part = target + ".part";
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                await AttemptAsync(file, part, progress, ct);
                File.Move(part, target, overwrite: true);
                return;
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
            catch (Exception e) when (IsTransient(e) && attempt < options.MaxAttempts)
            {
                var delay = TimeSpan.FromTicks(options.BaseDelay.Ticks << (attempt - 1));
                progress?.Report(new DownloadEvent(DownloadEventKind.Retry, file, Message: $"{Describe(e)}; retrying in {delay.TotalSeconds:0.#} s (attempt {attempt + 1} of {options.MaxAttempts})"));
                await Task.Delay(delay, ct);
            }
            catch (Exception e) when (IsTransient(e))
            {
                throw new DownloadException($"{file.Repository}/{file.File}: giving up after {options.MaxAttempts} attempts: {Describe(e)}", e);
            }
        }
    }

    private sealed class TransientHttpException : Exception
    {
        public TransientHttpException(string message) : base(message) { }
    }

    private static bool IsTransient(Exception e) => e is HttpRequestException or IOException or TransientHttpException or OperationCanceledException && e is not DownloadException;

    private static string Describe(Exception e) => e is OperationCanceledException ? "the connection stalled" : e.Message;

    private async Task AttemptAsync(ManifestFile file, string part, IProgress<DownloadEvent>? progress, CancellationToken ct)
    {
        long have = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (have > file.Size) { File.Delete(part); have = 0; }

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        if (have < file.Size)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, UrlOf(file));
            if (have > 0) request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(have, null);
            using var stall = CancellationTokenSource.CreateLinkedTokenSource(ct);
            stall.CancelAfter(options.StallTimeout);
            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, stall.Token);
            var status = (int)response.StatusCode;
            if (status == 416) { File.Delete(part); throw new TransientHttpException("the server rejected the range; starting again"); }
            if (status >= 500 || status == 429) throw new TransientHttpException($"{request.RequestUri} answered {status} {response.ReasonPhrase}");
            if (!response.IsSuccessStatusCode) throw new DownloadException($"{request.RequestUri} answered {status} {response.ReasonPhrase}");
            bool resumed = response.StatusCode == HttpStatusCode.PartialContent;
            if (!resumed) have = 0;   // the server ignored the range and sends everything again

            if (resumed) await HashPrefixAsync(part, hash, ct);
            await using var body = await response.Content.ReadAsStreamAsync(stall.Token);
            await using var output = new FileStream(part, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, BufferSize);
            var buffer = new byte[BufferSize];
            long done = have;
            var lastReport = DateTime.MinValue;
            while (true)
            {
                stall.CancelAfter(options.StallTimeout);
                int n = await body.ReadAsync(buffer, stall.Token);
                if (n == 0) break;
                await output.WriteAsync(buffer.AsMemory(0, n), ct);
                hash.AppendData(buffer, 0, n);
                done += n;
                if (DateTime.UtcNow - lastReport >= TimeSpan.FromSeconds(1)) { progress?.Report(new DownloadEvent(DownloadEventKind.Bytes, file, done)); lastReport = DateTime.UtcNow; }
            }
            progress?.Report(new DownloadEvent(DownloadEventKind.Bytes, file, done));
            have = done;
        }
        else await HashPrefixAsync(part, hash, ct);

        if (have < file.Size) throw new TransientHttpException($"the connection ended after {have} of {file.Size} bytes");
        if (have > file.Size) { File.Delete(part); throw new DownloadException($"{file.Repository}/{file.File}: received {have} bytes, {file.Size} expected"); }
        var actual = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
        if (actual != file.Sha256)
        {
            File.Delete(part);
            throw new DownloadException($"{file.Repository}/{file.File}: the downloaded file is not the expected one (sha256 {actual}, expected {file.Sha256}); it was deleted");
        }
    }

    private static async Task HashPrefixAsync(string part, IncrementalHash hash, CancellationToken ct)
    {
        await using var f = File.OpenRead(part);
        var buffer = new byte[BufferSize];
        int n;
        while ((n = await f.ReadAsync(buffer, ct)) > 0) hash.AppendData(buffer, 0, n);
    }
}
