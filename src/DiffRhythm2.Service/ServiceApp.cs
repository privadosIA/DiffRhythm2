using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace DiffRhythm2.Service;

/// <summary>The HTTP interface of the service: the five calls of the Python service, with its paths, bodies and status codes.</summary>
internal sealed partial class ServiceApp
{
    public const int MaxBodyBytes = 1_000_000;

    /// <summary>The calls this class answers (keep in step with <see cref="Handle"/>; a test compares it with the OpenAPI description and probes the service).</summary>
    public static readonly string[] RouteTable = { "GET /health", "POST /generate", "GET /jobs/{id}", "DELETE /jobs/{id}", "GET /jobs/{id}/audio" };

    private readonly ServiceSettings settings;
    private readonly JobEngine engine;
    private readonly HealthProbe health;
    private readonly byte[] expectedAuth;

    public ServiceApp(ServiceSettings settings, JobEngine engine, HealthProbe health)
    {
        this.settings = settings;
        this.engine = engine;
        this.health = health;
        expectedAuth = Encoding.UTF8.GetBytes("Bearer " + settings.Token);
    }

    /// <summary>Builds the web application listening on <paramref name="url"/>.</summary>
    public static WebApplication Build(ServiceApp api, string url)
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls(url);
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        var app = builder.Build();
        app.Run(api.Handle);
        return app;
    }

    [GeneratedRegex("^/jobs/([0-9a-f]+)(/audio)?$")]
    private static partial Regex JobPath();

    private async Task Handle(HttpContext ctx)
    {
        try { await Dispatch(ctx); }
        catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested) { /* the client went away: nobody is left to answer */ }
    }

    private async Task Dispatch(HttpContext ctx)
    {
        var path = ctx.Request.Path.Value ?? "/";
        if (path.Length > 1) path = path.TrimEnd('/');
        var method = ctx.Request.Method;

        if (await Docs.TryServe(ctx, path, method)) return;

        if (settings.Token.Length > 0 && !Authorized(ctx)) { await Json(ctx, 401, new ErrorBody("a bearer token is required")); return; }

        if (method == "GET" && path == "/health") { await Json(ctx, 200, health.Get()); return; }
        if (method == "POST" && path == "/generate") { await Generate(ctx); return; }

        var m = JobPath().Match(path);
        if (m.Success)
        {
            var id = m.Groups[1].Value;
            var view = engine.View(id);
            if (view is null) { await Json(ctx, 404, new ErrorBody("unknown job")); return; }
            bool audio = m.Groups[2].Success;
            if (method == "DELETE" && !audio) { engine.Forget(id); await Json(ctx, 200, new OkBody(true)); return; }
            if (method == "GET" && !audio) { await Json(ctx, 200, JobBody.Of(view)); return; }
            if (method == "GET" && audio) { await Audio(ctx, id); return; }
        }
        await Json(ctx, 404, new ErrorBody("not found"));
    }

    private bool Authorized(HttpContext ctx)
    {
        var given = Encoding.UTF8.GetBytes(ctx.Request.Headers.Authorization.ToString());
        return given.Length == expectedAuth.Length && CryptographicOperations.FixedTimeEquals(given, expectedAuth);
    }

    private async Task Generate(HttpContext ctx)
    {
        if (ctx.Request.ContentLength > MaxBodyBytes) { await Json(ctx, 413, new ErrorBody("the request is too large")); return; }
        var buffer = new MemoryStream();
        var chunk = new byte[16 * 1024];
        int n;
        try
        {
            while ((n = await ctx.Request.Body.ReadAsync(chunk, ctx.RequestAborted)) > 0)
            {
                await buffer.WriteAsync(chunk.AsMemory(0, n), ctx.RequestAborted);
                if (buffer.Length > MaxBodyBytes) { await Json(ctx, 413, new ErrorBody("the request is too large")); return; }
            }
        }
        // the client closed the connection (or stopped) before sending the length it declared: nothing to serve, and not an error of the service
        catch (BadHttpRequestException) { await Json(ctx, 400, new ErrorBody("the request body ended before its declared length")); return; }
        Pipeline.SongRequest request;
        try { request = RequestParser.Parse(buffer.GetBuffer().AsSpan(0, (int)buffer.Length)); }
        catch (BadRequestException e) { await Json(ctx, 400, new ErrorBody(e.Message)); return; }
        var job = engine.Submit(request);
        await Json(ctx, 202, new IdBody(job.Id));
    }

    private async Task Audio(HttpContext ctx, string id)
    {
        var audio = engine.Audio(id);
        if (audio is null) { await Json(ctx, 404, new ErrorBody("unknown job")); return; }
        if (audio.Value.Path is null) { await Json(ctx, 409, new ErrorBody($"the job is {audio.Value.State}")); return; }
        FileStream file;
        try { file = new FileStream(audio.Value.Path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete); }
        catch (IOException) { await Json(ctx, 410, new ErrorBody("the audio is gone")); return; }
        await using (file)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = "audio/wav";
            ctx.Response.ContentLength = file.Length;
            await file.CopyToAsync(ctx.Response.Body, ctx.RequestAborted);
        }
    }

    // ---- JSON ----------------------------------------------------------------------------------------------------------------------------

    private static readonly JsonSerializerOptions JsonOptions = new() { DefaultIgnoreCondition = JsonIgnoreCondition.Never };

    internal static async Task Json<T>(HttpContext ctx, int status, T body)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(body, JsonOptions);
        ctx.Response.StatusCode = status;
        ctx.Response.ContentType = "application/json";
        ctx.Response.ContentLength = bytes.Length;
        await ctx.Response.Body.WriteAsync(bytes, ctx.RequestAborted);
    }

    internal sealed record ErrorBody([property: JsonPropertyName("error")] string Error);
    internal sealed record OkBody([property: JsonPropertyName("ok")] bool Ok);
    internal sealed record IdBody([property: JsonPropertyName("id")] string Id);
    internal sealed record ProgressBody([property: JsonPropertyName("done")] int Done, [property: JsonPropertyName("total")] int Total);
    internal sealed record JobBody(
        [property: JsonPropertyName("id")] string Id, [property: JsonPropertyName("state")] string State, [property: JsonPropertyName("progress")] ProgressBody Progress,
        [property: JsonPropertyName("error")] string Error, [property: JsonPropertyName("seconds")] double Seconds)
    {
        public static JobBody Of(JobView v) => new(v.Id, v.State, new ProgressBody(v.Done, v.Total), v.Error, v.Seconds);
    }
}

/// <summary>The documentation endpoints (<c>/docs</c>, <c>/openapi.yaml</c>), served from the executable itself, without a token.</summary>
internal static class Docs
{
    private static readonly System.Reflection.Assembly Assembly = typeof(Docs).Assembly;

    private static string ContentType(string name) => System.IO.Path.GetExtension(name) switch
    {
        ".html" => "text/html; charset=utf-8",
        ".js" => "text/javascript; charset=utf-8",
        ".css" => "text/css; charset=utf-8",
        ".png" => "image/png",
        ".yaml" => "application/yaml; charset=utf-8",
        _ => "text/plain; charset=utf-8",
    };

    /// <summary>The text of the OpenAPI description.</summary>
    public static string OpenApi
    {
        get
        {
            using var s = Assembly.GetManifestResourceStream("openapi.yaml")!;
            using var r = new StreamReader(s, Encoding.UTF8);
            return r.ReadToEnd();
        }
    }

    /// <summary>Answers a documentation request (<c>/openapi.yaml</c>, <c>/docs</c>); false when the path or method is not one of them, so the caller goes on.</summary>
    public static async Task<bool> TryServe(HttpContext ctx, string path, string method)
    {
        string? resource = path switch
        {
            "/openapi.yaml" => "openapi.yaml",
            "/docs" => "docs/index.html",
            _ when path.StartsWith("/docs/", StringComparison.Ordinal) && !path.Contains("..") => "docs/" + path["/docs/".Length..],
            _ => null,
        };
        if (resource is null || (method != "GET" && method != "HEAD")) return false;
        var stream = Assembly.GetManifestResourceStream(resource);
        if (stream is null) return false;
        await Send(ctx, stream, resource);
        return true;
    }

    private static async Task Send(HttpContext ctx, Stream stream, string resource)
    {
        await using (stream)
        {
            ctx.Response.StatusCode = 200;
            ctx.Response.ContentType = ContentType(resource);
            ctx.Response.ContentLength = stream.Length;
            if (ctx.Request.Method != "HEAD") await stream.CopyToAsync(ctx.Response.Body, ctx.RequestAborted);
        }
    }
}
