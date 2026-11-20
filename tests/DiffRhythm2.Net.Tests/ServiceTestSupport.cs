using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace DiffRhythm2.Tests;

/// <summary>A song maker that does not need models: it writes a tiny WAV, and the lyrics steer what it does (FAIL, SLOW, TWO).</summary>
internal sealed class FakeMaker : ISongMaker
{
    public readonly List<SongRequest> Made = new();
    public readonly SemaphoreSlim Started = new(0);
    public readonly SemaphoreSlim Finish = new(0);
    public volatile int Concurrent, MaxConcurrent;

    public static byte[] Wav()
    {
        var b = new byte[44 + 8];
        Encoding.ASCII.GetBytes("RIFF").CopyTo(b, 0);
        BitConverter.GetBytes(b.Length - 8).CopyTo(b, 4);
        Encoding.ASCII.GetBytes("WAVEfmt ").CopyTo(b, 8);
        BitConverter.GetBytes(16).CopyTo(b, 16);
        BitConverter.GetBytes((short)1).CopyTo(b, 20);
        BitConverter.GetBytes((short)2).CopyTo(b, 22);
        BitConverter.GetBytes(48000).CopyTo(b, 24);
        BitConverter.GetBytes(48000 * 4).CopyTo(b, 28);
        BitConverter.GetBytes((short)4).CopyTo(b, 32);
        BitConverter.GetBytes((short)16).CopyTo(b, 34);
        Encoding.ASCII.GetBytes("data").CopyTo(b, 36);
        BitConverter.GetBytes(8).CopyTo(b, 40);
        return b;
    }

    public void Make(SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation)
    {
        lock (Made) Made.Add(request);
        int now = Interlocked.Increment(ref Concurrent);
        MaxConcurrent = Math.Max(MaxConcurrent, now);
        try
        {
            Started.Release();
            progress.Report((0, 3));
            if (request.Lyrics.Contains("FAIL")) throw new InvalidOperationException("this song cannot be made");
            progress.Report((1, 3));
            if (request.Lyrics.Contains("SLOW"))
            {
                // waits until the test lets it finish, or the job is cancelled
                while (!Finish.Wait(20)) cancellation.ThrowIfCancellationRequested();
            }
            progress.Report((2, 3));
            progress.Report((3, 3));
            File.WriteAllBytes(wavPath, Wav());
        }
        finally { Interlocked.Decrement(ref Concurrent); }
    }
}

/// <summary>A real Kestrel on a free local port serving the API with a given maker.</summary>
internal sealed class ServiceHost : IDisposable
{
    public readonly string Directory = System.IO.Directory.CreateTempSubdirectory("dr2-svc-").FullName;
    public readonly FakeMaker Maker;
    public readonly JobEngine Engine;
    public readonly HealthProbe Health;
    public readonly HttpClient Client;
    public readonly ServiceSettings Settings;
    private readonly WebApplication app;

    public ServiceHost(string token = "", FakeMaker? maker = null, ISongMaker? real = null, string? model = null, string device = "cpu")
    {
        Maker = maker ?? new FakeMaker();
        Settings = new ServiceSettings(model ?? System.IO.Path.Combine(Directory, "models"), token, 1, System.IO.Path.Combine(Directory, "jobs"), "test-1", device);
        Engine = new JobEngine(real ?? Maker, Settings.JobsDirectory);
        Health = new HealthProbe(Settings, Engine, TimeSpan.Zero)
        {
            MissingFiles = () => Array.Empty<string>(),
            PhonemizerProblem = () => null,
            DeviceProblem = () => null,
            Gpu = () => null,
        };
        app = ServiceApp.Build(new ServiceApp(Settings, Engine, Health), "http://127.0.0.1:0");
        app.Start();
        Client = new HttpClient { BaseAddress = new Uri(app.Urls.First()), Timeout = TimeSpan.FromSeconds(30) };
        if (token.Length > 0) Client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
    }

    public static StringContent Body(object o) => new(JsonSerializer.Serialize(o), Encoding.UTF8, "application/json");

    public async Task<(int Status, JsonElement Json)> Send(HttpMethod method, string path, HttpContent? content = null)
    {
        using var r = await Client.SendAsync(new HttpRequestMessage(method, path) { Content = content });
        var text = await r.Content.ReadAsStringAsync();
        return ((int)r.StatusCode, text.Length > 0 && r.Content.Headers.ContentType?.MediaType == "application/json" ? JsonDocument.Parse(text).RootElement.Clone() : default);
    }

    public Task<(int Status, JsonElement Json)> Get(string path) => Send(HttpMethod.Get, path);

    public async Task<string> Post(object body)
    {
        var (status, json) = await Send(HttpMethod.Post, "/generate", Body(body));
        Assert.Equal(202, status);
        return json.GetProperty("id").GetString()!;
    }

    public async Task<JsonElement> WaitFor(string id, string state, int seconds = 10)
    {
        for (var until = DateTime.UtcNow.AddSeconds(seconds); DateTime.UtcNow < until;)
        {
            var (_, j) = await Get("/jobs/" + id);
            if (j.ValueKind == JsonValueKind.Object && j.TryGetProperty("state", out var s) && s.GetString() == state) return j;
            await Task.Delay(10);
        }
        throw new TimeoutException($"job {id} never reached {state}");
    }

    public static object Song(string lyrics = "[verse]\nHello world tonight\nWe sing along", string style = "Pop, Piano") => new { lyrics, style };

    public void Dispose()
    {
        Maker.Finish.Release(100);
        Client.Dispose();
        app.StopAsync().GetAwaiter().GetResult();
        Engine.Dispose();
        try { System.IO.Directory.Delete(Directory, true); } catch (IOException) { }
    }
}
