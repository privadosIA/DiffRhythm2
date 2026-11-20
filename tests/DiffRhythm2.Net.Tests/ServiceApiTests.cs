using System.Net;
using System.Text;
using System.Text.Json;
using DiffRhythm2.Service;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>The HTTP interface against the contract of the Python service, over a real Kestrel with a fake song maker.</summary>
public class ServiceApiTests
{
    private readonly ITestOutputHelper log;

    public ServiceApiTests(ITestOutputHelper log) => this.log = log;

    [Fact]
    public async Task ASongFromRequestToFile()
    {
        using var h = new ServiceHost();
        var id = await h.Post(ServiceHost.Song());
        Assert.Matches("^[0-9a-f]{12}$", id);
        var done = await h.WaitFor(id, "done");
        Assert.Equal(3, done.GetProperty("progress").GetProperty("done").GetInt32());
        Assert.Equal(3, done.GetProperty("progress").GetProperty("total").GetInt32());
        Assert.Equal("", done.GetProperty("error").GetString());

        using var audio = await h.Client.GetAsync($"/jobs/{id}/audio");
        Assert.Equal(HttpStatusCode.OK, audio.StatusCode);
        Assert.Equal("audio/wav", audio.Content.Headers.ContentType!.MediaType);
        var bytes = await audio.Content.ReadAsByteArrayAsync();
        Assert.Equal("RIFF", Encoding.ASCII.GetString(bytes, 0, 4));
        Assert.Equal("WAVE", Encoding.ASCII.GetString(bytes, 8, 4));
        Assert.Equal(bytes.Length, audio.Content.Headers.ContentLength);

        Assert.Equal(200, (await h.Send(HttpMethod.Delete, "/jobs/" + id)).Status);
        Assert.Equal(404, (await h.Get("/jobs/" + id)).Status);
    }

    [Fact]
    public async Task Defaults_AreThe210Seconds_Guidance2_16Steps_AndSeed0()
    {
        using var h = new ServiceHost();
        await h.WaitFor(await h.Post(ServiceHost.Song()), "done");
        var r = h.Maker.Made.Single();
        Assert.Equal(210.0, r.Seconds);
        Assert.Equal(2.0, r.CfgStrength);
        Assert.Equal(16, r.Steps);
        Assert.Equal(0L, r.Seed);
    }

    [Fact]
    public async Task AStyleList_IsJoinedWithCommas()
    {
        using var h = new ServiceHost();
        await h.WaitFor(await h.Post(new { lyrics = "[verse]\nHello", style = new[] { "Pop", "Piano" } }), "done");
        Assert.Equal("Pop, Piano", h.Maker.Made.Single().Style);
    }

    [Fact]
    public async Task UnknownJob_And_UnknownPath_Are404()
    {
        using var h = new ServiceHost();
        var (s1, j1) = await h.Get("/jobs/abcdef012345");
        Assert.Equal(404, s1);
        Assert.Equal("unknown job", j1.GetProperty("error").GetString());
        var (s2, j2) = await h.Get("/nope");
        Assert.Equal(404, s2);
        Assert.Equal("not found", j2.GetProperty("error").GetString());
        var (s3, _) = await h.Send(HttpMethod.Put, "/health");
        Assert.Equal(404, s3);
    }

    [Fact]
    public async Task TrailingSlash_IsAccepted()
    {
        using var h = new ServiceHost();
        Assert.Equal(200, (await h.Get("/health/")).Status);
    }

    [Fact]
    public async Task MalformedAndTooLargeBodies()
    {
        using var h = new ServiceHost();
        var (s, j) = await h.Send(HttpMethod.Post, "/generate", new StringContent("not json", Encoding.UTF8, "application/json"));
        Assert.Equal(400, s);
        Assert.NotEqual("", j.GetProperty("error").GetString());
        Assert.Equal(0, h.Engine.Queued);

        var big = new StringContent(new string('x', 1_000_001), Encoding.UTF8, "application/json");
        var (s2, j2) = await h.Send(HttpMethod.Post, "/generate", big);
        Assert.Equal(413, s2);
        Assert.Equal("the request is too large", j2.GetProperty("error").GetString());
    }

    [Fact]
    public async Task AClientThatDropsBeforeItsBodyIsComplete_CreatesNoJob_AndTheServiceKeepsAnswering()
    {
        using var h = new ServiceHost();
        using (var raw = new System.Net.Sockets.TcpClient())
        {
            await raw.ConnectAsync(h.Client.BaseAddress!.Host, h.Client.BaseAddress.Port);
            var head = Encoding.ASCII.GetBytes("POST /generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{\"lyrics\": \"[verse]\\nHel");
            await raw.GetStream().WriteAsync(head);
            await raw.GetStream().FlushAsync();
            await Task.Delay(100);
        }   // closed with 480 bytes of the body still to come
        await Task.Delay(200);
        Assert.Equal(0, h.Engine.Queued);
        Assert.Empty(h.Maker.Made);
        var (status, _) = await h.Get("/health");
        Assert.Equal(200, status);
    }

    [Fact]
    public async Task AClientThatDropsWhileTheAudioIsRequested_DoesNotAffectTheJob()
    {
        using var h = new ServiceHost();
        var id = await h.Post(ServiceHost.Song());
        await h.WaitFor(id, "done");
        using (var raw = new System.Net.Sockets.TcpClient())
        {
            await raw.ConnectAsync(h.Client.BaseAddress!.Host, h.Client.BaseAddress.Port);
            await raw.GetStream().WriteAsync(Encoding.ASCII.GetBytes($"GET /jobs/{id}/audio HTTP/1.1\r\nHost: x\r\n\r\n"));
            await raw.GetStream().FlushAsync();
        }   // closed without reading a byte of the answer
        await Task.Delay(100);
        using var r = await h.Client.GetAsync($"/jobs/{id}/audio");
        Assert.Equal(200, (int)r.StatusCode);
        Assert.Equal(FakeMaker.Wav(), await r.Content.ReadAsByteArrayAsync());
    }

    [Fact]
    public async Task AnEmptyBody_IsAnEmptyObject()
    {
        using var h = new ServiceHost();
        var (s, j) = await h.Send(HttpMethod.Post, "/generate", new StringContent("", Encoding.UTF8, "application/json"));
        Assert.Equal(400, s);
        Assert.Equal("lyrics must be a non-empty text", j.GetProperty("error").GetString());
    }

    [Fact]
    public async Task EveryReferenceBody_GetsTheReferencesOutcome()
    {
        using var h = new ServiceHost();
        int ok = 0, refused = 0;
        foreach (var c in SongReference.Validation.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var body = c.GetProperty("body");
            var (status, json) = await h.Send(HttpMethod.Post, "/generate", new StringContent(body.GetRawText(), Encoding.UTF8, "application/json"));
            if (c.TryGetProperty("error", out var error))
            {
                Assert.True(status == 400, $"{name}: expected 400, got {status}");
                Assert.True(error.GetString() == json.GetProperty("error").GetString(), $"{name}: expected \"{error.GetString()}\", got \"{json.GetProperty("error").GetString()}\"");
                refused++;
            }
            else
            {
                Assert.True(status == 202, $"{name}: expected 202, got {status}: {json}");
                ok++;
                var made = await WaitMade(h, ok);
                var expected = c.GetProperty("ok");
                Assert.True(expected.GetProperty("style").GetString() == made.Validate().Style, $"{name}: style");
                Assert.Equal(expected.GetProperty("steps").GetInt32(), made.Steps);
                Assert.Equal(expected.GetProperty("seed").GetInt64(), made.Seed);
                Assert.Equal(expected.GetProperty("max_secs").GetDouble(), made.Seconds);
                Assert.Equal(expected.GetProperty("cfg_strength").GetDouble(), made.CfgStrength);
            }
        }
        log.WriteLine($"{ok} accepted and {refused} refused, all as the reference");
        Assert.Equal(56, ok + refused);
    }

    private static async Task<Pipeline.SongRequest> WaitMade(ServiceHost h, int count)
    {
        for (int i = 0; i < 500; i++)
        {
            lock (h.Maker.Made) if (h.Maker.Made.Count >= count) return h.Maker.Made[count - 1];
            await Task.Delay(10);
        }
        throw new TimeoutException();
    }

    [Fact]
    public async Task Audio_Before_TheJobIsDone_Is409()
    {
        using var h = new ServiceHost();
        var id = await h.Post(ServiceHost.Song("[verse]\nSLOW song"));
        await h.WaitFor(id, "running");
        var (s, j) = await h.Get($"/jobs/{id}/audio");
        Assert.Equal(409, s);
        Assert.Equal("the job is running", j.GetProperty("error").GetString());
        h.Maker.Finish.Release();
        await h.WaitFor(id, "done");
    }

    [Fact]
    public async Task Audio_OfADoneJobWhoseFileIsGone_Is410()
    {
        using var h = new ServiceHost();
        var id = await h.Post(ServiceHost.Song());
        await h.WaitFor(id, "done");
        File.Delete(Path.Combine(h.Settings.JobsDirectory, id + ".wav"));
        var (s, j) = await h.Get($"/jobs/{id}/audio");
        Assert.Equal(410, s);
        Assert.Equal("the audio is gone", j.GetProperty("error").GetString());
    }

    [Fact]
    public async Task Token_IsRequiredOnEveryCall_WhenConfigured()
    {
        using var h = new ServiceHost(token: "s3cret");
        using var anonymous = new HttpClient { BaseAddress = h.Client.BaseAddress };
        foreach (var path in new[] { "/health", "/jobs/abcdef012345", "/nope" })
        {
            using var r = await anonymous.GetAsync(path);
            Assert.Equal(HttpStatusCode.Unauthorized, r.StatusCode);
            Assert.Contains("a bearer token is required", await r.Content.ReadAsStringAsync());
        }
        using var post = await anonymous.PostAsync("/generate", ServiceHost.Body(ServiceHost.Song()));
        Assert.Equal(HttpStatusCode.Unauthorized, post.StatusCode);
        using var wrong = new HttpClient { BaseAddress = h.Client.BaseAddress };
        wrong.DefaultRequestHeaders.Authorization = new("Bearer", "other");
        Assert.Equal(HttpStatusCode.Unauthorized, (await wrong.GetAsync("/health")).StatusCode);
        Assert.Equal(200, (await h.Get("/health")).Status);   // the right token
    }

    [Fact]
    public async Task NoToken_MeansAnOpenApi()
    {
        using var h = new ServiceHost();
        using var anonymous = new HttpClient { BaseAddress = h.Client.BaseAddress };
        Assert.Equal(HttpStatusCode.OK, (await anonymous.GetAsync("/health")).StatusCode);
    }

    [Fact]
    public async Task Health_ReportsReadiness_Missing_AndTheQueue()
    {
        using var h = new ServiceHost();
        var (_, ok) = await h.Get("/health");
        Assert.True(ok.GetProperty("ready").GetBoolean());
        Assert.Equal("test-1", ok.GetProperty("version").GetString());
        Assert.Equal(JsonValueKind.Null, ok.GetProperty("gpu").ValueKind);
        Assert.Equal(0, ok.GetProperty("missing").GetArrayLength());

        var id = await h.Post(ServiceHost.Song("[verse]\nSLOW song"));
        await h.WaitFor(id, "running");
        await h.Post(ServiceHost.Song());
        var (_, busy) = await h.Get("/health");   // answers while a song is being made
        Assert.True(busy.GetProperty("busy").GetBoolean());
        Assert.Equal(1, busy.GetProperty("queued").GetInt32());
    }

    [Fact]
    public async Task Health_NotReady_ListsTheMissingFiles()
    {
        using var h = new ServiceHost();
        var probe = new HealthProbe(h.Settings, h.Engine, TimeSpan.Zero)
        {
            MissingFiles = () => new[] { "decoder.safetensors" },
            PhonemizerProblem = () => null,
            DeviceProblem = () => null,
            Gpu = () => new GpuView("cuda:0", null, null),
        };
        var view = probe.Get();
        Assert.False(view.Ready);
        Assert.Equal(new[] { "decoder.safetensors" }, view.Missing);
        Assert.Equal("cuda:0", view.Gpu!.Name);
    }

    [Fact]
    public void Health_WhenThePhonemizerOrTheDeviceFails_IsNotReady()
    {
        using var h = new ServiceHost();
        Assert.False(new HealthProbe(h.Settings, h.Engine, TimeSpan.Zero) { MissingFiles = () => Array.Empty<string>(), PhonemizerProblem = () => "no libespeak", DeviceProblem = () => null, Gpu = () => null }.Get().Ready);
        Assert.False(new HealthProbe(h.Settings, h.Engine, TimeSpan.Zero) { MissingFiles = () => Array.Empty<string>(), PhonemizerProblem = () => null, DeviceProblem = () => "no cuda", Gpu = () => null }.Get().Ready);
    }

    [Fact]
    public void Health_WithoutAModelDirectory_ListsAllSixFiles()
    {
        using var h = new ServiceHost();
        var probe = new HealthProbe(h.Settings, h.Engine, TimeSpan.Zero) { PhonemizerProblem = () => null, DeviceProblem = () => null, Gpu = () => null };
        var view = probe.Get();   // the directory does not exist
        Assert.False(view.Ready);
        Assert.Equal(6, view.Missing.Count);
    }

    [Fact]
    public void NvidiaSmiOutput_IsParsed()
    {
        var gpu = HealthProbe.ParseSmi("NVIDIA GeForce RTX 4070, 11200, 12282\n");
        Assert.Equal(new GpuView("NVIDIA GeForce RTX 4070", 11200, 12282), gpu);
        Assert.Equal(new GpuView("Some GPU", null, 8000), HealthProbe.ParseSmi("Some GPU, [N/A], 8000"));
        Assert.Null(HealthProbe.ParseSmi(""));
        Assert.Null(HealthProbe.ParseSmi("garbage"));
    }
}
