using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;
using MusicGen.Core.Testing;

namespace DiffRhythm2.Tests;

/// <summary>The OpenAPI description against the service: the same routes, the same status codes, the same defaults and limits, and the documentation served offline.</summary>
public class OpenApiContractTests
{
    private static readonly string Yaml = Docs.OpenApi;

    /// <summary>"METHOD /path" -> declared status codes, read with the layout convention stated at the top of the file.</summary>
    private static Dictionary<string, HashSet<string>> Operations()
    {
        var ops = new Dictionary<string, HashSet<string>>();
        bool inPaths = false, inResponses = false;
        string? path = null, key = null;
        foreach (var raw in Yaml.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.StartsWith("paths:", StringComparison.Ordinal)) { inPaths = true; continue; }
            if (inPaths && line.Length > 0 && !line.StartsWith(' ') && !line.StartsWith('#')) break;
            if (!inPaths) continue;
            var m = Regex.Match(line, @"^  (/\S*):$");
            if (m.Success) { path = m.Groups[1].Value; key = null; inResponses = false; continue; }
            m = Regex.Match(line, @"^    (get|post|put|patch|delete):$");
            if (m.Success) { key = $"{m.Groups[1].Value.ToUpperInvariant()} {path}"; ops[key] = new(); inResponses = false; continue; }
            if (line == "      responses:") { inResponses = true; continue; }
            if (key is not null && inResponses && Regex.Match(line, @"^        '(\d{3})':$") is { Success: true } s) ops[key].Add(s.Groups[1].Value);
        }
        return ops;
    }

    /// <summary>Properties of the SongRequest schema: field -> (key -> value).</summary>
    private static Dictionary<string, Dictionary<string, string>> RequestProperties()
    {
        var props = new Dictionary<string, Dictionary<string, string>>();
        bool inSchema = false, inProps = false;
        string? field = null;
        foreach (var raw in Yaml.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line == "    SongRequest:") { inSchema = true; continue; }
            if (inSchema && Regex.IsMatch(line, @"^    \S")) break;
            if (!inSchema) continue;
            if (line == "      properties:") { inProps = true; continue; }
            if (!inProps) continue;
            var f = Regex.Match(line, @"^        (\w+):$");
            if (f.Success) { field = f.Groups[1].Value; props[field] = new(); continue; }
            var kv = Regex.Match(line, @"^          (\w+): (\S+)$");
            if (kv.Success && field is not null) props[field][kv.Groups[1].Value] = kv.Groups[2].Value;
        }
        return props;
    }

    [Fact]
    public void TheDescription_ListsTheFiveCalls_AndTheServiceTableAgrees()
    {
        var ops = Operations();
        Assert.Equal(ServiceApp.RouteTable.OrderBy(x => x), ops.Keys.OrderBy(x => x));
    }

    [Fact]
    public async Task EveryRouteTheServiceAnswers_IsInTheDescription()
    {
        using var h = new ServiceHost();
        var id = await h.Post(ServiceHost.Song());
        await h.WaitFor(id, "done");
        var declared = Operations().Keys.ToHashSet();
        var served = new HashSet<string>();
        foreach (var method in new[] { "GET", "POST", "PUT", "PATCH", "DELETE" })
            foreach (var template in new[] { "/health", "/generate", "/jobs/{id}", "/jobs/{id}/audio", "/jobs", "/metrics", "/status", "/", "/jobs/{id}/cancel", "/api/health", "/v1/generate" })
            {
                string target = template;
                if (method == "DELETE" && template == "/jobs/{id}") target = "/jobs/" + await h.Post(ServiceHost.Song());   // a job of its own: the call deletes it
                else target = template.Replace("{id}", id);
                using var request = new HttpRequestMessage(new HttpMethod(method), target) { Content = method == "POST" ? new StringContent("{}") : null };
                using var response = await h.Client.SendAsync(request);
                var text = await response.Content.ReadAsStringAsync();
                bool notFound = response.StatusCode == HttpStatusCode.NotFound && text.Contains("\"not found\"");
                if (!notFound) served.Add($"{method} {template}");
            }
        Assert.Equal(declared.OrderBy(x => x), served.OrderBy(x => x));
    }

    [Fact]
    public async Task TheStatusCodesTheServiceAnswers_AreExactlyTheDeclaredOnes()
    {
        using var h = new ServiceHost(token: "tok");
        using var anonymous = new HttpClient { BaseAddress = h.Client.BaseAddress };
        var seen = new Dictionary<string, HashSet<string>>();
        void See(string op, int status) { if (!seen.TryGetValue(op, out var set)) seen[op] = set = new(); set.Add(status.ToString()); }
        async Task<int> Anon(HttpMethod m, string path) { using var r = await anonymous.SendAsync(new HttpRequestMessage(m, path)); return (int)r.StatusCode; }

        See("GET /health", (await h.Get("/health")).Status);
        See("GET /health", await Anon(HttpMethod.Get, "/health"));

        See("POST /generate", (await h.Send(HttpMethod.Post, "/generate", ServiceHost.Body(ServiceHost.Song()))).Status);
        See("POST /generate", (await h.Send(HttpMethod.Post, "/generate", ServiceHost.Body(new { lyrics = "", style = "x" }))).Status);
        See("POST /generate", (await h.Send(HttpMethod.Post, "/generate", new StringContent(new string('x', 1_000_001)))).Status);
        See("POST /generate", await Anon(HttpMethod.Post, "/generate"));

        var slow = await h.Post(ServiceHost.Song("[verse]\nSLOW"));
        await h.WaitFor(slow, "running");
        See("GET /jobs/{id}", (await h.Get("/jobs/" + slow)).Status);
        See("GET /jobs/{id}", (await h.Get("/jobs/abcdef012345")).Status);
        See("GET /jobs/{id}", await Anon(HttpMethod.Get, "/jobs/" + slow));
        See("GET /jobs/{id}/audio", (await h.Get($"/jobs/{slow}/audio")).Status);            // 409
        See("GET /jobs/{id}/audio", (await h.Get("/jobs/abcdef012345/audio")).Status);        // 404
        See("GET /jobs/{id}/audio", await Anon(HttpMethod.Get, $"/jobs/{slow}/audio"));       // 401
        h.Maker.Finish.Release();
        await h.WaitFor(slow, "done");
        See("GET /jobs/{id}/audio", (await h.Get($"/jobs/{slow}/audio")).Status);            // 200 (the body is not JSON; Get returns the status)
        File.Delete(Path.Combine(h.Settings.JobsDirectory, slow + ".wav"));
        See("GET /jobs/{id}/audio", (await h.Get($"/jobs/{slow}/audio")).Status);            // 410

        See("DELETE /jobs/{id}", (await h.Send(HttpMethod.Delete, "/jobs/" + slow)).Status);
        See("DELETE /jobs/{id}", (await h.Send(HttpMethod.Delete, "/jobs/" + slow)).Status);
        See("DELETE /jobs/{id}", await Anon(HttpMethod.Delete, "/jobs/" + slow));

        var declared = Operations();
        foreach (var (op, codes) in declared)
        {
            Assert.True(seen.TryGetValue(op, out var got), $"{op} was not exercised");
            Assert.True(codes.SetEquals(got!), $"{op}: declared {string.Join(",", codes.OrderBy(x => x))}, answered {string.Join(",", got!.OrderBy(x => x))}");
        }
    }

    [Fact]
    public void DocumentedDefaultsAndLimits_AreTheOnesTheServiceApplies()
    {
        var p = RequestProperties();
        double D(string field, string key) => double.Parse(p[field][key], System.Globalization.CultureInfo.InvariantCulture);
        Assert.Equal(SongRequest.DefaultSeconds, D("max_secs", "default"));
        Assert.Equal(10.0, D("max_secs", "minimum"));
        Assert.Equal(210.0, D("max_secs", "maximum"));
        Assert.Equal(SongRequest.DefaultCfgStrength, D("cfg_strength", "default"));
        Assert.Equal(0.5, D("cfg_strength", "minimum"));
        Assert.Equal(10.0, D("cfg_strength", "maximum"));
        Assert.Equal(SongRequest.DefaultSteps, (int)D("steps", "default"));
        Assert.Equal(4, (int)D("steps", "minimum"));
        Assert.Equal(64, (int)D("steps", "maximum"));
        Assert.Equal(0, (int)D("seed", "default"));
        Assert.Equal(0, (int)D("seed", "minimum"));
        Assert.Equal(SongRequest.MaxLyricsCharacters, (int)D("lyrics", "maxLength"));
    }

    [Fact]
    public async Task TheDocumentedLimits_AreEnforcedAtTheirEdges()
    {
        using var h = new ServiceHost();
        var p = RequestProperties();
        async Task<int> Status(string field, double value)
        {
            var body = new Dictionary<string, object> { ["lyrics"] = "[verse]\nHello", ["style"] = "Pop", [field] = field == "steps" || field == "seed" ? (object)(long)value : value };
            var id = (await h.Send(HttpMethod.Post, "/generate", ServiceHost.Body(body))).Status;
            return id;
        }
        foreach (var field in new[] { "max_secs", "cfg_strength", "steps" })
        {
            double lo = double.Parse(p[field]["minimum"], System.Globalization.CultureInfo.InvariantCulture), hi = double.Parse(p[field]["maximum"], System.Globalization.CultureInfo.InvariantCulture);
            Assert.Equal(202, await Status(field, lo));
            Assert.Equal(202, await Status(field, hi));
            double eps = field == "steps" ? 1 : 0.01;
            Assert.Equal(400, await Status(field, lo - eps));
            Assert.Equal(400, await Status(field, hi + eps));
        }
        Assert.Equal(202, await Status("seed", 0));
        Assert.Equal(400, await Status("seed", -1));
        var longLyrics = "[verse]\n" + new string('a', SongRequest.MaxLyricsCharacters);
        Assert.Equal(400, (await h.Send(HttpMethod.Post, "/generate", ServiceHost.Body(new { lyrics = longLyrics, style = "Pop" }))).Status);
    }

    [Fact]
    public async Task TheDocumentation_IsServedWithoutAToken_AndOffline()
    {
        using var h = new ServiceHost(token: "tok");
        using var anonymous = new HttpClient { BaseAddress = h.Client.BaseAddress };

        var yaml = await anonymous.GetStringAsync("/openapi.yaml");
        Assert.Equal(Docs.OpenApi, yaml);
        Assert.StartsWith("# Layout convention", yaml);
        Assert.Contains("openapi: 3.0.3", yaml);

        var page = await anonymous.GetStringAsync("/docs");
        Assert.Contains("swagger-ui", page);
        var assets = Regex.Matches(page + await anonymous.GetStringAsync("/docs/initializer.js"), @"(?:src|href)=""([^""]+)""").Select(m => m.Groups[1].Value).ToList();
        Assert.Contains("/docs/swagger-ui/swagger-ui-bundle.js", assets);
        Assert.Contains("/docs/swagger-ui/swagger-ui.css", assets);
        foreach (var asset in assets)
        {
            Assert.DoesNotMatch(@"^https?://", asset);   // nothing comes from the internet
            using var r = await anonymous.GetAsync(asset);
            Assert.Equal(HttpStatusCode.OK, r.StatusCode);
            Assert.True((await r.Content.ReadAsByteArrayAsync()).Length > 100, asset);
        }
        Assert.DoesNotContain("http://", await anonymous.GetStringAsync("/docs/initializer.js"));
        Assert.Contains("Apache License", await anonymous.GetStringAsync("/docs/swagger-ui/LICENSE"));

        using var health = await anonymous.GetAsync("/health");
        Assert.Equal(HttpStatusCode.Unauthorized, health.StatusCode);   // the API still needs the token
    }

    [Fact]
    public void TheThirdPartyNotes_NameSwaggerUi_ItsVersionAndLicense()
    {
        var root = Path.GetFullPath(Path.Combine(FixtureDirectory.Find().Dir, "..", ".."));
        var readme = File.ReadAllText(Path.Combine(root, "README.md"));
        var notes = readme[readme.IndexOf("## Third-party notes", StringComparison.Ordinal)..];
        Assert.Contains("Swagger UI", notes);
        Assert.Contains("5.32.0", notes);
        Assert.Contains("Apache-2.0", notes);
        Assert.True(File.Exists(Path.Combine(root, "src", "DiffRhythm2.Service", "Docs", "swagger-ui", "LICENSE")), "the license file of Swagger UI is bundled with it");
    }
}
