using System.Text.Json;
using DiffRhythm2.Pipeline;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>Request validation against what the service's own <c>Engine.validate</c> returns or raises for the same bodies, and the model directory.</summary>
public class SongRequestTests : IDisposable
{
    private readonly ITestOutputHelper log;
    private readonly string tmp = Directory.CreateTempSubdirectory("dr2-song-").FullName;

    public SongRequestTests(ITestOutputHelper log) => this.log = log;

    public void Dispose() => Directory.Delete(tmp, recursive: true);

    private static SongRequest FromBody(JsonElement body)
    {
        string? lyrics = body.TryGetProperty("lyrics", out var l) ? l.GetString() : null;
        string style = "";
        string[]? words = null;
        if (body.TryGetProperty("style", out var s))
        {
            if (s.ValueKind == JsonValueKind.Array) words = s.EnumerateArray().Select(e => e.GetString()!).ToArray();
            else style = s.GetString()!;
        }
        double seconds = body.TryGetProperty("max_secs", out var m) ? m.GetDouble() : SongRequest.DefaultSeconds;
        double cfg = body.TryGetProperty("cfg_strength", out var c) ? c.GetDouble() : SongRequest.DefaultCfgStrength;
        int steps = body.TryGetProperty("steps", out var st) ? st.GetInt32() : SongRequest.DefaultSteps;
        long? seed = body.TryGetProperty("seed", out var sd) && sd.TryGetInt64(out var sv) ? sv : null;
        return words is null ? new SongRequest(lyrics!, style, seconds, cfg, steps, seed) : SongRequest.FromWords(lyrics!, words, seconds, cfg, steps, seed);
    }

    [Fact]
    public void EveryCaseOfTheService_GivesTheSameOutcome()
    {
        int ok = 0, refused = 0, skipped = 0;
        foreach (var c in SongReference.Validation.EnumerateArray())
        {
            var name = c.GetProperty("name").GetString()!;
            var body = c.GetProperty("body");
            if (!c.GetProperty("typed").GetBoolean()) { skipped++; continue; }

            // a seed that does not fit 63 bits cannot be a typed value: it is refused when it is read
            if (body.TryGetProperty("seed", out var seedProp) && !seedProp.TryGetInt64(out _))
            {
                Assert.False(SongRequest.TryParseSeed(seedProp.GetRawText(), out _), name);
                Assert.Equal(SongRequest.SeedMessage, c.GetProperty("error").GetString());
                refused++;
                continue;
            }

            var request = FromBody(body);
            if (c.TryGetProperty("error", out var error))
            {
                var e = Assert.Throws<SongRequestException>(request.Validate);
                Assert.True(error.GetString() == e.Message, $"{name}: expected \"{error.GetString()}\", got \"{e.Message}\" ({e.Field})");
                refused++;
            }
            else
            {
                var expected = c.GetProperty("ok");
                var v = request.Validate();
                Assert.True(expected.GetProperty("lyrics").GetString() == v.Lyrics, $"{name}: lyrics differ");
                Assert.True(expected.GetProperty("style").GetString() == v.Style, $"{name}: style \"{expected.GetProperty("style").GetString()}\" vs \"{v.Style}\"");
                Assert.Equal(expected.GetProperty("max_secs").GetDouble(), v.Seconds);
                Assert.Equal(expected.GetProperty("cfg_strength").GetDouble(), v.CfgStrength);
                Assert.Equal(expected.GetProperty("steps").GetInt32(), v.Steps);
                if (body.TryGetProperty("seed", out _)) Assert.Equal(expected.GetProperty("seed").GetInt64(), v.Seed);
                else Assert.Null(v.Seed);        // the library picks a seed; the service's default of 0 belongs to its HTTP layer
                Assert.Equal((int)(expected.GetProperty("max_secs").GetDouble() * 5), v.Frames);
                ok++;
            }
        }
        log.WriteLine($"{ok} accepted and {refused} refused identically, {skipped} cases of the wrong JSON type not representable");
        Assert.True(ok >= 25 && refused >= 15);
    }

    [Fact]
    public void ErrorsNameTheField()
    {
        var good = new SongRequest("[verse]\nla la", "Pop");
        Assert.Equal("lyrics", Assert.Throws<SongRequestException>(() => (good with { Lyrics = "" }).Validate()).Field);
        Assert.Equal("lyrics", Assert.Throws<SongRequestException>(() => (good with { Lyrics = "[verse]\n[chorus]" }).Validate()).Field);
        Assert.Equal("style", Assert.Throws<SongRequestException>(() => (good with { Style = " , " }).Validate()).Field);
        Assert.Equal("max_secs", Assert.Throws<SongRequestException>(() => (good with { Seconds = 9 }).Validate()).Field);
        Assert.Equal("cfg_strength", Assert.Throws<SongRequestException>(() => (good with { CfgStrength = 11 }).Validate()).Field);
        Assert.Equal("steps", Assert.Throws<SongRequestException>(() => (good with { Steps = 65 }).Validate()).Field);
        Assert.Equal("seed", Assert.Throws<SongRequestException>(() => (good with { Seed = -1 }).Validate()).Field);
    }

    [Fact]
    public void StyleNormalization_AndDefaults()
    {
        Assert.Equal("Pop, Piano, Female vocals", SongRequest.NormalizeStyle("  Pop ,Piano,  Female vocals ,"));
        var v = new SongRequest("hello", "Pop").Validate();
        Assert.Equal((210.0, 2.0, 16, 1050), (v.Seconds, v.CfgStrength, v.Steps, v.Frames));
        Assert.Null(v.Seed);
        Assert.True(SongRequest.TryParseSeed("9223372036854775807", out var max) && max == long.MaxValue);
        Assert.False(SongRequest.TryParseSeed("-1", out _));
        Assert.False(SongRequest.TryParseSeed("1.5", out _));
        Assert.False(SongRequest.TryParseSeed("abc", out _));
    }

    [Fact]
    public void ModelDirectory_ListsEveryMissingFileAtOnce()
    {
        File.WriteAllText(Path.Combine(tmp, "config.json"), "{}");
        File.WriteAllText(Path.Combine(tmp, "model.safetensors"), "x");
        File.WriteAllText(Path.Combine(tmp, "decoder.json"), "{}");
        File.WriteAllText(Path.Combine(tmp, "mulan_text_tokenizer.json"), "{}");
        var dir = new ModelDirectory(tmp);
        var e = Assert.Throws<ModelDirectoryException>(dir.EnsureComplete);
        Assert.Equal(new[] { "decoder.safetensors", "mulan_text.safetensors" }, e.Missing);
        Assert.Contains("decoder.safetensors", e.Message);
        Assert.Contains("mulan_text.safetensors", e.Message);
        Assert.Equal(6, dir.Files.Count);
        Assert.Equal(1L, dir.Files.Single(f => f.Name == "model.safetensors").Length);

        File.WriteAllText(Path.Combine(tmp, "decoder.safetensors"), "x");
        File.WriteAllText(Path.Combine(tmp, "mulan_text.safetensors"), "x");
        dir.EnsureComplete();
    }

    [Fact]
    public void ModelDirectory_ThatDoesNotExist_ListsAllSixFiles()
    {
        var e = Assert.Throws<ModelDirectoryException>(new ModelDirectory(Path.Combine(tmp, "nope")).EnsureComplete);
        Assert.Equal(6, e.Missing.Count);
    }
}
