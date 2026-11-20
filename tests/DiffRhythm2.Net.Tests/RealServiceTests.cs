using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;
using DiffRhythm2.Text;

namespace DiffRhythm2.Tests;

/// <summary>A real song through the service on CPU (needs the six model files and the phonemizer library; skipped otherwise).</summary>
public class RealServiceTests
{
    [RequiresSongModels]
    public async Task ASongThroughTheService_IsTheSongThePipelineMakes()
    {
        if (!EspeakEngine.TryGetShared(out _, out var reason)) { Console.WriteLine("skipped: " + reason); return; }
        var model = Checkpoint.Dir!;
        var settings = new ServiceSettings(model, "", 1, "/unused", "test", "cpu");
        using var h = new ServiceHost(real: new SongMaker(settings), model: model);
        const string lyrics = "[verse]\nHello world tonight\nWe sing along";
        const string style = "Pop, Piano";

        var id = await h.Post(new { lyrics, style, max_secs = 10, steps = 4, seed = 1234 });
        var done = await h.WaitFor(id, "done", seconds: 600);
        Assert.True(done.GetProperty("progress").GetProperty("total").GetInt32() > 0);
        Assert.Equal("", done.GetProperty("error").GetString());
        var served = await h.Client.GetByteArrayAsync($"/jobs/{id}/audio");

        var expected = new SongGenerator(new SongOptions(model)).Generate(new SongRequest(lyrics, style, 10, 2.0, 4, 1234)).ToWavBytes();
        Assert.Equal(expected.Length, served.Length);
        Assert.True(expected.AsSpan().SequenceEqual(served), "the WAV served differs from the one the pipeline writes for the same request");
        Assert.Equal(1.0, (await h.Get("/health")).Json.GetProperty("ready").GetBoolean() ? 1.0 : 0.0);
    }

    [RequiresSongModels]
    public async Task ASongMadeInAWorkerProcess_IsTheSameSongByteForByte()
    {
        if (!EspeakEngine.TryGetShared(out _, out var reason)) { Console.WriteLine("skipped: " + reason); return; }
        var model = Checkpoint.Dir!;
        using var h = new ServiceHost(real: new ProcessSongMaker(), model: model);   // the worker is this build of the service, run as `dotnet diffrhythm2-service.dll --worker`
        const string lyrics = "[verse]\nHello world tonight\nWe sing along";
        const string style = "Pop, Piano";

        var id = await h.Post(new { lyrics, style, max_secs = 10, steps = 4, seed = 1234 });
        var done = await h.WaitFor(id, "done", seconds: 600);
        Assert.True(done.GetProperty("progress").GetProperty("total").GetInt32() > 0, "no progress came through the worker's output");
        var served = await h.Client.GetByteArrayAsync($"/jobs/{id}/audio");

        var expected = new SongGenerator(new SongOptions(model)).Generate(new SongRequest(lyrics, style, 10, 2.0, 4, 1234)).ToWavBytes();
        Assert.Equal(expected.Length, served.Length);
        Assert.True(expected.AsSpan().SequenceEqual(served), "the WAV made by the worker differs from the one the pipeline writes");
    }
}
