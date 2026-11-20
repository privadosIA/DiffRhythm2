using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;

namespace DiffRhythm2.Tests;

/// <summary>The queue and the worker with a fake song maker: order, states, progress, failures, forgetting, cancelling and retention.</summary>
public class JobEngineTests
{
    private static SongRequest Request(string lyrics = "[verse]\nHello") => new(lyrics, "Pop", Seed: 0);

    private static async Task<JobView> WaitFor(JobEngine e, string id, string state)
    {
        for (int i = 0; i < 1000; i++)
        {
            var v = e.View(id);
            if (v?.State == state) return v;
            await Task.Delay(10);
        }
        throw new TimeoutException($"{id} never reached {state}");
    }

    [Fact]
    public async Task OneSongAtATime_InTheOrderAccepted()
    {
        var maker = new FakeMaker();
        var dir = Directory.CreateTempSubdirectory("dr2-eng-").FullName;
        using var engine = new JobEngine(maker, dir);
        var a = engine.Submit(Request("[verse]\nSLOW one"));
        var b = engine.Submit(Request("[verse]\nSLOW two"));
        await WaitFor(engine, a.Id, "running");
        Assert.Equal("queued", engine.View(b.Id)!.State);
        Assert.Equal(1, engine.Queued);
        Assert.True(engine.Busy);

        maker.Finish.Release();
        await WaitFor(engine, b.Id, "running");
        Assert.Equal("done", engine.View(a.Id)!.State);
        maker.Finish.Release();
        await WaitFor(engine, b.Id, "done");
        Assert.Equal(1, maker.MaxConcurrent);
        Assert.Contains("one", maker.Made[0].Lyrics);
        Assert.Contains("two", maker.Made[1].Lyrics);
        Assert.False(engine.Busy);
    }

    [Fact]
    public async Task Progress_NeverDecreases_AndASeconds_Field_Grows()
    {
        var maker = new FakeMaker();
        using var engine = new JobEngine(maker, Directory.CreateTempSubdirectory("dr2-eng-").FullName);
        var j = engine.Submit(Request("[verse]\nSLOW song"));
        var running = await WaitFor(engine, j.Id, "running");
        Assert.Equal(3, running.Total);
        Assert.Equal(0, engine.View(engine.Submit(Request("[verse]\nqueued")).Id)!.Total);   // a queued job reports 0 of 0
        int last = running.Done;
        maker.Finish.Release();
        var done = await WaitFor(engine, j.Id, "done");
        Assert.True(done.Done >= last);
        Assert.Equal(3, done.Done);
        Assert.True(done.Seconds >= 0);
    }

    [Fact]
    public async Task AFailingSong_StaysWithItsJob_AndTheServiceGoesOn()
    {
        var maker = new FakeMaker();
        using var engine = new JobEngine(maker, Directory.CreateTempSubdirectory("dr2-eng-").FullName);
        var bad = engine.Submit(Request("[verse]\nFAIL now"));
        var good = engine.Submit(Request());
        var failed = await WaitFor(engine, bad.Id, "failed");
        Assert.Contains("this song cannot be made", failed.Error);
        Assert.StartsWith("InvalidOperationException", failed.Error);
        await WaitFor(engine, good.Id, "done");
        Assert.Null(engine.Audio(bad.Id)!.Value.Path);
    }

    private sealed class FailingMaker : ISongMaker
    {
        public void Make(SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation) => throw new JobFailure("DeviceUnavailableException: no cuda device");
    }

    [Fact]
    public async Task AJobFailure_IsTheJobsErrorAsItIs()
    {
        using var engine = new JobEngine(new FailingMaker(), Directory.CreateTempSubdirectory("dr2-eng-").FullName);
        var j = engine.Submit(Request());
        Assert.Equal("DeviceUnavailableException: no cuda device", (await WaitFor(engine, j.Id, "failed")).Error);
    }

    [Fact]
    public async Task ForgettingAQueuedJob_StopsItFromEverRunning()
    {
        var maker = new FakeMaker();
        using var engine = new JobEngine(maker, Directory.CreateTempSubdirectory("dr2-eng-").FullName);
        var a = engine.Submit(Request("[verse]\nSLOW a"));
        var b = engine.Submit(Request("[verse]\nb"));
        await WaitFor(engine, a.Id, "running");
        Assert.True(engine.Forget(b.Id));
        Assert.Null(engine.View(b.Id));
        maker.Finish.Release();
        await WaitFor(engine, a.Id, "done");
        await Task.Delay(100);
        Assert.Single(maker.Made);
    }

    [Fact]
    public async Task ForgettingTheRunningJob_CancelsIt_AndTheNextOneStartsAtOnce()
    {
        var maker = new FakeMaker();
        var dir = Directory.CreateTempSubdirectory("dr2-eng-").FullName;
        using var engine = new JobEngine(maker, dir);
        var a = engine.Submit(Request("[verse]\nSLOW a"));
        var b = engine.Submit(Request("[verse]\nb"));
        await WaitFor(engine, a.Id, "running");
        var clock = System.Diagnostics.Stopwatch.StartNew();
        Assert.True(engine.Forget(a.Id));
        await WaitFor(engine, b.Id, "done");
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(5), "the next song waited for the cancelled one");
        Assert.Null(engine.View(a.Id));
        Assert.False(File.Exists(Path.Combine(dir, a.Id + ".wav")));
        Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
    }

    [Fact]
    public async Task ForgettingADoneJob_DeletesItsAudio()
    {
        using var engine = new JobEngine(new FakeMaker(), Directory.CreateTempSubdirectory("dr2-eng-").FullName);
        var a = engine.Submit(Request());
        await WaitFor(engine, a.Id, "done");
        var path = engine.Audio(a.Id)!.Value.Path!;
        Assert.True(File.Exists(path));
        engine.Forget(a.Id);
        Assert.False(File.Exists(path));
        Assert.False(engine.Forget(a.Id));
    }

    [Fact]
    public async Task OnlyTheLast20FinishedJobs_AreKept()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-eng-").FullName;
        using var engine = new JobEngine(new FakeMaker(), dir);
        var jobs = Enumerable.Range(0, 21).Select(_ => engine.Submit(Request())).ToList();
        await WaitFor(engine, jobs[^1].Id, "done");
        Assert.Null(engine.View(jobs[0].Id));
        Assert.False(File.Exists(Path.Combine(dir, jobs[0].Id + ".wav")));
        Assert.All(jobs.Skip(1), j => Assert.NotNull(engine.View(j.Id)));
        Assert.Equal(20, Directory.GetFiles(dir, "*.wav").Length);
    }

    [Fact]
    public void StaleFilesOfAnEarlierRun_AreRemovedAtStart()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-eng-").FullName;
        File.WriteAllText(Path.Combine(dir, "old.wav"), "x");
        File.WriteAllText(Path.Combine(dir, "old2.wav.tmp"), "x");
        File.WriteAllText(Path.Combine(dir, "keep.txt"), "x");
        using var engine = new JobEngine(new FakeMaker(), dir);
        Assert.Equal(new[] { "keep.txt" }, Directory.GetFiles(dir).Select(Path.GetFileName));
    }

    [Fact]
    public async Task ADoneJob_NeverShowsAPartialFile()
    {
        var dir = Directory.CreateTempSubdirectory("dr2-eng-").FullName;
        var maker = new FakeMaker();
        using var engine = new JobEngine(maker, dir);
        var j = engine.Submit(Request());
        await WaitFor(engine, j.Id, "done");
        Assert.Equal(FakeMaker.Wav().Length, new FileInfo(engine.Audio(j.Id)!.Value.Path!).Length);
        Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
    }
}
