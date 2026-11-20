using System.Diagnostics;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;

namespace DiffRhythm2.Tests;

/// <summary>The supervisor against a scripted child that speaks the protocol (bash), so no model is involved. Skipped where bash is not available.</summary>
public class ProcessSongMakerTests : IDisposable
{
    private readonly string dir = Directory.CreateTempSubdirectory("dr2-psm-").FullName;
    private readonly SongRequest request = new("[verse]\nHello", "Pop", Seed: 1);

    public void Dispose() { try { Directory.Delete(dir, true); } catch (IOException) { } }

    private static bool HasBash => OperatingSystem.IsLinux() || OperatingSystem.IsMacOS();

    private ProcessSongMaker Maker(string script) =>
        new(wav => ("bash", new[] { "-c", script, "bash", wav }));

    private sealed class Collect : IProgress<(int Done, int Total)>
    {
        public readonly List<(int, int)> Items = new();
        public void Report((int Done, int Total) value) { lock (Items) Items.Add(value); }
    }

    private string Wav => Path.Combine(dir, "song.wav.tmp");

    [Fact]
    public void ASong_ReportsProgress_IgnoresStrayLines_AndLeavesTheFile()
    {
        if (!HasBash) return;
        var progress = new Collect();
        Maker("""
            read -r request
            echo 'a native library says hello'
            echo '{"done":1,"total":3}'
            echo '{"done":2,"total":3}'
            echo '{"done":3,"total":3}'
            printf RIFF > "$1"
            """).Make(request, Wav, progress, default);
        Assert.Equal(new[] { (1, 3), (2, 3), (3, 3) }, progress.Items);
        Assert.Equal("RIFF", File.ReadAllText(Wav));
    }

    [Fact]
    public void TheRequestArrivesOnStandardInput_AsOneLine()
    {
        if (!HasBash) return;
        var seen = Path.Combine(dir, "seen.txt");
        Maker($"read -r request; printf '%s' \"$request\" > '{seen}'; printf RIFF > \"$1\"").Make(request, Wav, new Collect(), default);
        Assert.Equal(WorkerProtocol.EncodeRequest(request), File.ReadAllText(seen));
    }

    [Fact]
    public void AFailure_BecomesTheJobsErrorVerbatim_AndLeavesNoFile()
    {
        if (!HasBash) return;
        var e = Assert.Throws<JobFailure>(() => Maker("""
            read -r request
            printf partial > "$1"
            echo 'some noise' >&2
            echo 'DeviceUnavailableException: no cuda device' >&2
            exit 1
            """).Make(request, Wav, new Collect(), default));
        Assert.Equal("DeviceUnavailableException: no cuda device", e.Message);
        Assert.False(File.Exists(Wav));
    }

    [Fact]
    public void AWorkerKilledBySignal_IsReportedAsEndedUnexpectedly_WithItsExitCode()
    {
        if (!HasBash) return;
        var e = Assert.Throws<JobFailure>(() => Maker("read -r request; kill -9 $$").Make(request, Wav, new Collect(), default));
        Assert.StartsWith("the song process ended unexpectedly (exit code 137)", e.Message);
    }

    [Fact]
    public void ExitingWith0WithoutTheFile_IsAFailure()
    {
        if (!HasBash) return;
        var e = Assert.Throws<JobFailure>(() => Maker("read -r request; exit 0").Make(request, Wav, new Collect(), default));
        Assert.Contains("without writing the song", e.Message);
    }

    [Fact]
    public void ACommandThatDoesNotExist_IsAFailureOfTheJob_NotACrash()
    {
        var maker = new ProcessSongMaker(wav => ("/nonexistent/program", new[] { wav }));
        Assert.ThrowsAny<Exception>(() => maker.Make(request, Wav, new Collect(), default));
    }

    [Fact]
    public void LotsOfErrorOutput_DoesNotBlockTheWorker()
    {
        if (!HasBash) return;
        Maker("read -r request; for i in $(seq 1 200000); do echo \"line $i of noise\" >&2; done; printf RIFF > \"$1\"")
            .Make(request, Wav, new Collect(), default);
        Assert.True(File.Exists(Wav));
    }

    [Fact]
    public void Cancelling_EndsTheWorkerAtOnce_RemovesItsFile_AndThrowsCancelled()
    {
        if (!HasBash) return;
        var pidFile = Path.Combine(dir, "pid");
        using var cts = new CancellationTokenSource();
        var maker = Maker($"read -r request; echo $$ > '{pidFile}'; printf partial > \"$1\"; sleep 60");
        var task = Task.Run(() => maker.Make(request, Wav, new Collect(), cts.Token));
        for (int i = 0; i < 200 && !File.Exists(pidFile); i++) Thread.Sleep(25);
        Thread.Sleep(100);
        var clock = Stopwatch.StartNew();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => task.GetAwaiter().GetResult());
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(2), $"cancelling took {clock.Elapsed.TotalSeconds:F1} s");
        var pid = int.Parse(File.ReadAllText(pidFile).Trim());
        Assert.False(Directory.Exists($"/proc/{pid}") && File.ReadAllText($"/proc/{pid}/stat").Split(' ')[2] != "Z", "the worker is still running");
        Assert.False(File.Exists(Wav));
    }

    [Fact]
    public void AlreadyCancelled_StartsNothing()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => new ProcessSongMaker(_ => throw new InvalidOperationException("must not start")).Make(request, Wav, new Collect(), cts.Token));
    }

    [Fact]
    public void TheCommandOfTheService_IsItsOwnExecutable_AsAWorker()
    {
        var (program, args) = ProcessSongMaker.CurrentExecutable("/x/song.wav");
        Assert.Equal("/x/song.wav", args[^1]);
        Assert.Equal("--worker", args[^2]);
        if (Path.GetFileNameWithoutExtension(program) == "dotnet") Assert.EndsWith("diffrhythm2-service.dll", args[0]);
    }
}
