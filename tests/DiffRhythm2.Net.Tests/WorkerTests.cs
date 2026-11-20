using System.IO.Pipes;
using System.Text;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Service;

namespace DiffRhythm2.Tests;

/// <summary>The worker side of the song process: the protocol lines and the worker's behavior, in-process with piped streams and a fake maker.</summary>
public class WorkerTests
{
    private static readonly SongRequest Sample = new("[verse]\nHola, ñandú \"quoted\"\nline two", "Pop, Piano", 30.5, 3.25, 12, 987654321012L);

    [Fact]
    public void TheRequestLine_RoundTrips_WithUnicodeQuotesAndNewlines()
    {
        var line = WorkerProtocol.EncodeRequest(Sample);
        Assert.DoesNotContain('\n', line);   // one line, whatever the lyrics hold
        Assert.Equal(Sample, WorkerProtocol.DecodeRequest(line));
        Assert.Equal(Sample with { Seed = null }, WorkerProtocol.DecodeRequest(WorkerProtocol.EncodeRequest(Sample with { Seed = null })));
    }

    [Fact]
    public void ProgressLines_AreReadBack_AndOtherLinesIgnored()
    {
        Assert.True(WorkerProtocol.TryDecodeProgress(WorkerProtocol.EncodeProgress(3, 8), out var p));
        Assert.Equal((3, 8), p);
        Assert.False(WorkerProtocol.TryDecodeProgress("some native library says hello", out _));
        Assert.False(WorkerProtocol.TryDecodeProgress("{not json", out _));
        Assert.False(WorkerProtocol.TryDecodeProgress("", out _));
    }

    private sealed class FakeSong : ISongMaker
    {
        public SongRequest? Got;
        public Action<string, IProgress<(int, int)>, CancellationToken> Body = (_, _, _) => { };
        public void Make(SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation)
        {
            Got = request;
            Body(wavPath, progress, cancellation);
        }
    }

    private static (int Code, string Out, string Err) RunWorker(string wav, string stdin, FakeSong maker)
    {
        var o = new StringWriter();
        var e = new StringWriter();
        int code = Worker.Run(wav, new StringReader(stdin), o, e, maker);
        return (code, o.ToString(), e.ToString());
    }

    [Fact]
    public void ASong_ReadsTheRequest_ReportsEachBlock_AndExits0()
    {
        var wav = Path.Combine(Path.GetTempPath(), "dr2-w-" + Guid.NewGuid().ToString("N") + ".wav");
        var maker = new FakeSong { Body = (path, progress, _) => { for (int i = 1; i <= 3; i++) progress.Report((i, 3)); File.WriteAllText(path, "RIFF"); } };
        try
        {
            var r = RunWorker(wav, WorkerProtocol.EncodeRequest(Sample) + "\n", maker);
            Assert.Equal(0, r.Code);
            Assert.Equal(Sample, maker.Got);
            Assert.Equal(new[] { WorkerProtocol.EncodeProgress(1, 3), WorkerProtocol.EncodeProgress(2, 3), WorkerProtocol.EncodeProgress(3, 3) }, r.Out.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => l.TrimEnd('\r')));
            Assert.Equal("", r.Err);
            Assert.True(File.Exists(wav));
        }
        finally { File.Delete(wav); }
    }

    [Fact]
    public void ASongThatFails_ExitsWith1_CauseOnTheLastErrorLine_AndLeavesNoFile()
    {
        var wav = Path.Combine(Path.GetTempPath(), "dr2-w-" + Guid.NewGuid().ToString("N") + ".wav");
        var maker = new FakeSong { Body = (path, _, _) => { File.WriteAllText(path, "partial"); throw new DeviceUnavailableException("cuda was requested but this machine has 0 CUDA device(s)\nsecond line"); } };
        var r = RunWorker(wav, WorkerProtocol.EncodeRequest(Sample) + "\n", maker);
        Assert.Equal(1, r.Code);
        Assert.Equal("DeviceUnavailableException: cuda was requested but this machine has 0 CUDA device(s) second line", r.Err.TrimEnd());
        Assert.False(File.Exists(wav));
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json\n")]
    public void ABadRequest_ExitsWith1_WithoutMakingASong(string stdin)
    {
        var maker = new FakeSong();
        var r = RunWorker("/unused.wav", stdin, maker);
        Assert.Equal(1, r.Code);
        Assert.Null(maker.Got);
        Assert.NotEqual("", r.Err);
    }

    [Fact]
    public async Task StandardInputClosedEarly_CancelsTheSong_ExitsWith3_AndRemovesThePartialFile()
    {
        var wav = Path.Combine(Path.GetTempPath(), "dr2-w-" + Guid.NewGuid().ToString("N") + ".wav");
        using var server = new AnonymousPipeServerStream(PipeDirection.Out);
        using var client = new AnonymousPipeClientStream(PipeDirection.In, server.ClientSafePipeHandle);
        var maker = new FakeSong
        {
            Body = (path, _, ct) =>
            {
                File.WriteAllText(path, "partial");
                while (true) { ct.ThrowIfCancellationRequested(); Thread.Sleep(10); }   // a song that never ends by itself
            },
        };
        var o = new StringWriter();
        var e = new StringWriter();
        var run = Task.Run(() => Worker.Run(wav, new StreamReader(client, new UTF8Encoding(false)), o, e, maker));
        var writer = new StreamWriter(server, new UTF8Encoding(false)) { AutoFlush = true };
        writer.WriteLine(WorkerProtocol.EncodeRequest(Sample));
        await Task.Delay(300);
        Assert.False(run.IsCompleted, "the worker must wait while its input stays open");
        writer.Dispose();   // the service is gone
        Assert.Equal(3, await run.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.False(File.Exists(wav));
    }

    [Fact]
    public async Task WhenTheServiceIsGone_TheAbandonActionRunsAtOnce_EvenIfTheSongIgnoresCancellation()
    {
        using var server = new AnonymousPipeServerStream(PipeDirection.Out);
        using var client = new AnonymousPipeClientStream(PipeDirection.In, server.ClientSafePipeHandle);
        var abandoned = new TaskCompletionSource();
        var release = new ManualResetEventSlim();
        var maker = new FakeSong { Body = (_, _, _) => release.Wait(TimeSpan.FromSeconds(30)) };   // a song deaf to its token
        var run = Task.Run(() => Worker.Run("/unused.wav", new StreamReader(client, new UTF8Encoding(false)), new StringWriter(), new StringWriter(), maker, () => abandoned.TrySetResult()));
        var writer = new StreamWriter(server, new UTF8Encoding(false)) { AutoFlush = true };
        writer.WriteLine(WorkerProtocol.EncodeRequest(Sample));
        await Task.Delay(200);
        writer.Dispose();
        await abandoned.Task.WaitAsync(TimeSpan.FromSeconds(5));
        release.Set();
        await run.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(abandoned.Task.IsCompletedSuccessfully);
        Assert.True(run.IsCompleted);
    }
}
