namespace DiffRhythm2.Service;

/// <summary>
/// The worker process: makes one song and exits, so that everything the song needed, in host and video memory, goes back to the system with the process.
/// Protocol in <see cref="WorkerProtocol"/>: the request is the first line of standard input, which is then watched; if it closes before the song is finished
/// (the service is gone) the song is abandoned.
/// </summary>
internal static class Worker
{
    /// <summary>Makes the song asked for on <paramref name="stdin"/>. <paramref name="abandon"/> runs when standard input closes before the song is finished (the service is gone): the real worker passes a function that ends the process at once, since nobody is waiting for the song.</summary>
    public static int Run(string wavPath, TextReader stdin, TextWriter stdout, TextWriter stderr, ISongMaker maker, Action? abandon = null)
    {
        using var cancellation = new CancellationTokenSource();
        Pipeline.SongRequest request;
        try
        {
            var line = stdin.ReadLine() ?? throw new InvalidOperationException("no request arrived on standard input");
            request = WorkerProtocol.DecodeRequest(line);
        }
        catch (Exception e)
        {
            stderr.WriteLine($"{e.GetType().Name}: {e.Message}");
            return WorkerProtocol.Failed;
        }

        // the service keeps standard input open while it waits; its end means the service is gone
        var watch = Task.Run(() =>
        {
            try { while (stdin.ReadLine() is not null) { } }
            catch (IOException) { }
            cancellation.Cancel();
            abandon?.Invoke();
        });

        var gate = new object();
        var progress = new SyncProgress(p => { lock (gate) { stdout.WriteLine(WorkerProtocol.EncodeProgress(p.Done, p.Total)); stdout.Flush(); } });
        try
        {
            maker.Make(request, wavPath, progress, cancellation.Token);
            return WorkerProtocol.Done;
        }
        catch (OperationCanceledException)
        {
            RemovePartial(wavPath);
            return WorkerProtocol.Cancelled;
        }
        catch (Exception e)
        {
            RemovePartial(wavPath);
            stderr.WriteLine($"{e.GetType().Name}: {e.Message.ReplaceLineEndings(" ")}");
            return WorkerProtocol.Failed;
        }
    }

    /// <summary>Removes the song's file and the temporary names it is written under.</summary>
    public static void RemovePartial(string wavPath)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(wavPath))!;
        var name = Path.GetFileName(wavPath);
        foreach (var path in new[] { wavPath, wavPath + ".tmp" }.Concat(Directory.Exists(dir) ? Directory.EnumerateFiles(dir, name + ".*.tmp") : Array.Empty<string>()))
            try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private sealed class SyncProgress : IProgress<(int Done, int Total)>
    {
        private readonly Action<(int Done, int Total)> handler;
        public SyncProgress(Action<(int Done, int Total)> handler) => this.handler = handler;
        public void Report((int Done, int Total) value) => handler(value);
    }
}
