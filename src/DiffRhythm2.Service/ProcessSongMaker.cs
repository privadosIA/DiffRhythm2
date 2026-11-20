using System.Diagnostics;
using System.Text;

namespace DiffRhythm2.Service;

/// <summary>
/// Makes each song in a new worker process (the service executable started with <c>--worker</c>), so that all the memory the song needed, including the
/// card's, goes back to the system when the process ends, and so that a song can be stopped at once by ending the process. See <see cref="Worker"/>.
/// </summary>
internal sealed class ProcessSongMaker : ISongMaker
{
    private const int StderrLines = 20;

    /// <summary>Builds the command that starts a worker for a WAV path: the program and its arguments.</summary>
    public delegate (string Program, IReadOnlyList<string> Arguments) CommandFactory(string wavPath);

    private readonly CommandFactory command;

    public ProcessSongMaker(CommandFactory? command = null) => this.command = command ?? CurrentExecutable;

    /// <summary>This executable as a worker: <c>dotnet service.dll --worker path</c> when run through the dotnet host, the program itself otherwise.</summary>
    public static (string Program, IReadOnlyList<string> Arguments) CurrentExecutable(string wavPath)
    {
        var path = Environment.ProcessPath ?? throw new InvalidOperationException("the path of this program is not known");
        bool host = Path.GetFileNameWithoutExtension(path).Equals("dotnet", StringComparison.OrdinalIgnoreCase);
        return host
            ? (path, new[] { typeof(ProcessSongMaker).Assembly.Location, WorkerProtocol.Argument, wavPath })
            : (path, new[] { WorkerProtocol.Argument, wavPath });
    }

    public void Make(Pipeline.SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        var (program, arguments) = command(wavPath);
        var utf8 = new UTF8Encoding(false);
        var info = new ProcessStartInfo(program)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            StandardInputEncoding = utf8,
            StandardOutputEncoding = utf8,
            StandardErrorEncoding = utf8,
        };
        foreach (var a in arguments) info.ArgumentList.Add(a);

        using var process = Process.Start(info) ?? throw new JobFailure("the song process could not be started");
        using var killOnCancel = cancellation.Register(() => Kill(process));
        try
        {
            // the request is one line; standard input then stays open until the process ends, which is how the worker knows the service is still there
            process.StandardInput.WriteLine(WorkerProtocol.EncodeRequest(request));
            process.StandardInput.Flush();
        }
        catch (IOException) { /* the worker already ended; its exit code says why */ }

        // The pumps are deliberately not tied to the job's token: they end when the process closes its streams, and cancelling kills the process. A token that
        // cancels a pump before it starts would make WaitAll throw and skip the clean-up below.
        var tail = new Queue<string>();
        var stdout = Task.Run(() =>
        {
            string? line;
            while ((line = process.StandardOutput.ReadLine()) is not null)
                if (WorkerProtocol.TryDecodeProgress(line, out var p)) progress.Report(p);
        }, CancellationToken.None);
        var stderr = Task.Run(() =>
        {
            string? line;
            while ((line = process.StandardError.ReadLine()) is not null)
            {
                if (line.Length == 0) continue;
                lock (tail) { tail.Enqueue(line); if (tail.Count > StderrLines) tail.Dequeue(); }
            }
        }, CancellationToken.None);

        process.WaitForExit();
        Task.WaitAll(stdout, stderr);

        if (cancellation.IsCancellationRequested)
        {
            Remove(wavPath);
            throw new OperationCanceledException(cancellation);
        }
        string? last;
        lock (tail) last = tail.Count > 0 ? tail.Last() : null;
        int code = process.ExitCode;
        if (code == WorkerProtocol.Done)
        {
            if (!File.Exists(wavPath)) throw new JobFailure("the song process ended without writing the song");
            return;
        }
        Remove(wavPath);
        if (code == WorkerProtocol.Failed && last is not null) throw new JobFailure(last);
        throw new JobFailure($"the song process ended unexpectedly (exit code {code})" + (last is null ? "" : ": " + last));
    }

    private static void Kill(Process process)
    {
        try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } catch (System.ComponentModel.Win32Exception) { }
    }

    private static void Remove(string wavPath)
    {
        foreach (var path in new[] { wavPath, wavPath + ".tmp" })
            try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}
