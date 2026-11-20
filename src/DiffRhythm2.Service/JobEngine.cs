using System.Collections.Concurrent;
using DiffRhythm2.Pipeline;

namespace DiffRhythm2.Service;

/// <summary>Makes a song and writes it to a WAV file. The real one wraps the pipeline; tests use a fake.</summary>
internal interface ISongMaker
{
    /// <summary>Makes the song for <paramref name="request"/> and writes the WAV to <paramref name="wavPath"/>. Reports blocks sampled as (done, total). Throws <see cref="OperationCanceledException"/> when cancelled.</summary>
    void Make(SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation);
}

/// <summary>A song that could not be made, with the cause already worded for the client: the job's <c>error</c> is its message, as it is.</summary>
internal sealed class JobFailure : Exception
{
    public JobFailure(string message) : base(message) { }
}

internal static class JobStates
{
    public const string Queued = "queued", Running = "running", Done = "done", Failed = "failed";
}

/// <summary>A requested song and what has become of it.</summary>
internal sealed class Job
{
    public string Id { get; } = Guid.NewGuid().ToString("N")[..12];
    public SongRequest Request { get; }
    public CancellationTokenSource Cancellation { get; } = new();
    public string State { get; set; } = JobStates.Queued;
    public int Done { get; set; }
    public int Total { get; set; }
    public string Error { get; set; } = "";
    public DateTime? Started { get; set; }
    public DateTime? Finished { get; set; }
    public string WavPath { get; }

    public Job(SongRequest request, string directory)
    {
        Request = request;
        WavPath = Path.Combine(directory, Id + ".wav");
    }
}

/// <summary>What <c>GET /jobs/{id}</c> shows of a job.</summary>
internal sealed record JobView(string Id, string State, int Done, int Total, string Error, double Seconds);

/// <summary>
/// The queue and the worker: one song at a time, in the order accepted. Jobs are kept in memory (the last <see cref="KeepFinished"/> finished ones with
/// their audio); forgetting a queued job stops it from ever running and forgetting a running one cancels it.
/// </summary>
internal sealed class JobEngine : IDisposable
{
    public const int KeepFinished = 20;

    private readonly ISongMaker maker;
    private readonly string directory;
    private readonly object gate = new();
    private readonly Dictionary<string, Job> jobs = new();
    private readonly List<string> order = new();
    private readonly BlockingCollection<Job> queue = new();
    private readonly Thread worker;
    private Job? current;

    public JobEngine(ISongMaker maker, string jobsDirectory)
    {
        this.maker = maker;
        directory = jobsDirectory;
        Directory.CreateDirectory(directory);
        foreach (var stale in Directory.EnumerateFiles(directory).Where(f => f.EndsWith(".wav", StringComparison.Ordinal) || f.EndsWith(".tmp", StringComparison.Ordinal)))
            TryDelete(stale);
        worker = new Thread(Work) { IsBackground = true, Name = "song-worker" };
        worker.Start();
    }

    public Job Submit(SongRequest request)
    {
        var job = new Job(request, directory);
        lock (gate)
        {
            jobs[job.Id] = job;
            order.Add(job.Id);
        }
        queue.Add(job);
        return job;
    }

    public JobView? View(string id)
    {
        lock (gate)
        {
            if (!jobs.TryGetValue(id, out var j)) return null;
            double seconds = j.Started is null ? 0 : Math.Round(((j.Finished ?? DateTime.UtcNow) - j.Started.Value).TotalSeconds, 1);
            return new JobView(j.Id, j.State, j.Done, j.Total, j.Error, seconds);
        }
    }

    /// <summary>The path of the audio of a job that is done, or null.</summary>
    public (string State, string? Path)? Audio(string id)
    {
        lock (gate) return jobs.TryGetValue(id, out var j) ? (j.State, j.State == JobStates.Done ? j.WavPath : null) : null;
    }

    public bool Forget(string id)
    {
        Job? job;
        lock (gate)
        {
            if (!jobs.Remove(id, out job)) return false;
            order.Remove(id);
        }
        if (job.State is JobStates.Queued or JobStates.Running) job.Cancellation.Cancel();
        if (job.State is not JobStates.Running) { TryDelete(job.WavPath); TryDelete(job.WavPath + ".tmp"); }   // a running job's worker cleans up when the song stops
        return true;
    }

    public int Queued { get { lock (gate) return jobs.Values.Count(j => j.State == JobStates.Queued); } }

    public bool Busy { get { lock (gate) return current is not null; } }

    private void Work()
    {
        foreach (var job in queue.GetConsumingEnumerable())
        {
            lock (gate)
            {
                if (!jobs.ContainsKey(job.Id)) continue;   // forgotten while it waited
                job.State = JobStates.Running;
                job.Started = DateTime.UtcNow;
                current = job;
            }
            string? error = null;
            var tmp = job.WavPath + ".tmp";
            try
            {
                maker.Make(job.Request, tmp, new SyncProgress(p => { lock (gate) { job.Total = p.Total; job.Done = Math.Max(job.Done, p.Done); } }), job.Cancellation.Token);
                File.Move(tmp, job.WavPath, overwrite: true);
            }
            catch (OperationCanceledException) when (job.Cancellation.IsCancellationRequested) { error = "cancelled"; TryDelete(tmp); }
            catch (Exception e)
            {
                error = e is JobFailure ? e.Message : $"{e.GetType().Name}: {e.Message}";
                TryDelete(tmp);
            }
            lock (gate)
            {
                current = null;
                job.Finished = DateTime.UtcNow;
                if (jobs.ContainsKey(job.Id))
                {
                    job.State = error is null ? JobStates.Done : JobStates.Failed;
                    job.Error = error ?? "";
                    Trim();
                }
                else TryDelete(job.WavPath);   // forgotten while it ran
            }
        }
    }

    private void Trim()
    {
        var finished = order.Where(i => jobs[i].State is JobStates.Done or JobStates.Failed).ToList();
        foreach (var old in finished.Take(Math.Max(0, finished.Count - KeepFinished)))
        {
            TryDelete(jobs[old].WavPath);
            jobs.Remove(old);
            order.Remove(old);
        }
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    public void Dispose()
    {
        queue.CompleteAdding();
        lock (gate) { current?.Cancellation.Cancel(); }
        worker.Join(TimeSpan.FromSeconds(10));
    }

    private sealed class SyncProgress : IProgress<(int Done, int Total)>
    {
        private readonly Action<(int Done, int Total)> handler;
        public SyncProgress(Action<(int Done, int Total)> handler) => this.handler = handler;
        public void Report((int Done, int Total) value) => handler(value);
    }
}
