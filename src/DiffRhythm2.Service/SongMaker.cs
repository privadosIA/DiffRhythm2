using System.Runtime.InteropServices;
using DiffRhythm2.Pipeline;

namespace DiffRhythm2.Service;

/// <summary>Makes songs with the pipeline. A generator is created for every song, so the models are loaded when it starts and released when it ends.</summary>
internal sealed class SongMaker : ISongMaker
{
    private readonly ServiceSettings settings;

    public SongMaker(ServiceSettings settings) => this.settings = settings;

    public void Make(SongRequest request, string wavPath, IProgress<(int Done, int Total)> progress, CancellationToken cancellation)
    {
        try
        {
            var generator = new SongGenerator(new SongOptions(settings.ModelDirectory, settings.Device));
            var result = generator.Generate(request, new Sync(p =>
            {
                if (p.Stage == SongStage.Network && p.Event == SongEvent.BlockDone) progress.Report((p.Done, p.Total));
            }), cancellation);
            result.WriteWav(wavPath);
        }
        finally { ReturnMemoryToTheSystem(); }
    }

    // The models are freed when a song ends, but glibc keeps the freed pages in its arenas: ask it to give them back, so an idle service stays small.
    [DllImport("libc", EntryPoint = "malloc_trim")]
    private static extern int MallocTrim(nuint pad);

    private static void ReturnMemoryToTheSystem()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        if (!OperatingSystem.IsLinux()) return;
        try { MallocTrim(0); } catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException) { }   // not glibc: nothing to do
    }

    private sealed class Sync : IProgress<SongProgress>
    {
        private readonly Action<SongProgress> handler;
        public Sync(Action<SongProgress> handler) => this.handler = handler;
        public void Report(SongProgress value) => handler(value);
    }
}
