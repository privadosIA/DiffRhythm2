using System.Text;
using DiffRhythm2.Service;

ServiceSettings settings;
try { settings = ServiceSettings.FromEnvironment(); }
catch (SettingsException e)
{
    await Console.Error.WriteLineAsync($"error: {e.Message}");
    return 2;
}

// a worker process: one song, then exit (see Worker)
if (args.Length == 2 && args[0] == WorkerProtocol.Argument)
{
    var utf8 = new UTF8Encoding(false);
    using var stdin = new StreamReader(Console.OpenStandardInput(), utf8);
    using var stdout = new StreamWriter(Console.OpenStandardOutput(), utf8) { AutoFlush = true };
    using var stderr = new StreamWriter(Console.OpenStandardError(), utf8) { AutoFlush = true };
    return Worker.Run(args[1], stdin, stdout, stderr, new SongMaker(settings), abandon: () =>
    {
        Worker.RemovePartial(args[1]);          // nobody is waiting for the song: stop now, not at the end of the block
        System.Diagnostics.Process.GetCurrentProcess().Kill();   // not Environment.Exit: it waits for the native threads that are computing
    });
}

ISongMaker maker = settings.Isolation == ServiceSettings.Process ? new ProcessSongMaker() : new SongMaker(settings);
using var engine = new JobEngine(maker, settings.JobsDirectory);
var api = new ServiceApp(settings, engine, new HealthProbe(settings, engine));
var app = ServiceApp.Build(api, $"http://0.0.0.0:{settings.Port}");
await app.StartAsync();
Console.WriteLine($"diffrhythm2 service {settings.Version} listening on :{settings.Port} (models: {settings.ModelDirectory})");
await app.WaitForShutdownAsync();
return 0;
