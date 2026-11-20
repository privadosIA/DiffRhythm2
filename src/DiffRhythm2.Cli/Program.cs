using DiffRhythm2.Cli;

// diffrhythm2 generate | check | --help  (see Cli.cs and docs/song-pipeline.md)
using var cancel = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancel.Cancel(); };
return Cli.Run(args, Console.Out, Console.Error, Console.In, cancel.Token);
