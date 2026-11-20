// The smallest program that makes a song with the library: dotnet run --project samples/MakeSong -- <model dir> <song.wav>
using DiffRhythm2.Pipeline;

if (args.Length != 2)
{
    await Console.Error.WriteLineAsync("usage: MakeSong <model directory> <output.wav>");
    return 2;
}

var lyrics = """
    [verse]
    Walking down the road tonight
    Stars are shining way up high
    [chorus]
    Sing along, sing along
    Feel the music all night long
    """;

// CPU in float32 by default; Device: "cuda" (with the CUDA build of the backend) gives float16 on the GPU.
var generator = new SongGenerator(new SongOptions(ModelDirectory: args[0]));
var request = new SongRequest(lyrics, "Pop, Piano, Female vocals", Seconds: 20, Seed: 1234);

var progress = new Progress<SongProgress>(p =>
{
    if (p.Event == SongEvent.BlockDone) Console.WriteLine($"block {p.Done}/{p.Total}");
});

try
{
    SongResult song = generator.Generate(request, progress);
    song.WriteWav(args[1]);                      // 16-bit, 48 kHz, stereo as the service makes it; WriteWav(path, stereo: false) for mono
    Console.WriteLine($"seed={song.Seed}, {song.Audio.Length / (double)song.SampleRate:F1} s written to {args[1]}");
    return 0;
}
catch (SongException e)                          // bad request, missing model files, no such device
{
    await Console.Error.WriteLineAsync(e.Message);
    return 1;
}
