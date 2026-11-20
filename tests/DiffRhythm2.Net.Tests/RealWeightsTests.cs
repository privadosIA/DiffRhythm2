using System.Diagnostics;
using System.Security.Cryptography;
using DiffRhythm2.Weights;

namespace DiffRhythm2.Tests;

/// <summary>The conversion of the real checkpoints against the files the Python tools wrote (skipped without the checkpoint directory and its sources).</summary>
public class RealWeightsTests : IDisposable
{
    private readonly string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "dr2-weights-test-" + Guid.NewGuid().ToString("N"));

    public void Dispose() { try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch (IOException) { } }

    private static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }

    [RequiresCheckpoint]
    public void RealCheckpoints_ConvertToTheSameBytesAsThePythonTools()
    {
        var ckpt = Checkpoint.Dir!;
        foreach (var file in new[] { "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json" })
            if (!File.Exists(Path.Combine(ckpt, file))) { Console.WriteLine($"skipped: {file} is not in the checkpoint directory"); return; }
        if (WeightConverter.FindSource(ckpt, "decoder.bin") is null || WeightConverter.FindSource(ckpt, "pytorch_model.bin") is null || WeightConverter.FindSource(ckpt, "tokenizer.json") is null)
        { Console.WriteLine("skipped: the pickled sources are not in the checkpoint directory"); return; }

        Directory.CreateDirectory(dir);
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long before = proc.WorkingSet64;
        long peak = 0;
        using var stop = new CancellationTokenSource();
        var monitor = Task.Run(() => { while (!stop.IsCancellationRequested) { proc.Refresh(); peak = Math.Max(peak, proc.WorkingSet64); Thread.Sleep(50); } });

        var results = WeightConverter.Convert(dir, new ConversionOptions(From: ckpt));
        stop.Cancel();
        monitor.Wait();

        Assert.All(results, r => Assert.Equal(ConversionStatus.Written, r.Status));
        foreach (var file in new[] { "decoder.safetensors", "mulan_text.safetensors", "mulan_text_tokenizer.json" })
            Assert.Equal(Sha(Path.Combine(ckpt, file)), Sha(Path.Combine(dir, file)));
        Console.WriteLine($"peak working set growth during conversion: {(peak - before) / 1048576.0:F0} MiB (sources: 3.1 GB)");
        Assert.True(peak - before < 600L * 1048576, $"the conversion should stream; the working set grew by {(peak - before) / 1048576} MiB");
    }

    [RequiresMulanAudio]
    public void RealAudioTower_ConvertsToTheSameBytesAsThePythonTool()
    {
        var ckpt = Checkpoint.Dir!;
        if (WeightConverter.FindSource(ckpt, "pytorch_model.bin") is null) { Console.WriteLine("skipped: pytorch_model.bin is not in the checkpoint directory"); return; }

        Directory.CreateDirectory(dir);
        using var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long before = proc.WorkingSet64;
        long peak = 0;
        using var stop = new CancellationTokenSource();
        var monitor = Task.Run(() => { while (!stop.IsCancellationRequested) { proc.Refresh(); peak = Math.Max(peak, proc.WorkingSet64); Thread.Sleep(50); } });

        var used = new List<string>();
        var result = WeightConverter.ConvertTensors(dir, WeightConverter.Audio, new ConversionOptions(From: ckpt, Audio: true), used, default);
        stop.Cancel();
        monitor.Wait();

        Assert.Equal(ConversionStatus.Written, result.Status);
        var header = SafeTensorsHeader.Read(Path.Combine(dir, WeightConverter.AudioFile));
        Assert.Equal(531, header.Entries.Count);
        Assert.Equal(18, header.Entries.Count(e => e.DType == "I64"));
        Assert.Equal(513, header.Entries.Count(e => e.DType == "F32"));
        Assert.Equal(Sha(Path.Combine(ckpt, WeightConverter.AudioFile)), Sha(Path.Combine(dir, WeightConverter.AudioFile)));
        Console.WriteLine($"peak working set growth during the audio conversion: {(peak - before) / 1048576.0:F0} MiB (output: 1.3 GB)");
        Assert.True(peak - before < 600L * 1048576, $"the conversion should stream; the working set grew by {(peak - before) / 1048576} MiB");
    }
}
