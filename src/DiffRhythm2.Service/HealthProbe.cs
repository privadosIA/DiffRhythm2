using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Serialization;
using DiffRhythm2.Pipeline;
using DiffRhythm2.Text;

namespace DiffRhythm2.Service;

internal sealed record GpuView([property: JsonPropertyName("name")] string Name, [property: JsonPropertyName("free_mb")] long? FreeMb, [property: JsonPropertyName("total_mb")] long? TotalMb);

internal sealed record HealthView(
    [property: JsonPropertyName("ready")] bool Ready, [property: JsonPropertyName("version")] string Version, [property: JsonPropertyName("busy")] bool Busy,
    [property: JsonPropertyName("queued")] int Queued, [property: JsonPropertyName("gpu")] GpuView? Gpu, [property: JsonPropertyName("missing")] IReadOnlyList<string> Missing);

/// <summary>
/// What <c>GET /health</c> says: whether the model files, the phonemizer library and the device are all usable, and what card there is. The checks are cheap and
/// cached for a few seconds, and never touch the worker, so health answers while a song is being made.
/// </summary>
internal sealed class HealthProbe
{
    private readonly ServiceSettings settings;
    private readonly JobEngine engine;
    private readonly TimeSpan cache;
    private readonly object gate = new();
    private (bool Ready, IReadOnlyList<string> Missing, GpuView? Gpu)? last;
    private DateTime lastAt;

    // seams for tests
    public Func<IReadOnlyList<string>> MissingFiles { get; init; }
    public Func<string?> PhonemizerProblem { get; init; } = () => EspeakEngine.TryGetShared(out _, out var why) ? null : why;
    public Func<string?> DeviceProblem { get; init; }
    public Func<GpuView?> Gpu { get; init; }

    public HealthProbe(ServiceSettings settings, JobEngine engine, TimeSpan? cacheFor = null)
    {
        this.settings = settings;
        this.engine = engine;
        cache = cacheFor ?? TimeSpan.FromSeconds(5);
        MissingFiles = () => Directory.Exists(settings.ModelDirectory) ? new ModelDirectory(settings.ModelDirectory).Missing : ModelDirectory.RequiredFiles.ToList();
        DeviceProblem = () =>
        {
            try { SongGenerator.ResolveDevice(settings.Device); return null; }
            catch (DeviceUnavailableException e) { return e.Message; }
        };
        Gpu = () => ReadGpu(settings.Device);
    }

    public HealthView Get()
    {
        (bool Ready, IReadOnlyList<string> Missing, GpuView? Gpu) state;
        lock (gate)
        {
            if (last is null || DateTime.UtcNow - lastAt >= cache)
            {
                var missing = MissingFiles();
                bool ready = missing.Count == 0 && PhonemizerProblem() is null && DeviceProblem() is null;
                last = (ready, missing, Gpu());
                lastAt = DateTime.UtcNow;
            }
            state = last.Value;
        }
        return new HealthView(state.Ready, settings.Version, engine.Busy, engine.Queued, state.Gpu, state.Missing);
    }

    // ---- the card ---------------------------------------------------------------------------------------------------------------------

    private static GpuView? ReadGpu(string device)
    {
        if (!device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)) return null;
        int index = device.Contains(':') ? int.Parse(device[(device.IndexOf(':') + 1)..], CultureInfo.InvariantCulture) : 0;
        var fallback = new GpuView($"cuda:{index}", null, null);
        try
        {
            var smi = FindNvidiaSmi();
            if (smi is null) return fallback;
            using var p = Process.Start(new ProcessStartInfo(smi, $"--query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits -i {index}")
            { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false });
            if (p is null) return fallback;
            var text = p.StandardOutput.ReadToEndAsync();
            if (!p.WaitForExit(2000)) { try { p.Kill(); } catch (InvalidOperationException) { } return fallback; }
            return ParseSmi(text.Result) ?? fallback;
        }
        catch (Win32Exception) { return fallback; }
        catch (InvalidOperationException) { return fallback; }
    }

    /// <summary>nvidia-smi at a fixed install location, not looked up through PATH (a binary planted on PATH would run with the service's rights); null when it is not there.</summary>
    private static string? FindNvidiaSmi()
    {
        var candidates = OperatingSystem.IsWindows()
            ? new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "nvidia-smi.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
            }
            : new[] { "/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi", "/bin/nvidia-smi" };
        return candidates.FirstOrDefault(File.Exists);
    }

    /// <summary>Reads the first line of <c>nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader,nounits</c>.</summary>
    public static GpuView? ParseSmi(string output)
    {
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault();
        if (line is null) return null;
        var parts = line.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 3 || parts[0].Length == 0) return null;
        long? Num(string s) => long.TryParse(s, NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : null;
        return new GpuView(parts[0], Num(parts[1]), Num(parts[2]));
    }
}
