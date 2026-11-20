using System.Globalization;

namespace DiffRhythm2.Service;

/// <summary>A setting that cannot be used. The message names the environment variable.</summary>
internal sealed class SettingsException : Exception
{
    public SettingsException(string message) : base(message) { }
}

/// <summary>What the service is told by its environment, with the variable names of the Python service.</summary>
/// <param name="ModelDirectory">DIFFRHYTHM2_CKPT: the six model files.</param>
/// <param name="Token">DIFFRHYTHM2_TOKEN: bearer token; empty leaves the API open.</param>
/// <param name="Port">DIFFRHYTHM2_PORT: the port to listen on, on all interfaces.</param>
/// <param name="JobsDirectory">DIFFRHYTHM2_JOBS: where finished songs are kept.</param>
/// <param name="Version">DIFFRHYTHM2_VERSION: the version string health reports.</param>
/// <param name="Device">DIFFRHYTHM2_DEVICE: cpu, cuda or cuda:N.</param>
/// <param name="Isolation">DIFFRHYTHM2_ISOLATION: <c>process</c> (a worker process per song) or <c>inprocess</c>.</param>
internal sealed record ServiceSettings(string ModelDirectory, string Token, int Port, string JobsDirectory, string Version, string Device, string Isolation = ServiceSettings.Process)
{
    public const string Ckpt = "DIFFRHYTHM2_CKPT", TokenVariable = "DIFFRHYTHM2_TOKEN", PortVariable = "DIFFRHYTHM2_PORT", Jobs = "DIFFRHYTHM2_JOBS",
        VersionVariable = "DIFFRHYTHM2_VERSION", DeviceVariable = "DIFFRHYTHM2_DEVICE", IsolationVariable = "DIFFRHYTHM2_ISOLATION";

    public const string Process = "process", InProcess = "inprocess";

    public static ServiceSettings FromEnvironment() => From(Environment.GetEnvironmentVariable);

    public static ServiceSettings From(Func<string, string?> get)
    {
        string Text(string name, string fallback) => get(name) is { Length: > 0 } v ? v : fallback;

        int port = 8788;
        if (get(PortVariable) is { Length: > 0 } p)
        {
            if (!int.TryParse(p, NumberStyles.None, CultureInfo.InvariantCulture, out port) || port < 1 || port > 65535)
                throw new SettingsException($"{PortVariable} must be a port number from 1 to 65535, got '{p}'");
        }

        var device = Text(DeviceVariable, "cpu");
        if (!IsDevice(device)) throw new SettingsException($"{DeviceVariable} must be cpu, cuda or cuda:N, got '{device}'");

        var isolation = Text(IsolationVariable, Process);
        if (isolation is not (Process or InProcess)) throw new SettingsException($"{IsolationVariable} must be process or inprocess, got '{isolation}'");

        return new ServiceSettings(Text(Ckpt, "/ckpt"), get(TokenVariable) ?? "", port, Text(Jobs, DefaultJobsDirectory()), Text(VersionVariable, "1"), device, isolation);
    }

    /// <summary>
    /// The service's own folder in the user's data directory (<c>$XDG_DATA_HOME</c> or <c>~/.local/share</c> on Linux), so the songs are not left in a directory every
    /// user can read; the temporary directory when the machine has no data directory (no <c>HOME</c>).
    /// </summary>
    internal static string DefaultJobsDirectory()
    {
        var data = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(data.Length > 0 ? data : Path.GetTempPath(), "diffrhythm2", "jobs");
    }

    public static bool IsDevice(string d) =>
        d.Equals("cpu", StringComparison.OrdinalIgnoreCase) || d.Equals("cuda", StringComparison.OrdinalIgnoreCase)
        || (d.StartsWith("cuda:", StringComparison.OrdinalIgnoreCase) && int.TryParse(d.AsSpan(5), NumberStyles.None, CultureInfo.InvariantCulture, out _));
}
