using System.Diagnostics;
using MusicGen.Core.Testing;

namespace DiffRhythm2.Tests;

/// <summary>The start script's usage errors and help (it never reaches dotnet or the models in these cases). Skipped where bash is not available.</summary>
public class StartScriptTests
{
    private static readonly string Script = Path.Combine(Path.GetFullPath(Path.Combine(FixtureDirectory.Find().Dir, "..", "..")), "scripts", "run-service.sh");

    private static (int Code, string Out, string Err)? Run(params string[] args)
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS()) return null;
        var psi = new ProcessStartInfo("bash") { RedirectStandardOutput = true, RedirectStandardError = true };
        psi.ArgumentList.Add(Script);
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        var o = p.StandardOutput.ReadToEndAsync();
        var e = p.StandardError.ReadToEndAsync();
        p.WaitForExit();
        return (p.ExitCode, o.Result, e.Result);
    }

    [Fact]
    public void Help_ListsEveryOptionAndVariable()
    {
        if (Run("--help") is not { } r) return;
        Assert.Equal(0, r.Code);
        foreach (var word in new[] { "--model", "--port", "--device", "--token-file", "--jobs", "--isolation", "DIFFRHYTHM2_ISOLATION", "--espeak-lib", "--espeak-data", "--build", "DIFFRHYTHM2_CKPT", "DIFFRHYTHM2_PORT", "DIFFRHYTHM2_DEVICE", "DIFFRHYTHM2_TOKEN", "8788" })
            Assert.Contains(word, r.Out);
    }

    [Theory]
    [InlineData(new[] { "--port", "abc" }, "port")]
    [InlineData(new[] { "--port", "70000" }, "port")]
    [InlineData(new[] { "--device", "gpu" }, "device")]
    [InlineData(new[] { "--isolation", "thread" }, "isolation")]
    [InlineData(new[] { "--bogus" }, "unknown option")]
    [InlineData(new[] { "--model" }, "needs a value")]
    [InlineData(new[] { "--token-file", "/nonexistent/token" }, "cannot read")]
    public void WrongUsage_ExitsWith2_AndSaysWhy(string[] args, string fragment)
    {
        if (Run(args) is not { } r) return;
        Assert.Equal(2, r.Code);
        Assert.Contains(fragment, r.Err);
    }
}
