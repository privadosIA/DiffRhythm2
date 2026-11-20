using DiffRhythm2.Service;

namespace DiffRhythm2.Tests;

/// <summary>The service's environment: names of the Python service, defaults, and messages that name the variable.</summary>
public class ServiceSettingsTests
{
    private static ServiceSettings From(params (string, string)[] vars) => ServiceSettings.From(name => vars.Where(v => v.Item1 == name).Select(v => v.Item2).FirstOrDefault());

    [Fact]
    public void Defaults()
    {
        var s = From();
        Assert.Equal("/ckpt", s.ModelDirectory);
        Assert.Equal("", s.Token);
        Assert.Equal(8788, s.Port);
        Assert.True(Path.IsPathRooted(s.JobsDirectory));
        Assert.EndsWith(Path.Combine("diffrhythm2", "jobs"), s.JobsDirectory);
        Assert.NotEqual("/tmp/diffrhythm2-jobs", s.JobsDirectory);
        Assert.Equal("1", s.Version);
        Assert.Equal("cpu", s.Device);
        Assert.Equal("process", s.Isolation);
    }

    [Fact]
    public void EveryVariable_IsRead()
    {
        var s = From(("DIFFRHYTHM2_CKPT", "/m"), ("DIFFRHYTHM2_TOKEN", "t"), ("DIFFRHYTHM2_PORT", "9000"), ("DIFFRHYTHM2_JOBS", "/j"), ("DIFFRHYTHM2_VERSION", "2"), ("DIFFRHYTHM2_DEVICE", "cuda:1"), ("DIFFRHYTHM2_ISOLATION", "inprocess"));
        Assert.Equal(new ServiceSettings("/m", "t", 9000, "/j", "2", "cuda:1", "inprocess"), s);
    }

    [Theory]
    [InlineData("DIFFRHYTHM2_PORT", "abc")]
    [InlineData("DIFFRHYTHM2_PORT", "0")]
    [InlineData("DIFFRHYTHM2_PORT", "70000")]
    [InlineData("DIFFRHYTHM2_PORT", "-1")]
    [InlineData("DIFFRHYTHM2_DEVICE", "gpu")]
    [InlineData("DIFFRHYTHM2_DEVICE", "cuda:x")]
    [InlineData("DIFFRHYTHM2_ISOLATION", "thread")]
    [InlineData("DIFFRHYTHM2_ISOLATION", "Process")]
    public void InvalidValues_NameTheVariable(string name, string value)
    {
        var e = Assert.Throws<SettingsException>(() => From((name, value)));
        Assert.Contains(name, e.Message);
        Assert.Contains($"'{value}'", e.Message);
    }

    [Fact]
    public void EmptyVariables_MeanTheDefault()
    {
        Assert.Equal(8788, From(("DIFFRHYTHM2_PORT", "")).Port);
    }
}
