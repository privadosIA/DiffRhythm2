namespace DiffRhythm2.Tests;

public class CheckpointEnvTests
{
    [Fact]
    public void TestProjectRuns() => Assert.True(true);

    [RequiresCheckpoint]
    public void CheckpointIsReachable() => Assert.NotNull(Checkpoint.Dir);
}
