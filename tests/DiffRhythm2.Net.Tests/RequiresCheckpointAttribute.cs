using MusicGen.Core.Testing;

namespace DiffRhythm2.Tests;

/// <summary>Checkpoint directory, taken from DIFFRHYTHM2_CKPT (never inside the repo); audio directory from DIFFRHYTHM2_AUDIO_DIR; device from DIFFRHYTHM2_DEVICE (cpu by default, or cuda).</summary>
public static class Checkpoint
{
    public static readonly CheckpointEnvironment Env = new("DIFFRHYTHM2_CKPT", "DIFFRHYTHM2_AUDIO_DIR", "DIFFRHYTHM2_DEVICE");

    public static string? Dir => Env.Dir;

    public static string? AudioDir => Env.AudioDir;

    public static string DeviceName => Env.DeviceName;
}

/// <summary>[Fact] that is skipped if DIFFRHYTHM2_CKPT does not point to an existing directory.</summary>
public sealed class RequiresCheckpointAttribute : RequiresCheckpointFactAttribute
{
    public RequiresCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>[Fact] that needs the checkpoint and only makes sense on CPU; skipped with an explicit reason otherwise.</summary>
public sealed class RequiresCpuCheckpointAttribute : RequiresCpuCheckpointFactAttribute
{
    public RequiresCpuCheckpointAttribute() : base(Checkpoint.Env) { }
}

/// <summary>[Fact] that writes audio to be listened to: it needs the checkpoint and DIFFRHYTHM2_AUDIO_DIR.</summary>
public sealed class RequiresAudioDirAttribute : RequiresAudioDirFactAttribute
{
    public RequiresAudioDirAttribute() : base(Checkpoint.Env) { }
}
