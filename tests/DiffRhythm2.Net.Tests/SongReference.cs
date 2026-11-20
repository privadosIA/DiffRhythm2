using System.Text.Json;
using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Tests;

/// <summary>The song pipeline fixtures: tests/fixtures/song (small ones) and tests/fixtures/song_e2e (the reference song), each verified against its MANIFEST.md.</summary>
public static class SongReference
{
    private static readonly Lazy<FixtureDirectory> Small = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "song")));
    private static readonly Lazy<FixtureDirectory> EndToEnd = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "song_e2e")));
    private static JsonDocument? validation;
    private static Dictionary<string, Tensor>? noise, audio, song, cudaT;
    private static readonly Lazy<FixtureDirectory> CudaDirectory = new(() => new FixtureDirectory(Path.Combine(FixtureDirectory.Find().Dir, "song_cuda")));

    public static FixtureDirectory SmallDir => Small.Value;
    public static FixtureDirectory E2eDir => EndToEnd.Value;

    /// <summary>The cases given to the service's validation: name, body, typed, and ok or error.</summary>
    public static JsonElement Validation => (validation ??= JsonDocument.Parse(SmallDir.ReadAllText("validate.json"))).RootElement;

    /// <summary>noise_f32, noise_f16 (drawn in half) and noise_f16_via_cast, each [4,1,10,64], for seed 1234.</summary>
    public static Dictionary<string, Tensor> Noise => noise ??= SmallDir.Load("noise.safetensors");

    /// <summary>stereo_in/stereo_out and pcm_in/pcm_out.</summary>
    public static Dictionary<string, Tensor> Audio => audio ??= SmallDir.Load("audio.safetensors");

    /// <summary>tokens, style, noise, latent, latent64, wave, stereo of the reference song.</summary>
    public static Dictionary<string, Tensor> Song => song ??= E2eDir.Load("song.safetensors");

    public static JsonElement SongMeta => JsonDocument.Parse(E2eDir.ReadAllText("meta.json")).RootElement;

    /// <summary>The CUDA reference (informational): noise_cuda_f16 / noise_cuda_f32 [4,1,10,64] and, from the service's own CUDA song, latent, wave, stereo.</summary>
    public static Dictionary<string, Tensor> Cuda => cudaT ??= CudaDirectory.Value.Load("noise_cuda.safetensors").Concat(CudaDirectory.Value.Load("song_cuda.safetensors")).ToDictionary(kv => kv.Key, kv => kv.Value);

    public const long Seed = 1234;
}
