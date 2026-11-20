using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.AudioTower;

/// <summary>
/// The front end of MuQ-large, as the reference runs it in float32: <c>torchaudio.transforms.MelSpectrogram</c> (24 kHz, 2048-point periodic Hann
/// window, hop 240, centered with reflection padding, power 2, 128 HTK bands from 0 to 12 kHz, no band normalization), <c>AmplitudeToDB</c> (power,
/// floor 1e-10, no dynamic-range limit), removal of the last frame, and normalization with the mean and standard deviation of the MuQ-large training
/// set (the <c>stat</c> entry of the MuQ-large configuration). The operations are the ones torchaudio performs, in the same order, so the result agrees
/// with the reference to a few units of float32 rounding.
/// </summary>
public sealed class MelFrontEnd : IDisposable
{
    /// <summary>Sample rate of the audio, in Hz.</summary>
    public const int SampleRate = WavReader.RequiredSampleRate;

    /// <summary>FFT size and window length.</summary>
    public const int NFft = 2048;

    /// <summary>Hop between frames, in samples.</summary>
    public const int Hop = 240;

    /// <summary>Number of mel bands.</summary>
    public const int Mels = 128;

    /// <summary>Number of frequency bins of the one-sided spectrum.</summary>
    public const int Bins = NFft / 2 + 1;

    /// <summary>Mean of the decibel mel spectrogram over the MuQ-large training set (<c>melspec_2048_mean</c>).</summary>
    public const double Mean = 6.768444971712967;

    /// <summary>Standard deviation of the decibel mel spectrogram over the MuQ-large training set (<c>melspec_2048_std</c>).</summary>
    public const double Std = 18.417922652295623;

    private const double DecibelFloor = 1e-10;

    private readonly Tensor window;
    private readonly Tensor filterBank;
    private readonly Device device;

    /// <summary>Creates the front end on a device (the window and the filter bank are kept there).</summary>
    public MelFrontEnd(Device? device = null)
    {
        this.device = device ?? CPU;
        using var _ = no_grad();
        window = hann_window(NFft, periodic: true, device: this.device);
        filterBank = FilterBank().to(this.device);
    }

    /// <summary>The HTK mel filter bank <c>[1025, 128]</c> exactly as <c>torchaudio.functional.melscale_fbanks(1025, 0, 12000, 128, 24000)</c> builds it, in float32.</summary>
    public static Tensor FilterBank()
    {
        using var _ = no_grad();
        using var allFreqs = linspace(0, SampleRate / 2, Bins);                                   // float32
        double mMin = HzToMel(0.0), mMax = HzToMel(SampleRate / 2.0);
        using var mPts = linspace(mMin, mMax, Mels + 2);
        using var fPts = MelToHz(mPts);

        using var fDiff = fPts[1..] - fPts[..^1];                                                   // [Mels + 1]
        using var slopes = fPts.unsqueeze(0) - allFreqs.unsqueeze(1);                              // [Bins, Mels + 2]
        using var zero = zeros(1);
        using var down = (slopes[.., ..^2] * -1.0) / fDiff[..^1];
        using var up = slopes[.., 2..] / fDiff[1..];
        using var fb = maximum(zero, minimum(down, up));
        return fb.contiguous().clone();
    }

    private static double HzToMel(double freq) => 2595.0 * Math.Log10(1.0 + freq / 700.0);

    private static Tensor MelToHz(Tensor mels)
    {
        using var scaled = mels / 2595.0;
        using var ten = full_like(scaled, 10.0);
        using var pow = ten.pow(scaled);
        using var minusOne = pow - 1.0;
        return minusOne * 700.0;
    }

    /// <summary>
    /// The decibel mel spectrogram of a clip, <c>[1, 128, n / 240 + 1]</c> (the last frame is still there). The clip must have at least
    /// <see cref="AudioClips.MinimumClipSamples"/> samples.
    /// </summary>
    public Tensor Decibels(ReadOnlySpan<float> clip)
    {
        if (clip.Length < AudioClips.MinimumClipSamples)
            throw new ArgumentException($"the clip has {clip.Length} samples; at least {AudioClips.MinimumClipSamples} are needed", nameof(clip));
        using var _ = no_grad();
        using var wave = tensor(clip.ToArray(), new long[] { 1, clip.Length }, ScalarType.Float32, device);
        // The centering is done here (reflection padding of n_fft / 2 on both sides, which is what stft(center: true) does) and stft is called without it: libtorch's
        // own centering path reads a freed object now and then and brought the process down with SIGSEGV (found with MALLOC_PERTURB_, on a CI runner). The result is the same.
        using var batched = wave.unsqueeze(0);                                                      // [1, 1, n]: reflection padding wants a channel dimension
        using var padded3 = nn.functional.pad(batched, new long[] { NFft / 2, NFft / 2 }, PaddingModes.Reflect);
        using var padded = padded3.squeeze(0);                                                      // [1, n + n_fft]
        using var spec = stft(padded, NFft, Hop, NFft, window, center: false, normalized: false, onesided: true, return_complex: true);
        using var power = spec.abs().pow(2.0);                                                      // [1, Bins, frames]
        using var mel = matmul(power.transpose(-1, -2), filterBank).transpose(-1, -2);              // [1, Mels, frames]
        using var clamped = clamp(mel, min: DecibelFloor);
        return log10(clamped) * 10.0;
    }

    /// <summary>The input of the encoder for a clip: <see cref="Decibels"/> without its last frame, normalized, shape <c>[1, 128, n / 240]</c>.</summary>
    public Tensor Features(ReadOnlySpan<float> clip)
    {
        using var _ = no_grad();
        using var db = Decibels(clip);
        using var cut = db[.., .., ..^1];
        using var centered = cut - Mean;
        return centered / Std;
    }

    /// <summary>Releases the window and the filter bank.</summary>
    public void Dispose()
    {
        window.Dispose();
        filterBank.Dispose();
    }
}
