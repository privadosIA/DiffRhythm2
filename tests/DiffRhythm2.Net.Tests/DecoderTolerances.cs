namespace DiffRhythm2.Tests;

/// <summary>
/// Tolerances of the BigVGAN fp32 parity gates (max abs error divided by the largest magnitude of the reference tensor, or per-channel relative
/// error for RMS). They are frozen from the measured errors (the tests print every figure) and from the reference's own run-to-run variance:
/// the same reference code, run twice in the same process, is bit-identical, but two separate processes differ by up to 6e-7 abs on the
/// 40-frame waveform (1.4e-6 relative), 2.6e-5 abs on the largest stage signal (stage 5, values up to 6.7: 3.8e-6 relative) and 1.6e-5 on the
/// clipped loud waveform (1.6e-5 relative), because thread scheduling changes the summation order of the convolutions (tools/README.md).
/// </summary>
public static class DecoderTolerances
{
    /// <summary>
    /// Stage signals. Measured worst case 5.5e-6 (stage 5 of the random chunk; 1.1e-6 for the earlier stages), which is the reference's own
    /// cross-process variance of that stage (3.8e-6); gated at 5e-5.
    /// </summary>
    public const double Stage = 5e-5;

    /// <summary>
    /// Waveforms (a chunk, the 40-frame decode, each seam window, the clipped loud chunk). Measured worst case 1.9e-6 for real latents and
    /// 1.6e-5 for the loud chunk, whose 92074 clipped samples sit at +-1 and amplify the difference; that equals the reference's own cross-process
    /// drift there (1.6e-5). Gated at 5e-5.
    /// </summary>
    public const double Wave = 5e-5;

    /// <summary>Per-channel RMS of every stage (all channels, not only the stored ones): measured worst case 5.6e-7, gated at 1e-5.</summary>
    public const double Rms = 1e-5;
}
