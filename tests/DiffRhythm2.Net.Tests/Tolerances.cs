namespace DiffRhythm2.Tests;

/// <summary>
/// Relative tolerances (max abs error divided by the largest magnitude of the reference tensor) of the fp32 parity gates. They are frozen from the measured errors (the tests print every figure; see design D6 of add-dit-block-flow-matching).
/// </summary>
public static class Tolerances
{
    /// <summary>Single pieces and whole-DiT forwards: measured at 9e-7 or better (cache values of layer 15), gated at 1e-5.</summary>
    public const double Piece = 1e-5;

    /// <summary>
    /// The first block (15 Euler steps x 2 DiT evaluations, conditioned only on the lyrics) and every single forward: measured at 1e-6 or better
    /// (block 0, with guidance 8e-7, without lyrics 5e-7, guidance off 9e-7), gated at 1e-5.
    /// </summary>
    public const double Block = 1e-5;

    /// <summary>
    /// Multi-block runs. The sampler amplifies rounding strongly (the fp32 reference is 1.2e-5, 1.2e-3, 2.5e-3, 0.19 and 3.4e-2 away from float64 on
    /// blocks 0-4), so a fixed relative tolerance means nothing there: the port is gated against that distance instead (design D6). Measured ratios
    /// port-vs-reference over reference-vs-float64 were 0.2, 0.12, 0.12 and 0.27 on blocks 0-3; the gate is 1.0.
    /// </summary>
    public const double Sensitivity = 1.0;

    /// <summary>
    /// MuLan text tower stages (token ids in, one forward pass, no sampler to amplify anything), each fed the reference's previous stage. Measured on
    /// seven prompts: embeddings 4.9e-8, encoder layer 0 1.3e-7, encoder output 3.2e-7, transformer layer 0 2.0e-7, transformer output 4.6e-7 relative
    /// to the largest magnitude; the reference is bit-for-bit reproducible across processes, so there is no noise to allow for. Gated at 1e-5.
    /// </summary>
    public const double Tower = 1e-5;

    /// <summary>Final embeddings from text (tokenizer included): measured 8.7e-7 relative to the largest component (about 0.15), the 512-position text 1.4e-7. Gated at 1e-5.</summary>
    public const double Embedding = 1e-5;

    /// <summary>Cosine similarity between an embedding and the reference's: measured 1.000000000 on every prompt. Gated at 1 - 1e-7.</summary>
    public const double EmbeddingCosine = 0.9999999;

    /// <summary>
    /// Waveform of the reference song (mono and both stereo channels) made here from the same request and seed on CPU in float32. Measured 1.1e-5 relative to the
    /// peak (SNR 107 dB); the latent it comes from is 4.9e-7 away in block 0 and 1.6e-5 in block 1 (the reference itself is 1.2e-3 from float64 there). Gated at 1e-4.
    /// </summary>
    public const double SongWave = 1e-4;

    /// <summary>Largest difference, in 16-bit steps, between our WAV and the reference song's. Measured 1 (2,444 of 384,000 samples differ by one step: rounding of the waveform's last bits). Gated at 2.</summary>
    public const int SongWavSteps = 2;

    /// <summary>
    /// Mel filter bank, decibel spectrogram and normalized features against torchaudio (add-muq-mulan-audio). The port runs the same libtorch operations in the
    /// same order, and the error measured on all three test signals is exactly 0 (bit for bit). Gated at 1e-6 relative to the largest magnitude, which allows a few
    /// float32 units of difference from another CPU's FFT or matrix-multiply code without letting a wrong window, bank or constant through.
    /// </summary>
    public const double MelFilterBank = 1e-6;

    /// <summary>See <see cref="MelFilterBank"/>.</summary>
    public const double MelDecibels = 1e-6;

    /// <summary>See <see cref="MelFilterBank"/>.</summary>
    public const double MelFeatures = 1e-6;

    /// <summary>
    /// MuQ-large stages (add-muq-mulan-audio), each fed the reference's previous stage: the subsampling is exactly equal, Conformer layers 1, 6 and 12 measured 2.4e-7,
    /// 2.2e-7 and 2.2e-7 and the final layer norm 1.9e-7, relative to the largest magnitude of the stage. The reference is bit-for-bit reproducible across processes.
    /// Gated at 1e-5, like the other towers.
    /// </summary>
    public const double MuqStage = 1e-5;

    /// <summary>The 12 layers chained from the spectrogram: measured 3.7e-7 (the error does not grow through the layers). Gated at 1e-5.</summary>
    public const double MuqChain = 1e-5;

    /// <summary>
    /// Audio embeddings (projection, mean, latent, clip and signal; add-muq-mulan-audio), from the spectrogram and from raw samples: measured 1.2e-7 to 2.8e-7 relative to the
    /// largest component (about 0.15) for the projection and the clips, exactly 0 for the mean, the latent and the normalization when each is fed the reference's previous
    /// stage. Gated at 1e-5, like the other towers.
    /// </summary>
    public const double AudioEmbedding = 1e-5;

    /// <summary>Absolute difference of a similarity score (a dot product of two embeddings of length at most 1) from the reference's: measured 1e-7 at most. Gated at 1e-6.</summary>
    public const double Similarity = 1e-6;
}
