using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Sampling;

/// <summary>
/// The reference's end-of-song rule (<c>sample_block_cache</c>): after each block, if the last frame is close to an all-ones frame (mean squared
/// error over its channels &lt;= 0.05) the song is over. The sequence is then cut by scanning backwards from the last frame past the run of such
/// frames, cutting at the first frame outside the run, and THAT frame is excluded too (the reference slices with the negative index it stopped at).
/// If every frame is close to all-ones the result is empty. The extra dropped frame is kept on purpose, for parity.
/// </summary>
public static class EndOfSong
{
    public const double Threshold = 0.05;

    /// <summary>The end position (number of frames to keep) when the song ended in the last block, or null when sampling goes on.</summary>
    /// <param name="stream"><c>[1, N, channels]</c>: every frame generated so far.</param>
    public static long? EndPosition(Tensor stream)
    {
        using var scope = NewDisposeScope();
        long n = stream.shape[1];
        if (n == 0) return null;
        var mse = (stream[0] - 1).pow(2).mean(new long[] { -1 }).cpu().to(ScalarType.Float32).data<float>().ToArray();   // one value per frame

        bool Close(long fromEnd) => Math.Abs(mse[n + fromEnd]) <= Threshold;            // fromEnd is the reference's negative index
        long pos = -1;
        if (!Close(pos)) return null;
        while (Close(pos) && Math.Abs(pos) < n)
            pos--;
        return n + pos;
    }
}
