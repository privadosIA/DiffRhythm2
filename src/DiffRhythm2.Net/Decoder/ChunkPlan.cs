namespace DiffRhythm2.Decoder;

/// <summary>
/// One chunk of the chunked decoding: which latent frames go in and which part of its audio is copied where.
/// All sample figures are in output samples; <see cref="ChunkStart"/>/<see cref="ChunkEnd"/> are offsets inside the chunk's own audio.
/// </summary>
/// <param name="InputStart">First latent frame of the chunk.</param>
/// <param name="InputLength">Latent frames in the chunk.</param>
/// <param name="OutStart">First sample of the final waveform this chunk writes.</param>
/// <param name="OutEnd">One past the last sample of the final waveform this chunk writes.</param>
/// <param name="ChunkStart">First sample of the chunk's audio that is copied.</param>
/// <param name="ChunkEnd">One past the last sample of the chunk's audio that is copied.</param>
public readonly record struct ChunkSpan(int InputStart, int InputLength, long OutStart, long OutEnd, long ChunkStart, long ChunkEnd);

/// <summary>
/// The geometry of <c>Generator.decode_audio</c> as a pure function (so it can be checked without any weights): chunks of
/// <c>chunkSize</c> frames starting every <c>chunkSize − overlap</c> frames, plus a final chunk aligned to the end when the regular ones do not
/// end exactly there. Each overlap is cut at its midpoint (<c>overlap/2</c> frames, integer division, from each chunk edge that has a neighbour);
/// chunks are pasted in order, so where the final chunk overlaps its predecessor by more than <c>overlap</c> frames its audio wins.
/// A sequence shorter than one chunk (which the reference cannot decode) is a single chunk of its own length; an empty one has no chunks.
/// </summary>
public static class ChunkPlan
{
    public static IReadOnlyList<ChunkSpan> Create(int frames, int samplesPerFrame, int chunkSize = 20, int overlap = 5)
    {
        if (frames < 0) throw new ArgumentOutOfRangeException(nameof(frames));
        if (chunkSize <= 0 || overlap < 0 || overlap >= chunkSize) throw new ArgumentException($"invalid chunking: chunk size {chunkSize}, overlap {overlap}");
        if (frames == 0) return Array.Empty<ChunkSpan>();
        if (frames < chunkSize) return new[] { new ChunkSpan(0, frames, 0, (long)frames * samplesPerFrame, 0, (long)frames * samplesPerFrame) };

        int hop = chunkSize - overlap;
        var starts = new List<int>();
        for (int i = 0; i <= frames - chunkSize; i += hop) starts.Add(i);
        if (starts[^1] + chunkSize != frames) starts.Add(frames - chunkSize);   // final chunk, aligned to the end

        long total = (long)frames * samplesPerFrame;
        long chunkSamples = (long)chunkSize * samplesPerFrame;
        long cut = (long)(overlap / 2) * samplesPerFrame;
        var plan = new List<ChunkSpan>(starts.Count);
        for (int n = 0; n < starts.Count; n++)
        {
            bool last = n == starts.Count - 1;
            long tStart = last ? total - chunkSamples : (long)starts[n] * samplesPerFrame;
            long tEnd = tStart + chunkSamples;
            long cStart = 0, cEnd = chunkSamples;
            if (n > 0) { tStart += cut; cStart += cut; }
            if (!last) { tEnd -= cut; cEnd -= cut; }
            plan.Add(new ChunkSpan(starts[n], chunkSize, tStart, tEnd, cStart, cEnd));
        }
        return plan;
    }
}
