using DiffRhythm2.Text;

namespace DiffRhythm2.Tests;

/// <summary>
/// An espeak stand-in that answers with what the reference's espeak (1.51) answered for the same chunk of text, recorded while the golden run
/// ran. It lets the whole phonemizer pipeline be verified on a machine without the library; a chunk that was not recorded fails loudly.
/// </summary>
public sealed class ReplayEngine : IEspeakEngine
{
    private readonly Dictionary<string, string> transcripts = G2pReference.Transcripts;

    public string TextToPhonemes(string text) =>
        transcripts.TryGetValue(text, out var raw) ? raw : throw new KeyNotFoundException($"the reference never phonemized the chunk \"{text}\" (it is not in transcripts.json)");
}
