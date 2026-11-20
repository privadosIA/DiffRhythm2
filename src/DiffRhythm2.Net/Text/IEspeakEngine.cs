namespace DiffRhythm2.Text;

/// <summary>
/// The espeak-ng call the phonemizer is built on: a text chunk to the raw IPA that <c>espeak_TextToPhonemes</c> returns for it, clauses joined
/// with a space, phonemes separated by <c>_</c>, stress marks still present. Implemented by <see cref="EspeakEngine"/>; tests replace it with
/// the recorded answers of the reference.
/// </summary>
public interface IEspeakEngine
{
    string TextToPhonemes(string text);
}
