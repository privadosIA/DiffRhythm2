using DiffRhythm2.Text;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>The phonemizer pipeline (cleaning, punctuation split and restore, tidying of each chunk) verified by replaying the reference's recorded espeak answers: no library needed.</summary>
public class EspeakPhonemizerTests
{
    private readonly ITestOutputHelper log;

    public EspeakPhonemizerTests(ITestOutputHelper log) => this.log = log;

    private sealed class FakeEngine : IEspeakEngine
    {
        public readonly List<string> Seen = new();

        public string TextToPhonemes(string text)
        {
            Seen.Add(text);
            return "ab_c d";   // a made-up answer, enough to see how chunks are tidied and joined
        }
    }

    // ---- punctuation split and restore --------------------------------------------------------------------------------------------------

    [Fact]
    public void Marks_SplitTheTextAndKeepTheirPlace()
    {
        var (chunks, marks) = Punctuation.Preserve(new[] { "hello, my world!" });
        Assert.Equal(new[] { "hello", "my world" }, chunks);
        Assert.Equal(new[] { (",", MarkPosition.Inside), ("!", MarkPosition.End) }, marks.Select(m => (m.Text.Trim(), m.Position)).ToArray());

        (chunks, marks) = Punctuation.Preserve(new[] { ",lead" });
        Assert.Equal(new[] { "lead" }, chunks);
        Assert.Equal(MarkPosition.Begin, marks[0].Position);

        (chunks, marks) = Punctuation.Preserve(new[] { "?!" });
        Assert.Empty(chunks);
        Assert.Equal(MarkPosition.Alone, marks[0].Position);
    }

    [Fact]
    public void CommaAndPeriodBetweenDigits_AreNotMarks()
    {
        var (chunks, marks) = Punctuation.Preserve(new[] { "version 1.2.3 and 4,5" });
        Assert.Equal(new[] { "version 1.2.3 and 4,5" }, chunks);
        Assert.Empty(marks);

        // only a separator with a digit on BOTH sides is exempt: here the period follows a letter, so it is a mark
        (chunks, marks) = Punctuation.Preserve(new[] { "wait.5" });
        Assert.Equal(new[] { "wait", "5" }, chunks);

        (chunks, _) = Punctuation.Preserve(new[] { "end. 5" });
        Assert.Equal(new[] { "end", "5" }, chunks);
    }

    [Fact]
    public void Restore_PutsEachMarkBackAsItsOwnItem()
    {
        var (chunks, marks) = Punctuation.Preserve(new[] { "a, b. c" });
        var text = chunks.Select(c => c.ToUpperInvariant()).ToList();
        // a space inside a mark (", " and ". " here) becomes the word separator
        Assert.Equal(new[] { "A,|_|B.|_|C" }, Punctuation.Restore(text, marks, "|_|", strip: true));
    }

    // ---- chunk tidying and cleaning ------------------------------------------------------------------------------------------------------

    [Fact]
    public void TidyChunk_JoinsWordsAndPhones_RemovesFlagsAndStress()
    {
        Assert.Equal("ab|c|_|d", EspeakPhonemizer.TidyChunk("ab_c d"));
        Assert.Equal("h|ɛ|l|oʊ|_|w|ɜː|l|d", EspeakPhonemizer.TidyChunk("h_ˈɛ_l_ˌoʊ w_ˈɜː_l_d"));
        Assert.Equal("a|b|_|c", EspeakPhonemizer.TidyChunk("a_b (fr)c"));      // the flag is removed, its word kept
        Assert.Equal("a|b|", EspeakPhonemizer.TidyChunk("a__b_"));             // runs collapse; a trailing separator survives here and is stripped at the end of Phonemize
        Assert.Equal("", EspeakPhonemizer.TidyChunk("(en)"));
    }

    [Theory]
    [InlineData("  Hello，world。Bye！ ", "Hello,world.Bye!")]
    [InlineData("Wait...", "Wait…")]
    [InlineData("it’s", "it's")]
    [InlineData("Dr. Dre & the Co.", "Dr.Dre the Co.")]
    [InlineData("a   b\tc", "a b c")]
    [InlineData("well - ok", "well ok")]
    public void Clean_FollowsTheReference(string text, string expected) => Assert.Equal(expected, EspeakPhonemizer.Clean(text));

    [Fact]
    public void Clean_UsesPythonWordCharacters()
    {
        Assert.Equal("café", EspeakPhonemizer.Clean("café"));
        Assert.Equal("cafe", EspeakPhonemizer.Clean("café"));      // a combining accent is not a Python word character, so it is removed
        Assert.Equal("2² ½", EspeakPhonemizer.Clean("2² ½"));   // superscript two and one half are numeric, so they stay
        Assert.Equal("I you", EspeakPhonemizer.Clean("I ♥ you"));
    }

    [Fact]
    public void Chunks_ReachTheEngineWithoutTheMarks()
    {
        var engine = new FakeEngine();
        Assert.Equal("ab|c|_|d|,|ab|c|_|d", new EspeakPhonemizer(engine).Phonemize("Hello, world"));
        Assert.Equal(new[] { "Hello", "world" }, engine.Seen);
    }

    // ---- replay of the reference ----------------------------------------------------------------------------------------------------------

    [Fact]
    public void Cleaning_MatchesTheReference_OnTheWholeCorpus()
    {
        foreach (var line in G2pReference.GoodLines)
            Assert.Equal(line.GetProperty("cleaned").GetString(), EspeakPhonemizer.Clean(line.GetProperty("normalized").GetString()!));
    }

    [Fact]
    public void Phonemizer_ReplayingEspeak_MatchesTheReference_OnTheWholeCorpus()
    {
        var phonemizer = new EspeakPhonemizer(new ReplayEngine());
        int n = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            Assert.Equal(line.GetProperty("phonemizer").GetString(), phonemizer.Phonemize(line.GetProperty("normalized").GetString()!));
            n++;
        }
        log.WriteLine($"{n} corpus lines phonemized identically from the recorded espeak answers");
        Assert.True(n > 90);
    }
}
