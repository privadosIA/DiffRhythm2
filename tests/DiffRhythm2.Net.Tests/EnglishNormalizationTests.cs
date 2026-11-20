using System.Numerics;
using System.Text.Json;
using DiffRhythm2.Text;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>Number words and text normalization against the tables dumped from the reference (setuptools' vendored inflect). None of this needs espeak.</summary>
public class EnglishNormalizationTests
{
    private readonly ITestOutputHelper log;

    public EnglishNormalizationTests(ITestOutputHelper log) => this.log = log;

    private static void AssertTable(ITestOutputHelper log, string name, JsonElement table, Func<string, string> compute)
    {
        int ok = 0, expectedErrors = 0;
        var failures = new List<string>();
        foreach (var entry in table.EnumerateObject())
        {
            bool referenceFails = entry.Value.ValueKind == JsonValueKind.Object;
            try
            {
                var got = compute(entry.Name);
                if (referenceFails) failures.Add($"{entry.Name}: the reference fails ({entry.Value.GetProperty("error").GetString()}) but the port returned \"{got}\"");
                else if (got != entry.Value.GetString()) failures.Add($"{entry.Name}: expected \"{entry.Value.GetString()}\", got \"{got}\"");
                else ok++;
            }
            catch (G2pException) when (referenceFails) { expectedErrors++; }
            catch (G2pException e) { failures.Add($"{entry.Name}: the port fails ({e.Message}) where the reference gives \"{entry.Value.GetString()}\""); }
        }
        log.WriteLine($"{name}: {ok} equal, {expectedErrors} errors on both sides, {failures.Count} differences");
        Assert.True(failures.Count == 0, string.Join("\n", failures.Take(15)));
    }

    [Fact]
    public void Cardinals_MatchTheReferenceTable() =>
        AssertTable(log, "cardinal", G2pReference.Numbers.GetProperty("cardinal"), k => NumberWords.ToWords(BigInteger.Parse(k), andWord: ""));

    [Fact]
    public void Years_InGroupsOfTwo_MatchTheReferenceTable() =>
        AssertTable(log, "year", G2pReference.Numbers.GetProperty("year"), k => NumberWords.ToWords(BigInteger.Parse(k), andWord: "", zero: "oh", group: 2));

    [Fact]
    public void OrdinalStrings_MatchTheReferenceTable() =>
        AssertTable(log, "ordinal", G2pReference.Numbers.GetProperty("ordinal"), k => NumberWords.ToWords(k));

    [Fact]
    public void Fractions_MatchTheReferenceTable() =>
        AssertTable(log, "fraction", G2pReference.Numbers.GetProperty("fraction"), k =>
        {
            var parts = k.Split('/');
            return EnglishNormalizer.FractionToWords(BigInteger.Parse(parts[0]), BigInteger.Parse(parts[1]));
        });

    [Fact]
    public void PowersOfTen_StopWhereTheReferenceStops()
    {
        foreach (var entry in G2pReference.Numbers.GetProperty("powers_of_ten").EnumerateObject())
        {
            var n = BigInteger.Pow(10, int.Parse(entry.Name));
            if (entry.Value.ValueKind == JsonValueKind.Object)
                Assert.Throws<NumberOutOfRangeException>(() => NumberWords.ToWords(n, andWord: ""));
            else
                Assert.StartsWith(entry.Value.GetString()!, NumberWords.ToWords(n, andWord: ""));
        }
        Assert.Equal(36, NumberWords.MaxDigits);
    }

    [Fact]
    public void NumberOutOfRange_NamesTheNumber()
    {
        var e = Assert.Throws<NumberOutOfRangeException>(() => NumberWords.ToWords(BigInteger.Pow(10, 40)));
        Assert.Contains("1" + new string('0', 40), e.Message);
    }

    [Fact]
    public void OrdinalOfSpokenNumbers_FollowsTheReference()
    {
        Assert.Equal("eighth", NumberWords.Ordinal("eight"));
        Assert.Equal("twentieth", NumberWords.Ordinal("twenty"));
        Assert.Equal("one hundredth", NumberWords.Ordinal("one hundred"));
        Assert.Equal("twenty-first", NumberWords.Ordinal("twenty-one"));
        Assert.Equal("fourth", NumberWords.Ordinal("four"));
    }

    private void CompareNormalizationCases(string table, Func<string, string> normalize)
    {
        var failures = new List<string>();
        int ok = 0, errors = 0;
        foreach (var entry in G2pReference.Numbers.GetProperty(table).EnumerateObject())
        {
            bool referenceFails = entry.Value.TryGetProperty("error", out var err);
            try
            {
                var got = normalize(entry.Name);
                if (referenceFails) failures.Add($"\"{entry.Name}\": the reference fails ({err.GetProperty("type").GetString()}) but the port returned \"{got}\"");
                else if (got != entry.Value.GetProperty("out").GetString()) failures.Add($"\"{entry.Name}\": expected \"{entry.Value.GetProperty("out").GetString()}\", got \"{got}\"");
                else ok++;
            }
            catch (G2pException e) when (referenceFails) { errors++; log.WriteLine($"\"{entry.Name}\": both fail ({e.Message})"); }
            catch (G2pException e) { failures.Add($"\"{entry.Name}\": the port fails ({e.Message})"); }
        }
        log.WriteLine($"{table}: {ok} equal, {errors} errors on both sides, {failures.Count} differences");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void NumberRules_MatchTheReference() => CompareNormalizationCases("normalize_numbers", EnglishNormalizer.NormalizeNumbers);

    [Fact]
    public void AbbreviationsAndNumbers_MatchTheReference() => CompareNormalizationCases("english_to_ipa_text", EnglishNormalizer.Normalize);

    [Fact]
    public void WholeCorpus_NormalizedText_MatchesTheReference()
    {
        int checkedLines = 0;
        foreach (var line in G2pReference.GoodLines)
        {
            var text = line.GetProperty("text").GetString()!;
            Assert.Equal(line.GetProperty("normalized").GetString(), EnglishNormalizer.Normalize(text));
            checkedLines++;
        }
        log.WriteLine($"{checkedLines} corpus lines normalized identically");
        Assert.True(checkedLines > 90);
    }

    [Fact]
    public void Abbreviations_OnlyReplaceWholeWords()
    {
        Assert.Equal("stop the doctor", EnglishNormalizer.ExpandAbbreviations("stop the doctor"));
        Assert.Equal("mister Smith", EnglishNormalizer.ExpandAbbreviations("Mr Smith"));
        Assert.Equal("mister. Smith", EnglishNormalizer.ExpandAbbreviations("MR. Smith"));
    }
}
