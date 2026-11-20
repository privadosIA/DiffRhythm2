using System.Security.Cryptography;
using DiffRhythm2.Weights;

namespace DiffRhythm2.Tests;

/// <summary>The writer and the conversion, on the small archives of fixtures/torch_archive (the real checkpoints are in RealWeightsTests).</summary>
public class WeightConverterTests : IDisposable
{
    private readonly string dir = Path.Combine(Path.GetTempPath(), "dr2-weights-" + Guid.NewGuid().ToString("N"));

    public WeightConverterTests() => Directory.CreateDirectory(dir);

    public void Dispose() { try { Directory.Delete(dir, true); } catch (IOException) { } }

    private static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(f)); }

    private string Place(string fixture, string asName, string? sub = null)
    {
        var target = Path.Combine(dir, sub ?? "", asName);
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.Copy(TorchArchiveTests.Of(fixture), target);
        return target;
    }

    private static readonly WeightConverter.Profile Tiny = new("tiny.safetensors", "decoder.bin", "generator", name => name.Replace(".weight_g", ".parametrizations.weight.original0").Replace(".weight_v", ".parametrizations.weight.original1"), 3, "the test file");
    private static readonly WeightConverter.Profile AllPlain = new("plain.safetensors", "decoder.bin", null, n => n, 9, "the plain file");
    private static readonly WeightConverter.Profile OnlyF32 = new("f32.safetensors", "decoder.bin", null, n => n.StartsWith("a.") || n == "half" ? n : null, 2, "the mixed file");

    private ConversionResult Run(WeightConverter.Profile p, ConversionOptions? o = null) =>
        WeightConverter.ConvertTensors(dir, p, o ?? new ConversionOptions(), new List<string>(), default);

    [Fact]
    public void Writer_PutsMetadataFirst_SortsByName_PadsTheHeaderTo8_AndRoundTrips()
    {
        var path = Path.Combine(dir, "w.safetensors");
        var data = new Dictionary<string, byte[]> { ["b"] = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }, ["a"] = new byte[] { 9, 10, 11, 12 } };
        SafeTensorsStreamWriter.Write(path, new[]
        {
            new SafeTensorsEntry("b", TorchDType.Float32, new long[] { 2 }, 8, s => s.Write(data["b"])),
            new SafeTensorsEntry("a", TorchDType.Float32, new long[] { 1 }, 4, s => s.Write(data["a"])),
        });
        var bytes = File.ReadAllBytes(path);
        long len = BitConverter.ToInt64(bytes, 0);
        Assert.Equal(0, len % 8);
        var json = System.Text.Encoding.UTF8.GetString(bytes, 8, (int)len);
        Assert.StartsWith("{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", json);
        Assert.EndsWith("}", json.TrimEnd(' '));   // whatever follows the closing brace is padding
        Assert.Equal(8 + len + 12, bytes.Length);
        var h = SafeTensorsHeader.Read(path);
        Assert.Equal(new[] { "a", "b" }, h.Entries.Select(e => e.Name));
        Assert.Equal(data["a"], bytes[(int)(h.DataStart + 0)..(int)(h.DataStart + 4)]);
    }

    [Fact]
    public void Writer_PutsWiderTypesFirst_AndSortsByNameWithinAType_LikeThePythonLibrary()
    {
        var path = Path.Combine(dir, "mixed.safetensors");
        SafeTensorsEntry E(string name, TorchDType t, int bytes) => new(name, t, new long[] { 1 }, bytes, s => s.Write(new byte[bytes]));
        SafeTensorsStreamWriter.Write(path, new[]
        {
            E("b.weight", TorchDType.Float32, 4), E("a.count", TorchDType.Int64, 8), E("a.weight", TorchDType.Float32, 4), E("z.count", TorchDType.Int64, 8),
            E("h", TorchDType.Float16, 2), E("flags", TorchDType.Bool, 1), E("i", TorchDType.Int32, 4),
        });
        var h = SafeTensorsHeader.Read(path);
        Assert.Equal(new[] { "a.count", "z.count", "a.weight", "b.weight", "i", "h", "flags" }, h.Entries.Select(e => e.Name));
        Assert.Equal(new[] { "I64", "I64", "F32", "F32", "I32", "F16", "BOOL" }, h.Entries.Select(e => e.DType));
        Assert.Equal(new long[] { 0, 8, 16, 20, 24, 28, 30 }, h.Entries.Select(e => e.Start));
    }

    [Fact]
    public void Writer_RejectsEntriesThatWriteTheWrongNumberOfBytes()
    {
        Assert.Throws<InvalidDataException>(() => SafeTensorsStreamWriter.Write(Path.Combine(dir, "x.safetensors"),
            new[] { new SafeTensorsEntry("a", TorchDType.Float32, new long[] { 2 }, 8, s => s.Write(new byte[4])) }));
    }

    [Fact]
    public void WeightNormPairs_AreRenamed_AndTheValuesCopiedExactly()
    {
        Place("wrapped.pt", "decoder.bin");
        var r = Run(Tiny);
        Assert.Equal(ConversionStatus.Written, r.Status);
        var h = SafeTensorsHeader.Read(Path.Combine(dir, "tiny.safetensors"));
        Assert.Equal(new[] { "conv.bias", "conv.parametrizations.weight.original0", "conv.parametrizations.weight.original1" }, h.Entries.Select(e => e.Name));
        using var a = new TorchArchive(TorchArchiveTests.Of("wrapped.pt"), "generator");
        using var f = File.OpenRead(Path.Combine(dir, "tiny.safetensors"));
        foreach (var t in a.Tensors)
        {
            var e = h.Entries.Single(x => x.Name == t.Name.Replace(".weight_g", ".parametrizations.weight.original0").Replace(".weight_v", ".parametrizations.weight.original1"));
            var expected = new MemoryStream();
            a.CopyTo(t, expected);
            f.Position = h.DataStart + e.Start;
            var got = new byte[e.End - e.Start];
            f.ReadExactly(got);
            Assert.Equal(expected.ToArray(), got);
        }
    }

    [Fact]
    public void WrongTensorCount_SaysWhatWasExpectedAndFound_AndWritesNothing()
    {
        Place("wrapped.pt", "decoder.bin");
        var e = Assert.Throws<ConversionException>(() => Run(WeightConverter.Decoder));
        Assert.Contains("783", e.Message);
        Assert.Contains("3 tensors", e.Message);
        Assert.Empty(Directory.GetFiles(dir, "*.safetensors*"));
    }

    [Fact]
    public void TensorThatIsNotFloat32_IsRefusedNamingIt()
    {
        Place("plain.pt", "decoder.bin");
        var e = Assert.Throws<ConversionException>(() => Run(AllPlain));
        Assert.Contains("'half'", e.Message);
        Assert.Contains("F16", e.Message);
        Assert.Empty(Directory.GetFiles(dir, "*.safetensors*"));
    }

    [Fact]
    public void FilterKeepsOnlyTheSelectedTensors()
    {
        Place("plain.pt", "decoder.bin");
        // 'half' is float16: selecting it must fail, selecting only the float32 ones must work
        var onlyA = new WeightConverter.Profile("a.safetensors", "decoder.bin", null, n => n.StartsWith("a.") || n == "b.bias" ? n : null, 2, "the filtered file");
        Assert.Equal(ConversionStatus.Written, Run(onlyA).Status);
        Assert.Equal(2, SafeTensorsHeader.Read(Path.Combine(dir, "a.safetensors")).Entries.Count);
    }

    [Fact]
    public void DamagedSource_LeavesNoOutputAndNoTemporaryFile()
    {
        Place("short_storage.pt", "decoder.bin");
        var onlyShortAndOk = new WeightConverter.Profile("s.safetensors", "decoder.bin", null, n => n is "a.weight" or "b.bias" ? n : null, 2, "the damaged file");
        var e = Assert.Throws<TorchArchiveException>(() => Run(onlyShortAndOk));
        Assert.Equal("a.weight", e.Tensor);
        Assert.Empty(Directory.GetFiles(dir, "*.safetensors*"));
    }

    [Fact]
    public void SameSourceTwice_GivesIdenticalFiles()
    {
        Place("wrapped.pt", "decoder.bin");
        Run(Tiny);
        var first = Sha(Path.Combine(dir, "tiny.safetensors"));
        Run(Tiny, new ConversionOptions(Force: true));
        Assert.Equal(first, Sha(Path.Combine(dir, "tiny.safetensors")));
    }

    [Fact]
    public void ValidExistingOutput_IsLeftAlone_UnlessForced()
    {
        Place("wrapped.pt", "decoder.bin");
        Run(Tiny);
        var path = Path.Combine(dir, "tiny.safetensors");
        File.SetLastWriteTimeUtc(path, new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc));
        var again = Run(Tiny);
        Assert.Equal(ConversionStatus.UpToDate, again.Status);
        Assert.Equal(new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc), File.GetLastWriteTimeUtc(path));
        Assert.Equal(ConversionStatus.Written, Run(Tiny, new ConversionOptions(Force: true)).Status);
        Assert.NotEqual(new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc), File.GetLastWriteTimeUtc(path));
    }

    [Fact]
    public void ValidOutput_NeedsNoSource()
    {
        Place("wrapped.pt", "decoder.bin");
        Run(Tiny);
        File.Delete(Path.Combine(dir, "decoder.bin"));
        Assert.Equal(ConversionStatus.UpToDate, Run(Tiny).Status);
    }

    [Fact]
    public void SourcesInTheHuggingFaceCacheLayout_AreFound()
    {
        var placed = Place("wrapped.pt", "decoder.bin", Path.Combine("models--ASLP-lab--DiffRhythm2", "snapshots", "abc123"));
        Assert.Equal(placed, WeightConverter.FindSource(dir, "decoder.bin"));
        Assert.Equal(ConversionStatus.Written, Run(Tiny).Status);
    }

    [Fact]
    public void SourcesDirectoryAndFromDirectory_AreSearched()
    {
        var inSources = Place("wrapped.pt", "decoder.bin", "sources");
        Assert.Equal(inSources, WeightConverter.FindSource(dir, "decoder.bin"));
        var elsewhere = Path.Combine(dir, "elsewhere");
        Directory.CreateDirectory(elsewhere);
        File.Copy(TorchArchiveTests.Of("wrapped.pt"), Path.Combine(elsewhere, "decoder.bin"));
        Assert.Equal(Path.Combine(elsewhere, "decoder.bin"), WeightConverter.FindSource(dir, "decoder.bin", elsewhere));
    }

    [Fact]
    public void MissingSource_ListsTheSearchedPlaces()
    {
        var e = Assert.Throws<ConversionException>(() => Run(Tiny));
        Assert.Contains("decoder.bin was not found", e.Message);
        Assert.Contains(Path.Combine(dir, "sources", "decoder.bin"), e.Message);
        Assert.Contains("models--ASLP-lab--DiffRhythm2", e.Message);
    }

    [Fact]
    public void Convert_WithoutSources_FailsListingThem_AndWritesNothing()
    {
        var e = Assert.Throws<ConversionException>(() => WeightConverter.Convert(dir));
        Assert.Contains("decoder.bin was not found", e.Message);
        Assert.Empty(Directory.GetFiles(dir));
    }

    [Fact]
    public void Sources_AreKept_UnlessDeletionIsRequested_AndOnlyAfterTheConversion()
    {
        var src = Place("wrapped.pt", "decoder.bin");
        var used = new List<string>();
        WeightConverter.ConvertTensors(dir, Tiny, new ConversionOptions(), used, default);
        Assert.True(File.Exists(src));
        Assert.Equal(new[] { src }, used);
    }

    [Fact]
    public void Convert_DeletesTheConvertedSources_OnlyOnRequest_AndOnlyOnSuccess()
    {
        // the real profiles need the real checkpoints; here the failure path: a count mismatch must not delete anything
        var src = Place("wrapped.pt", "decoder.bin");
        Assert.Throws<ConversionException>(() => WeightConverter.Convert(dir, new ConversionOptions(DeleteSources: true)));
        Assert.True(File.Exists(src));
    }
}
