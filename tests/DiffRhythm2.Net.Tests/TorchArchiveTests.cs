using System.Security.Cryptography;
using System.Text.Json;
using DiffRhythm2.Weights;
using MusicGen.Core.Testing;

namespace DiffRhythm2.Tests;

/// <summary>The torch.save reader against archives written by the real torch.save (fixtures/torch_archive).</summary>
public class TorchArchiveTests
{
    internal static readonly string Dir = Path.Combine(FixtureDirectory.Find().Dir, "torch_archive");
    internal static string Of(string name) => Path.Combine(Dir, name);

    private static JsonElement Expected(string key)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Of("expected.json")));
        return doc.RootElement.GetProperty(key).Clone();
    }

    private static string Sha(TorchArchive a, TorchTensorInfo t)
    {
        using var h = new HashingStream();
        a.CopyTo(t, h);
        return h.Hex();
    }

    private static void AssertMatches(string archive, string key, string? stateKey = null)
    {
        using var a = new TorchArchive(Of(archive), stateKey);
        var expected = Expected(key).EnumerateArray().ToList();
        Assert.Equal(expected.Count, a.Tensors.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            var t = a.Tensors[i];
            Assert.Equal(expected[i].GetProperty("name").GetString(), t.Name);
            Assert.Equal(expected[i].GetProperty("dtype").GetString(), t.DType.SafeTensorsName());
            Assert.Equal(expected[i].GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()), t.Shape);
            Assert.Equal(expected[i].GetProperty("sha256").GetString(), Sha(a, t));
        }
    }

    [Fact] public void StateDictionary_ListsEveryNameTypeShapeAndByte() => AssertMatches("plain.pt", "plain.pt");

    [Fact] public void WrappedStateDictionary_IsReadUnderItsKey() => AssertMatches("wrapped.pt", "wrapped.pt", "generator");

    [Fact]
    public void MissingWrapperKey_IsReported()
    {
        var e = Assert.Throws<TorchArchiveException>(() => new TorchArchive(Of("wrapped.pt"), "nope"));
        Assert.Contains("nope", e.Message);
    }

    [Fact] public void SharedStorage_EachTensorIsReadFromItsOwnOffset() => AssertMatches("shared.pt", "shared.pt");

    [Fact] public void DeflatedEntries_AreAccepted() => AssertMatches("deflated.pt", "deflated.pt");

    [Fact]
    public void HostilePickle_IsRefusedByNameAndNothingRuns()
    {
        var e = Assert.Throws<TorchArchiveException>(() => new TorchArchive(Of("hostile.pt")));
        Assert.Contains("posix.system", e.Message.Replace(" ", "."), StringComparison.Ordinal);   // pickle spells os.system as "posix system" on Linux
        Assert.Contains("nothing was run", e.Message);
    }

    [Fact]
    public void UnsupportedOpcode_IsRefusedNamingItAndWhereItIs()
    {
        var e = Assert.Throws<TorchArchiveException>(() => new TorchArchive(Of("unsupported.pt")));
        Assert.Contains("opcode", e.Message);
        Assert.Contains("offset", e.Message);
    }

    [Fact]
    public void NonContiguousTensor_IsRefusedNamingIt()
    {
        using var a = new TorchArchive(Of("noncontig.pt"));
        var t = Assert.Single(a.Tensors);
        var e = Assert.Throws<TorchArchiveException>(() => a.CopyTo(t, Stream.Null));
        Assert.Equal("t", e.Tensor);
        Assert.Contains("contiguous", e.Message);
    }

    [Fact]
    public void LegacyFormat_IsRefusedSayingSo()
    {
        var e = Assert.Throws<TorchArchiveException>(() => new TorchArchive(Of("legacy.pt")));
        Assert.Contains("legacy", e.Message);
    }

    [Fact]
    public void ArchiveCutInHalf_IsReportedAsDamaged()
    {
        var e = Assert.Throws<TorchArchiveException>(() => new TorchArchive(Of("cut.pt")));
        Assert.Contains("damaged", e.Message);
    }

    [Fact]
    public void ShortStorage_NamesTheTensor()
    {
        using var a = new TorchArchive(Of("short_storage.pt"));
        var name = Expected("short_storage_first_tensor").GetString()!;
        var t = a.Tensors.Single(x => x.Name == name);
        var e = Assert.Throws<TorchArchiveException>(() => a.CopyTo(t, Stream.Null));
        Assert.Equal(name, e.Tensor);
        Assert.Contains("cut short", e.Message);
        // the other tensors are still readable
        a.CopyTo(a.Tensors.Single(x => x.Name == "b.bias"), Stream.Null);
    }

    [Fact]
    public void NotAZipFile_IsRefused()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "hello");
            Assert.Contains("not a torch.save archive", Assert.Throws<TorchArchiveException>(() => new TorchArchive(path)).Message);
        }
        finally { File.Delete(path); }
    }
}

/// <summary>A write-only stream that hashes what it is given.</summary>
internal sealed class HashingStream : Stream
{
    private readonly IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    public string Hex() => Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    public override void Write(byte[] buffer, int offset, int count) => hash.AppendData(buffer, offset, count);
    public override void Write(ReadOnlySpan<byte> buffer) => hash.AppendData(buffer);
    public override bool CanRead => false;
    public override bool CanSeek => false;
    public override bool CanWrite => true;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
    public override void Flush() { }
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    protected override void Dispose(bool disposing) { if (disposing) hash.Dispose(); base.Dispose(disposing); }
}
