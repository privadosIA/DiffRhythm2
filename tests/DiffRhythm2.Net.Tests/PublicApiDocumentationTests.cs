using System.Reflection;
using System.Xml.Linq;
using DiffRhythm2.Pipeline;
using Xunit.Abstractions;

namespace DiffRhythm2.Tests;

/// <summary>Every public type and member of the pipeline namespace carries XML documentation: the library is meant to be used from other projects.</summary>
public class PublicApiDocumentationTests
{
    private readonly ITestOutputHelper log;

    public PublicApiDocumentationTests(ITestOutputHelper log) => this.log = log;

    private static HashSet<string> Documented()
    {
        var assembly = typeof(SongGenerator).Assembly;
        var xml = Path.ChangeExtension(assembly.Location, ".xml");
        Assert.True(File.Exists(xml), $"the documentation file {xml} was not generated");
        return XDocument.Load(xml).Descendants("member")
            .Where(m => !string.IsNullOrWhiteSpace(m.Element("summary")?.Value) || m.Element("inheritdoc") is not null)
            .Select(m => m.Attribute("name")!.Value).ToHashSet();
    }

    private static bool IsCompilerGenerated(MemberInfo m) =>
        m.GetCustomAttribute<System.Runtime.CompilerServices.CompilerGeneratedAttribute>() is not null
        || m.Name is "EqualityContract" or "Deconstruct" or "PrintMembers" or "<Clone>$" or "Equals" or "GetHashCode" or "ToString" or "GetType" or "MemberwiseClone" or "Finalize";

    [Fact]
    public void EveryPublicTypeAndMemberOfThePipelineNamespace_HasASummary()
    {
        var documented = Documented();
        var missing = new List<string>();
        var types = typeof(SongGenerator).Assembly.GetExportedTypes().Where(t => t.Namespace is "DiffRhythm2.Pipeline" or "DiffRhythm2.Weights").ToList();
        Assert.True(types.Count >= 10, "the pipeline namespace should have its types");

        foreach (var type in types)
        {
            string tid = "T:" + type.FullName!.Replace('+', '.');
            if (!documented.Contains(tid)) missing.Add(tid);
            var flags = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
            foreach (var p in type.GetProperties(flags).Where(p => !IsCompilerGenerated(p)))
                if (!documented.Contains($"P:{type.FullName}.{p.Name}")) missing.Add($"P:{type.FullName}.{p.Name}");
            foreach (var f in type.GetFields(flags).Where(f => !IsCompilerGenerated(f) && !f.IsSpecialName))
                if (!documented.Contains($"F:{type.FullName}.{f.Name}")) missing.Add($"F:{type.FullName}.{f.Name}");
            foreach (var group in type.GetMethods(flags).Where(m => !m.IsSpecialName && !IsCompilerGenerated(m)).GroupBy(m => m.Name))
            {
                int have = documented.Count(d => d.StartsWith($"M:{type.FullName}.{group.Key}(", StringComparison.Ordinal) || d == $"M:{type.FullName}.{group.Key}");
                if (have < group.Count()) missing.Add($"M:{type.FullName}.{group.Key} ({have} of {group.Count()} overloads documented)");
            }
            int ctors = type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).Length;
            int documentedCtors = documented.Count(d => d.StartsWith($"M:{type.FullName}.#ctor", StringComparison.Ordinal));
            if (ctors > 0 && documentedCtors < 1 && !type.IsAbstract && !type.IsEnum) missing.Add($"M:{type.FullName}.#ctor");
        }
        log.WriteLine($"{types.Count} public types checked, {missing.Count} undocumented");
        Assert.True(missing.Count == 0, "undocumented public API:\n" + string.Join("\n", missing));
    }
}
