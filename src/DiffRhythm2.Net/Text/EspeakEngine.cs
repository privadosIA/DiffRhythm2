using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace DiffRhythm2.Text;

/// <summary>Where to find espeak-ng. Null members fall back to the environment (<c>DIFFRHYTHM2_ESPEAK_LIB</c>, <c>DIFFRHYTHM2_ESPEAK_DATA</c>) and then to the platform's standard names and the library's own data path.</summary>
public sealed record EspeakOptions(string? LibraryPath = null, string? DataPath = null);

/// <summary>espeak-ng could not be loaded or initialized. The message lists what was tried and how to fix it.</summary>
public sealed class EspeakUnavailableException : G2pException
{
    public EspeakUnavailableException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>
/// espeak-ng through its C library, the way <c>phonemizer</c> reaches it: <c>espeak_Initialize</c> once, the <c>en-us</c> voice, and
/// <c>espeak_TextToPhonemes</c> clause by clause in IPA mode with <c>_</c> between phonemes. The library keeps global state (the engine and the
/// voice), so there is one engine per process and every call is serialized. The library is a runtime dependency found on the machine, never
/// bundled (espeak-ng is GPL-3).
/// </summary>
public sealed class EspeakEngine : IEspeakEngine
{
    public const string LibraryVariable = "DIFFRHYTHM2_ESPEAK_LIB";
    public const string DataVariable = "DIFFRHYTHM2_ESPEAK_DATA";

    private const int AudioOutputSynchronous = 0x02;
    private const int TextModeUtf8 = 1;
    private const string Voice = "en-us";
    private static readonly int PhonemeMode = ('_' << 8) | 0x02;   // IPA, with '_' as the phoneme separator (espeak-ng >= 1.49)

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int InitializeFn(int output, int bufferLength, IntPtr path, int options);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr InfoFn(IntPtr pathData);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate int SetVoiceByNameFn(IntPtr name);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr TextToPhonemesFn(ref IntPtr text, int textMode, int phonemeMode);

    private static readonly object Gate = new();
    private static EspeakEngine? shared;
    private static EspeakUnavailableException? sharedFailure;

    private readonly TextToPhonemesFn textToPhonemes;

    /// <summary>The library that was loaded.</summary>
    public string LibraryPath { get; }

    /// <summary>The version string the library reports (<c>espeak_Info</c>), such as <c>1.51</c>.</summary>
    public string VersionText { get; }

    /// <summary>The data directory the library reports.</summary>
    public string DataPath { get; }

    /// <summary>The version as numbers (major, minor, patch; missing parts are 0).</summary>
    public Version Version { get; }

    private EspeakEngine(string libraryPath, string versionText, string dataPath, TextToPhonemesFn textToPhonemes)
    {
        LibraryPath = libraryPath;
        VersionText = versionText;
        DataPath = dataPath;
        this.textToPhonemes = textToPhonemes;
        var numeric = new string(versionText.TakeWhile(c => char.IsDigit(c) || c == '.').ToArray()).Trim('.');
        Version = Version.TryParse(numeric.Count(c => c == '.') == 0 ? numeric + ".0" : numeric, out var v) ? v : new Version(0, 0);
    }

    /// <summary>The process-wide engine, configured from the environment. Throws <see cref="EspeakUnavailableException"/> if it cannot be loaded.</summary>
    public static EspeakEngine Shared
    {
        get
        {
            lock (Gate)
            {
                if (shared is not null) return shared;
                if (sharedFailure is not null) throw sharedFailure;
                try { return shared = Load(new EspeakOptions()); }
                catch (EspeakUnavailableException e) { sharedFailure = e; throw; }
            }
        }
    }

    /// <summary>Tries to get the shared engine; on failure returns false and says why.</summary>
    public static bool TryGetShared(out EspeakEngine? engine, out string reason)
    {
        try { engine = Shared; reason = ""; return true; }
        catch (EspeakUnavailableException e) { engine = null; reason = e.Message; return false; }
    }

    /// <summary>Loads and initializes the library. The library can be initialized only once per process: call this directly only when you do not want <see cref="Shared"/>.</summary>
    public static EspeakEngine Load(EspeakOptions options)
    {
        var tried = new List<string>();
        var explicitPath = options.LibraryPath ?? Environment.GetEnvironmentVariable(LibraryVariable);
        var candidates = explicitPath is { Length: > 0 } ? new[] { explicitPath } : StandardNames();

        IntPtr handle = IntPtr.Zero;
        string loadedFrom = "";
        foreach (var name in candidates)
        {
            tried.Add(name);
            if (NativeLibrary.TryLoad(name, out handle)) { loadedFrom = name; break; }
            handle = IntPtr.Zero;
        }
        if (handle == IntPtr.Zero)
            throw new EspeakUnavailableException(
                $"libespeak-ng could not be loaded (tried: {string.Join(", ", tried)}). Install espeak-ng, or set {LibraryVariable} to the library file " +
                $"(and {DataVariable} to its espeak-ng-data directory). The reference output was produced with espeak-ng 1.51.");

        var dataPath = options.DataPath ?? Environment.GetEnvironmentVariable(DataVariable);
        try
        {
            var initialize = Function<InitializeFn>(handle, "espeak_Initialize");
            var info = Function<InfoFn>(handle, "espeak_Info");
            var setVoice = Function<SetVoiceByNameFn>(handle, "espeak_SetVoiceByName");
            var textToPhonemes = Function<TextToPhonemesFn>(handle, "espeak_TextToPhonemes");

            // the library may keep the pointer to the data path, so it is never freed
            var pathPtr = dataPath is { Length: > 0 } ? Marshal.StringToCoTaskMemUTF8(dataPath) : IntPtr.Zero;
            int sampleRate = initialize(AudioOutputSynchronous, 0, pathPtr, 0);
            if (sampleRate <= 0) throw new EspeakUnavailableException($"espeak_Initialize failed (returned {sampleRate}) for {loadedFrom}; check {DataVariable} points to the espeak-ng-data directory");

            var dataOut = Marshal.AllocCoTaskMem(IntPtr.Size);
            Marshal.WriteIntPtr(dataOut, IntPtr.Zero);
            var versionText = Marshal.PtrToStringUTF8(info(dataOut)) ?? "";
            var reportedData = Marshal.PtrToStringUTF8(Marshal.ReadIntPtr(dataOut)) ?? "";
            Marshal.FreeCoTaskMem(dataOut);

            var voice = Marshal.StringToCoTaskMemUTF8(Voice);
            try
            {
                int status = setVoice(voice);
                if (status != 0) throw new EspeakUnavailableException($"the espeak voice \"{Voice}\" could not be selected (status {status}); is the data directory complete?");
            }
            finally { Marshal.FreeCoTaskMem(voice); }

            var engine = new EspeakEngine(loadedFrom, versionText, reportedData, textToPhonemes);
            if (engine.Version < new Version(1, 49)) throw new EspeakUnavailableException($"espeak-ng {versionText} is too old (1.49 or later is needed for IPA output with a phoneme separator)");
            return engine;
        }
        catch (Exception e) when (e is EntryPointNotFoundException or DllNotFoundException)
        {
            throw new EspeakUnavailableException($"{loadedFrom} does not look like libespeak-ng ({e.Message})", e);
        }
    }

    private static IEnumerable<string> StandardNames()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return new[] { "libespeak-ng.dll", "espeak-ng.dll" };
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return new[] { "libespeak-ng.dylib", "libespeak-ng.1.dylib" };
        return new[] { "libespeak-ng.so.1", "libespeak-ng.so" };
    }

    private static T Function<T>(IntPtr library, string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(library, name));

    /// <summary>
    /// The raw IPA espeak returns for a chunk of text: one answer per clause, joined by a space, phonemes separated by <c>_</c>, stress marks included.
    /// Calls are serialized because the library has global state.
    /// </summary>
    public string TextToPhonemes(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var buffer = Marshal.AllocCoTaskMem(bytes.Length + 1);
        try
        {
            Marshal.Copy(bytes, 0, buffer, bytes.Length);
            Marshal.WriteByte(buffer, bytes.Length, 0);
            var clauses = new List<string>();
            lock (Gate)
            {
                var cursor = buffer;   // the library advances this pointer clause by clause and sets it to NULL at the end
                while (cursor != IntPtr.Zero)
                {
                    var answer = textToPhonemes(ref cursor, TextModeUtf8, PhonemeMode);
                    var clause = answer == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(answer);
                    if (!string.IsNullOrEmpty(clause)) clauses.Add(clause);
                }
            }
            return string.Join(" ", clauses);
        }
        finally { Marshal.FreeCoTaskMem(buffer); }
    }

    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"espeak-ng {VersionText} ({LibraryPath})");
}
