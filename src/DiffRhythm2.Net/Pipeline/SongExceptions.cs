namespace DiffRhythm2.Pipeline;

/// <summary>Base of the errors the song pipeline raises on purpose: a request that cannot be served, missing files, a device that is not there.</summary>
public class SongException : Exception
{
    /// <summary>Creates the error with its message.</summary>
    public SongException(string message, Exception? inner = null) : base(message, inner) { }
}

/// <summary>A request that breaks a limit of the service. <see cref="Field"/> names the offending field (<c>lyrics</c>, <c>style</c>, <c>max_secs</c>, <c>cfg_strength</c>, <c>steps</c> or <c>seed</c>) and the message is the service's.</summary>
public sealed class SongRequestException : SongException
{
    /// <summary>The service's name for the field: <c>lyrics</c>, <c>style</c>, <c>max_secs</c>, <c>cfg_strength</c>, <c>steps</c> or <c>seed</c>.</summary>
    public string Field { get; }

    /// <summary>Creates the error for a field.</summary>
    public SongRequestException(string field, string message) : base(message) => Field = field;
}

/// <summary>The model directory lacks files; the message lists all of them.</summary>
public sealed class ModelDirectoryException : SongException
{
    /// <summary>The names of all the missing files.</summary>
    public IReadOnlyList<string> Missing { get; }

    /// <summary>Creates the error for a directory and the files it lacks.</summary>
    public ModelDirectoryException(string directory, IReadOnlyList<string> missing)
        : base($"the model directory {directory} lacks {missing.Count} file(s): {string.Join(", ", missing)}") => Missing = missing;
}

/// <summary>The requested device is not available on this machine.</summary>
public sealed class DeviceUnavailableException : SongException
{
    /// <summary>Creates the error with its message.</summary>
    public DeviceUnavailableException(string message) : base(message) { }
}
