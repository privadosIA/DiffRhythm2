using Xunit;

// The libtorch RNG is global: tests that seed it (manual_seed) cannot run in parallel with others that use it.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
