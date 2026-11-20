using TorchSharp;
using static TorchSharp.torch;

namespace DiffRhythm2.Sampling;

/// <summary>
/// Explicit Euler over a fixed time grid, as torchdiffeq's fixed-grid solver does it for <c>method="euler"</c>:
/// <c>y(i+1) = y(i) + (t(i+1) - t(i)) * f(t(i), y(i))</c>, with the step computed from the grid tensor itself (so in its dtype).
/// A grid of N points gives N - 1 steps.
/// </summary>
public static class EulerSolver
{
    /// <summary>The uniform grid <c>linspace(0, 1, steps)</c> in the model's dtype (float32 by default), no time warping (the reference does not use sway sampling).</summary>
    public static Tensor TimeGrid(int steps, ScalarType dtype = ScalarType.Float32)
    {
        if (steps < 2) throw new ArgumentException($"at least 2 time points are needed to integrate (got {steps})");
        return linspace(0, 1, steps, dtype: dtype);
    }

    /// <param name="f">Receives the 0-dim time tensor and the state; returns the derivative.</param>
    /// <param name="y0">The state at the first grid point.</param>
    /// <param name="grid">The time points to integrate over, in increasing order (see <see cref="TimeGrid"/>).</param>
    /// <param name="states">When given, receives a copy of the state at every grid point (the first one is <paramref name="y0"/>).</param>
    public static Tensor Integrate(Func<Tensor, Tensor, Tensor> f, Tensor y0, Tensor grid, List<Tensor>? states = null)
    {
        var y = y0;
        states?.Add(y0.detach().clone());
        for (int i = 0; i + 1 < grid.shape[0]; i++)
        {
            Tensor next;
            using (var scope = NewDisposeScope())
            {
                var t0 = grid[i];
                var dt = grid[i + 1] - t0;
                next = (y + dt * f(t0, y)).MoveToOuterDisposeScope();
            }
            if (!ReferenceEquals(y, y0)) y.Dispose();
            y = next;
            states?.Add(y.detach().clone());                    // outside the scope above, so it survives
        }
        return y;
    }
}
