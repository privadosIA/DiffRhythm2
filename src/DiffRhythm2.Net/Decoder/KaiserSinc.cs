namespace DiffRhythm2.Decoder;

/// <summary>
/// The Kaiser-windowed sinc low-pass filter of the anti-aliased activation (alias-free-torch's <c>kaiser_sinc_filter1d</c>, as vendored in the
/// reference). The checkpoint stores these filters as buffers and the loader reads them; this computation exists so that a test can prove the
/// stored values are exactly what the formula gives, and so that a configuration with other ratios would not need new data.
/// </summary>
public static class KaiserSinc
{
    /// <summary>The filter taps (sum 1), computed in double precision. <paramref name="kernelSize"/> may be even (the case here, 12) or odd.</summary>
    public static double[] Filter(double cutoff, double halfWidth, int kernelSize)
    {
        bool even = kernelSize % 2 == 0;
        int halfSize = kernelSize / 2;

        double deltaF = 4 * halfWidth;
        double a = 2.285 * (halfSize - 1) * Math.PI * deltaF + 7.95;
        double beta = a > 50.0 ? 0.1102 * (a - 8.7)
                    : a >= 21.0 ? 0.5842 * Math.Pow(a - 21, 0.4) + 0.07886 * (a - 21.0)
                    : 0.0;

        var taps = new double[kernelSize];
        if (cutoff == 0) return taps;
        for (int n = 0; n < kernelSize; n++)
        {
            double time = even ? (n - halfSize) + 0.5 : n - halfSize;
            taps[n] = 2 * cutoff * KaiserWindow(n, kernelSize, beta) * Sinc(2 * cutoff * time);
        }
        double sum = taps.Sum();
        for (int n = 0; n < kernelSize; n++) taps[n] /= sum;
        return taps;
    }

    /// <summary>torch.kaiser_window(N, periodic=False, beta): I0(beta·sqrt(1 − ((n − M/2)/(M/2))²)) / I0(beta) with M = N − 1.</summary>
    private static double KaiserWindow(int n, int length, double beta)
    {
        if (length == 1) return 1.0;
        double m = length - 1;
        double r = (n - m / 2) / (m / 2);
        return BesselI0(beta * Math.Sqrt(Math.Max(0.0, 1 - r * r))) / BesselI0(beta);
    }

    private static double Sinc(double x) => x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);

    /// <summary>Modified Bessel function of the first kind, order 0, by its power series (converges fast for the arguments used here).</summary>
    private static double BesselI0(double x)
    {
        double term = 1.0, sum = 1.0, q = x * x / 4;
        for (int k = 1; k < 200; k++)
        {
            term *= q / ((double)k * k);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }
}
