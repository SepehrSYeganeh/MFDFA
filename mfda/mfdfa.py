import numpy as np


def MFDFAm(timeseries: np.ndarray,
           order: int = 1,
           q_arr: np.ndarray = 2
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # segment sizes, fluctuation, frac order

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # Fractal powers as floats
    q_arr = np.asarray_chkfinite(q_arr, dtype=float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q_arr = q_arr[(q_arr < -.1) + (q_arr > .1)]

    # Size of array
    N = timeseries.size

    # Window size as powers of 2
    max_pow = int(np.log2(N)) - 2
    min_pow = int(np.log2(order + 2)) + 1
    segment_sizes = np.logspace(min_pow, max_pow, max_pow - min_pow + 1, base=2, dtype=int)

    # "Profile" of the series
    Y = np.cumsum(timeseries - np.mean(timeseries))

    # Detrended Fluctuation
    F = np.zeros((len(segment_sizes), len(q_arr)))

    # Iterate over window sizes
    for s, size in enumerate(segment_sizes):
        # Number of segments of size s
        N_s = int(N / size)

        # Reshape Y to segments
        segments = Y[N % size:].reshape(N_s, size)

        # X values at each segment
        X = np.arange(size)

        # Detrended value at each segment
        detrended = np.zeros(N_s)

        # Iterate over segments and calculate detrended values
        for nu, segment in enumerate(segments):
            # Perform a polynomial fit to each window
            y = np.polyval(np.polyfit(X, segment, order), X)
            detrended[nu] = np.mean(np.fabs(segment - y))

        # F(s, q)
        for i, q in enumerate(q_arr):
            if q > 0:
                F[s, i] = np.power(np.mean(np.power(detrended, q)), 1 / q)
            else:
                mask = detrended != 0
                F[s, i] = np.power(np.mean(np.power(detrended[mask], q)), 1 / q)

    # Return: segment sizes, F(q, s), fractal orders
    return segment_sizes, F.transpose(), q_arr


def random_shuffle_surrogate():
    pass


def rank_wise_surrogate():
    pass


def random_phase_surrogate():
    pass


def structure_function(xi: NDArray, q_arr: NDArray) -> tuple[NDArray, NDArray]:
    # TODO: complete
    # size of time series
    T = xi.size

    # list of lag
    min_pow = 0
    max_pow = np.log2(T).astype(int) - 1
    tau_arr = np.logspace(min_pow, max_pow, max_pow - min_pow + 1, base=2, dtype=int)

    # structure function
    S = np.array([
        [np.mean(np.power(np.fabs(xi[tau:] - xi[:-tau]), q)) for tau in tau_arr]
        for q in q_arr
    ])

    return tau_arr, S


def generalised_Hurst_exponent(x: NDArray, y: NDArray) -> float:
    x_log = np.log(x)
    y_log = np.log(y)
    return np.polyfit(x_log, y_log, 1)[0]


def multifractal_scaling_exponent(h_q: NDArray, frac_ord: NDArray) -> NDArray:
    """
    τ(q) = q h(q) - 1
    """
    return frac_ord * h_q - 1


def singularity_strength(tau_q: NDArray, frac_ord: NDArray) -> NDArray:
    """
    α(q) = τ'(q)
    """
    return np.array([(tau_q[q + 1] - tau_q[q]) / (frac_ord[q + 1] - frac_ord[q])
                     for q in range(frac_ord.size - 1)])


def singularity_spectrum(alpha_q: NDArray, tau_q: NDArray, frac_ord: NDArray) -> NDArray:
    """
    D(α) = q α - τ(q)
    """
    return np.array([frac_ord[q] * alpha_q[q] - tau_q[q]
                     for q in range(alpha_q.size)])

# TODO: white noise
