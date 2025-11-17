import numpy as np


def MFDFAm(timeseries: np.ndarray,
           poly_order: int = 1,
           q_arr: np.ndarray = 2
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # segment sizes, fluctuation, frac order
    """
    :param timeseries:
    :param poly_order: order of polynomial
    :param q_arr: order of fluctuation function
    :return: segment sizes, fluctuation function F(q, s), q array
    """

    # assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # fractal powers as floats
    q_arr = np.asarray_chkfinite(q_arr, dtype=float)

    # ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1.
    q_arr = q_arr[(q_arr < -.1) + (q_arr > .1)]

    # size of array
    N = timeseries.size

    # segment size as powers of 2
    max_pow = int(np.log2(N)) - 2
    min_pow = int(np.log2(poly_order + 2)) + 3  # TODO: is this correct?
    segment_sizes = np.logspace(min_pow, max_pow, max_pow - min_pow + 1, base=2, dtype=int)

    # detrended profile
    Y = np.cumsum(timeseries - np.mean(timeseries))

    # detrended fluctuation function F(s, q)
    F = np.zeros((len(segment_sizes), len(q_arr)))

    # iterate over window sizes
    for s, size in enumerate(segment_sizes):
        # number of segments
        N_s = N // size

        # reshape Y to segments
        if N % size == 0:
            segments = Y.reshape(N_s, size)
        else:
            segments = np.vstack((Y[N % size:].reshape(N_s, size), Y[:-(N % size)].reshape(N_s, size)))

        # X values at each segment
        X = np.arange(size)

        # detrended value at each segment f(nu, s)
        detrended = np.zeros(segments.shape[0])

        # iterate over segments and calculate detrended values
        for nu, segment in enumerate(segments):
            y = np.polyval(np.polyfit(X, segment, poly_order), X)  # polynomial fit to each segment
            detrended[nu] = np.var(segment - y)

        # F(s, q)
        for i, q in enumerate(q_arr):
            if q > 0:
                F[s, i] = np.power(np.mean(np.power(detrended, q / 2)), 1 / q)
            else:
                mask = detrended != 0
                F[s, i] = np.power(np.mean(np.power(detrended[mask], q / 2)), 1 / q)

    return segment_sizes, F.transpose(), q_arr


def generalised_Hurst_exponent(x: np.ndarray, y: np.ndarray) -> float:
    x_log = np.log(x)
    y_log = np.log(y)
    return np.polyfit(x_log, y_log, 1)[0]


def multifractal_scaling_exponent(h_q: np.ndarray, frac_ord: np.ndarray) -> np.ndarray:
    """
    τ(q) = q h(q) - 1
    """
    return frac_ord * h_q - 1


def singularity_strength(tau_q: np.ndarray, frac_ord: np.ndarray) -> np.ndarray:
    """
    α(q) = τ'(q)
    """
    return np.array([(tau_q[q + 1] - tau_q[q]) / (frac_ord[q + 1] - frac_ord[q])
                     for q in range(frac_ord.size - 1)])


def singularity_spectrum(alpha_q: np.ndarray, tau_q: np.ndarray, frac_ord: np.ndarray) -> np.ndarray:
    """
    D(α) = q α - τ(q)
    """
    return np.array([frac_ord[q] * alpha_q[q] - tau_q[q]
                     for q in range(alpha_q.size)])


def structure_function(xi: np.ndarray, q_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
