import numpy as np


def fgn(N: int, H: float) -> np.ndarray:
    """
    Generates fractional Gaussian noise with a Hurst index H in (0,1). If
    H = 1/2 this is simply Gaussian noise.

    Parameters
    ----------
    N: int
        Size of fractional Gaussian noise to generate.

    H: float
        Hurst exponent H in (0,1).

    Returns
    -------
    f: np.ndarray
        A array of size N of fractional Gaussian noise with a Hurst index H.
    """

    # Asserts
    assert isinstance(N, int), "Size must be an integer number"
    assert isinstance(H, float), "Hurst index must be a float in (0,1)"

    # Generate linspace
    k = np.linspace(0, N - 1, N)

    # Correlation function
    cor = 0.5 * (abs(k - 1) ** (2 * H)
                 - 2 * abs(k) ** (2 * H)
                 + abs(k + 1) ** (2 * H)
                 )

    # Eigenvalues of the correlation function
    eigenvals = \
        np.sqrt(
            np.fft.fft(
                np.real(
                    np.concatenate(
                        [cor[:], 0, cor[1:][::-1]], axis=None
                    )
                )
            )
        )

    # Two normal distributed noises to be convoluted
    gn = np.random.normal(0.0, 1.0, N)
    gn2 = np.random.normal(0.0, 1.0, N)

    # This is the Davies–Harte method
    w = np.concatenate(
        [
            (eigenvals[0] / np.sqrt(2 * N)) * gn[0],
            (eigenvals[1:N] / np.sqrt(4 * N)) * (gn[1:] + 1j * gn2[1:]),
            (eigenvals[N] / np.sqrt(2 * N)) * gn2[0],
            (eigenvals[N + 1:] / np.sqrt(4 * N))
            * (gn[1:][:: - 1] - 1j * gn2[1:][:: - 1])
        ], axis=None)

    # Perform fft. Only first N entry are useful
    f = np.fft.fft(w).real[:N] * ((1.0 / N) ** H)

    return f


def generate_fgn_noise(H: float, theta: float, sigma: float) -> np.ndarray:
    t_final = 10000
    dt = 0.001
    N = int(t_final / dt)
    X = np.zeros(N)
    dB = (t_final ** H) * fgn(N, H)
    for i in range(1, N):
        X[i] = X[i - 1] - theta * X[i - 1] * dt + sigma * dB[i - 1]
    return X
