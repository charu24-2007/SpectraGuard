"""
SignalProcessingService: DSP operations on IQ data.
All heavy computation stays in Python/backend.
"""
import numpy as np


def compute_spectrum(iq: np.ndarray) -> dict:
    """
    Compute FFT-based frequency spectrum from IQ sample.
    Input: iq shape (1024, 2), col0=I col1=Q
    Returns normalized frequency axis and magnitude in dB.
    """
    I = iq[:, 0].astype(np.float64)
    Q = iq[:, 1].astype(np.float64)
    complex_signal = I + 1j * Q

    N = len(complex_signal)
    fft_vals = np.fft.fftshift(np.fft.fft(complex_signal))
    magnitude_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)

    # Normalized frequency axis: -0.5 to +0.5
    freq_axis = np.fft.fftshift(np.fft.fftfreq(N)).tolist()

    return {
        "frequency": freq_axis,
        "magnitude": magnitude_db.tolist(),
        "freq_label": "Normalized Frequency (cycles/sample)",
        "mag_label": "Magnitude (dB)",
    }


def compute_spectrogram(iq: np.ndarray, nperseg: int = 64, noverlap: int = 32) -> dict:
    """
    Compute STFT-based spectrogram from IQ sample.
    Input: iq shape (1024, 2), col0=I col1=Q
    Returns time axis, frequency axis, and power matrix (dB).
    """
    from scipy.signal import spectrogram as scipy_spectrogram

    I = iq[:, 0].astype(np.float64)
    Q = iq[:, 1].astype(np.float64)
    complex_signal = I + 1j * Q

    # scipy's spectrogram for complex signals
    freqs, times, Sxx = scipy_spectrogram(
        complex_signal,
        fs=1.0,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        window="hann",
    )

    # Shift so DC is in center
    freqs = np.fft.fftshift(freqs)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # Power in dB
    power_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

    # Downsample for frontend performance (max 64 freq bins × 64 time steps)
    # Sxx shape: (nperseg, n_time_steps)
    return {
        "time": times.tolist(),
        "frequency": freqs.tolist(),
        "power": power_db.tolist(),   # list of lists: [freq_bins][time_steps]
        "freq_label": "Normalized Frequency",
        "time_label": "Time (sample window)",
    }
