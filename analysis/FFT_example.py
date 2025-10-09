import numpy as np
from scipy import signal
import numpy as np

def spectral_features(x, fs):
    N = len(x)
    X = np.fft.rfft(x * np.hanning(N))
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mag = np.abs(X)
    power = mag**2
    total_energy = power.sum()
    centroid = (freqs * mag).sum() / mag.sum()
    bw = np.sqrt(((freqs - centroid)**2 * mag).sum() / mag.sum())
    # roll-off 85%
    cum = np.cumsum(power)
    roll_idx = np.searchsorted(cum, 0.85 * total_energy)
    roll_freq = freqs[roll_idx]
    # entropy
    p = power / (power.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return dict(total_energy=total_energy, centroid=centroid, bandwidth=bw, rolloff=roll_freq, entropy=entropy)



if __name__ == '__main__':
    # x - 1D numpy array: отсчёты сигнала (например, ток в А)
    # fs - sampling frequency в Гц (например, 1000.0)
    N = len(x)
    # односторонний спектр для real-сигнала
    X = np.fft.rfft(x * np.hanning(N))   # окно Ханна чтобы уменьшить leakage
    freqs = np.fft.rfftfreq(N, d=1.0/fs) # соответствующие частоты в Гц
    amplitude = np.abs(X) / (N/2)        # нормированная амплитуда
    power = (np.abs(X)**2) / N           # простая оценка мощности

    # или оценить PSD более аккуратно (Welch)
    f_psd, Pxx = signal.welch(x, fs=fs, window='hann', nperseg=1024, noverlap=512)


# from scipy import signal
# f, Cxy = signal.coherence(x, y, fs=fs, nperseg=1024)