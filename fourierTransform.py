import wfdb
import numpy as np
import matplotlib.pyplot as plt

def discrete_fourier_transform(signal, sampling_frequency):
    N = len(signal)
    k = np.arange(N)
    n = k.reshape(N, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    frequencies = k * sampling_frequency / N
    return frequencies, np.dot(W, signal)

def plot_discrete_fourier_transform(signal, sampling_frequency, title):
    frequencies, fourier_transform = discrete_fourier_transform(signal, sampling_frequency)

    positive_freq_indices = (frequencies >= 0) & (frequencies <= sampling_frequency / 2)
    plt.plot(frequencies[positive_freq_indices], np.abs(fourier_transform[positive_freq_indices]))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)


def find_max_amplitude_frequency(signal, sampling_frequency):
    frequencies, fourier_transform = discrete_fourier_transform(signal, sampling_frequency)
    
    # Get the positive frequencies and their corresponding amplitudes
    positive_freq_indices = (frequencies >= 0) & (frequencies <= sampling_frequency / 2)
    positive_frequencies = frequencies[positive_freq_indices]
    positive_amplitudes = np.abs(fourier_transform[positive_freq_indices])
    
    # Find the frequency with the maximum amplitude
    max_amplitude_frequency = positive_frequencies[np.argmax(positive_amplitudes)]
    
    return max_amplitude_frequency


record = wfdb.rdrecord(r'database\100', sampto=1000)
data = record.p_signal
ECG1 = data[:, 0]
timesRealECG1 = np.arange(len(ECG1)) / record.fs  # Adjusted the time calculation

plt.subplot(2, 1, 1)
plt.plot(timesRealECG1, ECG1, label='Real ECG1', color='blue')
plt.title('Real ECG1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

sampling_frequency = record.fs  # Adjusted the sampling frequency to match the record

plt.subplot(2, 1, 2)
plot_discrete_fourier_transform(ECG1, sampling_frequency, "DFT up to 1000 Hz")  # Adjusted the title
max_amplitude_frequency = find_max_amplitude_frequency(ECG1, sampling_frequency)
plt.axvline(x=max_amplitude_frequency, color='red', linestyle='--', label=f'Max Amplitude Frequency: {max_amplitude_frequency:.2f} Hz')
plt.legend()
plt.tight_layout()
plt.show()