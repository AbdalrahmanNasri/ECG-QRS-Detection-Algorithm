import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Reading the record and annotation
record = wfdb.rdrecord(r'database\100', sampto= 4000)
annotation = wfdb.rdann(r'database\100', 'atr', sampto= 4000)

# Extracting the ECG signals
data = record.p_signal
ECG1 = data[:, 0]
ECG2 = data[:, 1]

# Wavelet transform parameters
wavelet = 'db4'
level = 3
signal_length = len(ECG1)

# Decomposition
coeffs1 = pywt.wavedec(ECG1, wavelet, level=level)
coeffs2 = pywt.wavedec(ECG2, wavelet, level=level)

# Time vectors for plotting
timesRealECG = np.arange(len(ECG1), dtype=float)

# QRS complex locations from annotations
qrs_indices = annotation.sample

# Square wave parameters
square_wave_duration = 20  # Adjust as needed
square_wave_amplitude = 1  # Adjust as needed
square_wave_base = -0.5  # Adjust as needed

# Initialize square wave signal
square_wave_signal = np.full_like(ECG1, square_wave_base)

# Create the square wave signal
for index in qrs_indices:
    start = max(0, index - square_wave_duration // 2)
    end = min(len(ECG1), index + square_wave_duration // 2)
    square_wave_signal[start:end] = square_wave_amplitude

# Reinitialize the square wave signal to remove the first square
square_wave_signal[:qrs_indices[1] - 10] = square_wave_base


# Plotting
plt.figure(figsize=(12, 6))

# ECG1 with square wave overlay
plt.subplot(2, 1, 1)
plt.plot(timesRealECG, ECG1, label='Real ECG1')
plt.plot(timesRealECG, square_wave_signal, 'r-', label='QRS Square Wave')
plt.title('ECG1 with QRS Square Wave Overlay')

plt.subplot(2,1,2)
plt.plot(timesRealECG, ECG2, label='Real ECG1')
plt.plot(timesRealECG, square_wave_signal, 'r-', label='QRS Square Wave')

plt.title('ECG2 with QRS Square Wave Overlay')
plt.legend()

# Repeat for ECG2 if desired

plt.tight_layout()
plt.show()
