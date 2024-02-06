import numpy as np
import matplotlib.pyplot as plt
import wfdb

record = wfdb.rdrecord(r'database\100', sampto=300)
annotation = wfdb.rdann(r'database\100', 'atr', sampto=300)

data = record.p_signal
ECG1 = data[:, 0]
ECG2 = data[:, 1]

times = np.arange(len(ECG1), dtype=float) / record.fs


def fourier_coefficients(t, signal, num_terms):
    coefficients = []
    for n in range(num_terms):
        an = 2 * np.trapz(signal * np.cos(2 * np.pi * n * t), t)
        bn = 2 * np.trapz(signal * np.sin(2 * np.pi * n * t), t)
        coefficients.append((an, bn))

    return coefficients


def reconstruct_signal(t, coefficients):
    signal = np.zeros_like(t)
    individual_terms = []

    for n, (an, bn) in enumerate(coefficients):
        term = an * np.cos(2 * np.pi * n * t) + bn * np.sin(2 * np.pi * n * t)
        individual_terms.append(term)

        signal += term

    return signal, individual_terms


def square_wave(t, frequency, amplitude=1.0):
    return amplitude * (1 + np.sign(np.sin(2 * np.pi * frequency * t))) / 2


# PARAMETERS
frequency = 5.0  # Adjust the frequency to make the square wave narrower
threshold = 0.15  # Adjust the threshold based on your signal characteristics
num_terms = 1000

# generate dataset
t = np.linspace(0, len(ECG1) / record.fs, len(ECG1), endpoint=False)

coefficients1 = fourier_coefficients(t, ECG1, num_terms)
coefficients2 = fourier_coefficients(t, ECG2, num_terms)

reconstructed_signal1, individual_terms1 = reconstruct_signal(t, coefficients1)
reconstructed_signal2, individual_terms2 = reconstruct_signal(t, coefficients2)

# Plot the original ECG signal and the reconstructed signal
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, ECG1, label='Original ECG1')
plt.title("Original ECG1 Signal")
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Overlay the square wave at annotation.sample positions
square_wave_overlay = square_wave(t, frequency)
square_wave_overlay *= (ECG1 > threshold)  # Multiply by ECG1 above the threshold
plt.scatter(annotation.sample / record.fs, [threshold] * len(annotation.sample),
            color='red', marker='o', label='Annotation Sample Points')
plt.plot(t, square_wave_overlay, label='Square Wave Overlay', linestyle='--', color='red')

plt.legend()

plt.subplot(3, 1, 2)
for n, term in enumerate(individual_terms1):
    plt.plot(t, term, label=f'Term {n + 1}')

plt.title('Sin and Cos Terms for ECG1')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(t, reconstructed_signal1, label='Reconstructed Signal ECG1')
plt.title("Reconstructed ECG1 Signal")
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.legend()
plt.tight_layout()
plt.show()