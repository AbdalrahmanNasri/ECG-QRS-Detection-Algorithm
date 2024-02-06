import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

# Reading the record
record = wfdb.rdrecord(r'database\100', sampto= 60000)
annotation = wfdb.rdann(r'database\100', 'atr', sampto= 60000)

# annotation.sample = list[annotation.sample]
data = record.p_signal
channel1 = data[:, 0]

signal_length = len(channel1)

print(pywt.wavelist(kind='discrete'))

wavelets = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7"]
waveletsLevel = 5

# Initializing the dataset array
ecgDataset = np.zeros(((waveletsLevel + 1) * len(wavelets), signal_length))
print(len(ecgDataset))


channel1 = channel1 - np.mean(channel1)
minValue = min(channel1)
maxValue = max(channel1)
channel1 = (channel1 - minValue) / (maxValue - minValue)
ecgDataset[0] = channel1

counterWaveletAssigment = 0
for i, wavelet in enumerate(wavelets):
    coeffs1 = pywt.wavedec(channel1, wavelet, level=waveletsLevel)

    time_values = [np.linspace(0, signal_length, len(coef), endpoint=False, dtype=int) for coef in coeffs1]

    print(len(time_values[:][0]), len(time_values[0][:]), max(time_values[:][0]), min(time_values[0][:]))

    for j, coef in enumerate(coeffs1):
        coef = coef - np.mean(coef)

        minValue = min(coef)
        maxValue = max(coef)
        coef = (coef - minValue) / (maxValue - minValue)
        ecgDataset[counterWaveletAssigment][time_values[j]] = coef
        counterWaveletAssigment = counterWaveletAssigment + 1

# Visualization of wavelet transforms as images
step_size = 300
iteration_number = 50 
imageNumber = int(signal_length / step_size)
print(len(ecgDataset[0, :]), len(ecgDataset[:, 0]))



data_directory = r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\project\QRS1_'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
file_path = os.path.join(data_directory)

cmap = 'viridis'
counter = 0

window_size = 10  

for qrs_index in annotation.sample:
    start = max(qrs_index - window_size, 0)
    end = min(qrs_index + window_size, signal_length)

    # Ensure the segment size is consistent
    if end - start != 2 * window_size:
        continue

    newMatrix = np.zeros(((waveletsLevel + 1) * len(wavelets), 2 * window_size), dtype=float)
    newMatrix[:, :] = 255 * ecgDataset[:, start:end]

    # The QRS index is at the center of newMatrix
    qrs_matrix_index = window_size  # QRS complex is in the middle of the window
    # newMatrix[:, qrs_matrix_index] = 255  # Overlay red vertical line at QRS location

    counter += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(newMatrix, cmap=cmap)

    # Save the image
    file_name = f'QRS_{counter}.png'
    file_path = os.path.join(data_directory, file_name)
    plt.savefig(file_path)
    plt.close()