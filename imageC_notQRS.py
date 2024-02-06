import wfdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pywt
import pandas as pd
import os
import cv2
import sys

# Reading the record
record = wfdb.rdrecord(r'database\100', sampto=7500)
annotation = wfdb.rdann(r'database\100','atr',sampto=7500)

# ectrating signal
data = record.p_signal
channel1 = data[:, 0]

signal_length = len(channel1)

print(pywt.wavelist(kind='discrete'))


wavelets = ["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7"]
waveletsLevel = 5

# Initializing the dataset array
ecgDataset = np.zeros(((waveletsLevel+1)*len(wavelets), signal_length))
print(len(ecgDataset))

# store in array
channel1 = channel1 - np.mean(channel1)
minValue = min(channel1)
maxValue = max(channel1)
channel1 = (channel1 - minValue) / (maxValue - minValue)
ecgDataset[0] = channel1


counterWaveletAssigment = 0
for i, wavelet in enumerate(wavelets):
    coeffs1 = pywt.wavedec(channel1, wavelet, level=waveletsLevel)

    time_values = [np.linspace(0, signal_length, len(coef), endpoint = False, dtype=int) for coef in coeffs1]

    print (len(time_values[:][0]), len(time_values[0][:]),max(time_values[:][0]),min(time_values[0][:]))
    
    for j, coef in enumerate(coeffs1):
        coef = coef - np.mean(coef)

        minValue = min(coef)
        maxValue = max(coef)
        coef = (coef - minValue)/(maxValue - minValue)
        ecgDataset[counterWaveletAssigment][time_values[j]]= coef
        counterWaveletAssigment = counterWaveletAssigment + 1


# Visualization of wavelet transforms as images
step_size = 300
iteration_number = 20
imageNumber = int(signal_length/ step_size)
print(len(ecgDataset[0,:]),len(ecgDataset[:,0]))

# saving images
data_directory = r'F:\study__university\my_books_of_mekatro\unversity&myWork\Fourth\First\AI\project\project1\not_QRS3'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
file_path = os.path.join(data_directory)    

cmap = 'viridis'
counter = 1043

for i in range(0, signal_length - step_size, iteration_number):
    newMatrix = np.zeros(((waveletsLevel + 1) * len(wavelets), iteration_number), dtype=float)


    newMatrix[:, :] = 255 * ecgDataset[:, i:i + iteration_number]

    counter = counter + 1
   
    plt.figure(figsize=(8, 8))
    plt.imshow(newMatrix, cmap=cmap)  # Move this line before plt.savefig
    # Overlay red vertical lines at QRS complex locations
    qrs_indices = annotation.sample[annotation.sample < i + iteration_number]
    #for qrs_index in qrs_indices:
        #newMatrix[:, qrs_index] = 255
    # Save the image
    file_name = f'notQRS_{counter}.png'
    file_path = os.path.join(data_directory, file_name)
    
    plt.savefig(file_path)
    plt.close()  # Close the figure after saving