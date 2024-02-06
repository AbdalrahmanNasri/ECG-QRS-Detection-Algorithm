import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pywt

record = wfdb.rdrecord(r'database\100', sampto=4000)
annotation = wfdb.rdann(r'database\100','atr',sampto=4000)

data = record.p_signal
ECG1 = data[:,0]
ECG2 = data[:,1]

wavelet = 'db4'
level = 3 # 360 - 180, 180 - 90, 90 - 45, 45 - 0

signal_length = len(ECG1)

coeffs1 = pywt.wavedec(ECG1,wavelet,level=level)
coeffs2 = pywt.wavedec(ECG2,wavelet,level=level)

time_values = [np.linspace(0,signal_length, len(coef), endpoint=False) for coef in coeffs1]
timesRealECG = np.arange(len(ECG1),dtype=float)

# visualize the wavelet coefficient

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(timesRealECG,ECG1, label='Real ECG1')

for i, coef in enumerate(coeffs1):
    plt.plot(time_values[i],coef + i*2,label=f'Level{i}')

plt.title('Wavelet Coeffients - ECG1')
plt.legend()

plt.subplot(2,1,2)
plt.plot(timesRealECG,ECG2, label='Real ECG1')

for i, coef in enumerate(coeffs2):
    plt.plot(time_values[i],coef + i*2,label=f'Level{i}')

plt.title('Wavelet Coeffients - ECG2')
plt.legend()

plt.tight_layout()
plt.show()