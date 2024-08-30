#ensayo realizado el 26 de agosto en el portico, no se considero dano en ningun elemento estructural
import numpy as np
import matplotlib.pyplot as plt
from funciones import leer_data, signals_processing, modal_shape_plot
from pyoma import SSIcovStaDiag, SSIModEX 

import os
path=os.chdir(str('ensayos/ensayo01')) #This command changes directory
sensores=["sensor_5.txt","sensor_6.txt","sensor_7.txt","sensor_8.txt"]
data=leer_data(sensores,2)

#realizamos el pre-procesamiento 
fs=100 #frecuencia de muestreo
fmin=2 #frecuencia minima
fmax=30 #frecuencia maxima
data=signals_processing(data,fs,fmin,fmax)
np.set_printoptions(linewidth=1000)
data=data.T

# Run SSI
br = 35
SSIcov=SSIcovStaDiag(data, fs, br)

# Define list/array with the peaks identified from the plot picos aproximados segun el ploteo
FreQ = [4.01572, 12.323141, 20.3621488, 26.560132] # identified peaks

# Extract the modal properties
Res_SSIcov = SSIModEX(FreQ, SSIcov[1])
fn_ex = Res_SSIcov['Frequencies']
zeta_ex = Res_SSIcov['Damping']
ms_ex = Res_SSIcov['Mode Shapes'].real
ms_ex = ms_ex.T

#ploteando los modos
modal_shape_plot(fn_ex,ms_ex,"Results using SSI-COV ")
print('frecuencias obtenidas con el ssicov')
print(fn_ex)
print('modos obtenidos con el ssicov')
print(ms_ex)

# Calcular y visualizar el espectrograma para cada señal en la matriz de datos
import seaborn as sns

# Configurar los estilos de seaborn
sns.set()

plt.figure(figsize=(12, 8))

total_time = len(data) / fs

for i in range(data.shape[1]):
    plt.subplot(data.shape[1], 1, i + 1)
    plt.specgram(data[:, i], Fs=fs, NFFT=256, cmap='jet', scale='dB', vmin=-100, vmax=0)
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram - Sensor {}'.format(i + 1))
    plt.ylim(0, 30)
    plt.xlim(0, 50)

plt.tight_layout()
plt.show()






