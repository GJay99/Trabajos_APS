import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio

#%%ECG con ruido
fs_ecg = 1000 # Hz

sio.whosmat('files/ECG_TP4.mat')
mat_struct = sio.loadmat('./files/ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.squeeze()
N = len(ecg_one_lead)
t = np.arange(N)/fs_ecg

plt.figure(0)
plt.plot(t[5000:12000],ecg_one_lead[5000:12000])
plt.grid()
plt.title("7 ciclos de ECG con ruido")
plt.xlabel("tiempo [s]")
plt.ylabel("ECG []")
plt.axhline(0,color="black")

K = np.arange(25, 40, 2)
L = N // K

ecg = ecg_one_lead.astype(np.float64)
potx = np.mean(ecg**2)

for l in L:
    freq, PSD = sig.welch(ecg_one_lead,fs_ecg,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2f}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=39
L = N//K
freq, PSD = sig.welch(ecg,fs_ecg,"hann",L)

ref = max(PSD)/2
indices = np.where(PSD >= ref)[0]
BW_3 = freq[indices[-1]]-freq[indices[0]]

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice = np.searchsorted(potencia, 0.99 * potencia[-1])
BW = freq[indice]

print(f"Ancho de banda -3dB: {BW_3}")
print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(1, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma de Welch de ECG con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([0,BW+1])
plt.grid()
plt.title("Periodograma de Welch de ECG con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)

#%% ECG sin ruido
fs_ecg = 1000 # Hz

ecg_one_lead = np.load('files/ecg_sin_ruido.npy')
ecg_one_lead = ecg_one_lead.squeeze()
N = len(ecg_one_lead)
t = np.arange(N)/fs_ecg

plt.figure(2)

plt.plot(t[5000:12000],ecg_one_lead[5000:12000])
plt.grid()
plt.title("7 ciclos de ECG sin ruido")
plt.xlabel("tiempo [s]")
plt.ylabel("ECG []")
plt.axhline(0,color="black")

K = np.arange(25, 40, 2)
L = N // K

ecg = ecg_one_lead.astype(np.float64)
potx = np.mean(ecg**2)
for l in L:
    freq, PSD = sig.welch(ecg_one_lead,fs_ecg,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2f}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=37
L = N//K
freq, PSD = sig.welch(ecg,fs_ecg,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice = np.searchsorted(potencia, 0.99 * potencia[-1])
BW = freq[indice]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(3, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma de Welch de ECG sin ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([0,BW+5])
plt.grid()
plt.title("Periodograma de Welch de ECG sin ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)

#%% PPG con ruido
fs_ppg = 400 # Hz

ppg = np.genfromtxt('files/PPG.csv', delimiter=',', skip_header=1)
N = len(ppg)

t = np.arange(N)/fs_ppg

plt.figure(4)
plt.plot(t,ppg)
plt.grid()
plt.title("PPG con ruido")
plt.xlabel("tiempo [s]")
plt.ylabel("PPG []")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

K = np.arange(30, 50, 2)
L = N // K

ppg_para_pot = ppg.astype(np.float64)
potx = np.mean(ppg_para_pot**2)

for l in L:
    freq, PSD = sig.welch(ppg,fs_ppg,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2f}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=44
L = N//K
freq, PSD = sig.welch(ppg,fs_ppg,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice = np.searchsorted(potencia, 0.99 * potencia[-1])
BW = freq[indice]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(5, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma de Welch de PPG con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([0,8])
plt.grid()
plt.title("Periodograma de Welch de PGG con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)
#%% PPG sin ruido
fs_ppg = 400 # Hz

ppg = np.load('files/ppg_sin_ruido.npy')
N = len(ppg)

t = np.arange(N)/fs_ppg

plt.figure(6)
plt.plot(ppg)
plt.grid()
plt.title("PPG sin ruido")
plt.xlabel("tiempo [s]")
plt.ylabel("PPG []")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

K = np.arange(30, 50, 2)
L = N // K

ppg_para_pot = ppg.astype(np.float64)
potx = np.mean(ppg_para_pot**2)

for l in L:
    freq, PSD = sig.welch(ppg,fs_ppg,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2f}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=46
L = N//K
freq, PSD = sig.welch(ppg,fs_ppg,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice_low = np.searchsorted(potencia, 0.005 * potencia[-1])
indice_up = np.searchsorted(potencia, 0.995 * potencia[-1])
BW = freq[indice_up] - freq[indice_low]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(7, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma de Welch de PPG sin ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(indice_up,color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([0,BW+1])
plt.grid()
plt.title("Periodograma de Welch de PGG sin ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(indice_up,color="green",linewidth=1)

#%% Prueba
fs_prueba, wav_prueba = sio.wavfile.read('files/prueba psd.wav')

N = len(wav_prueba)
t = np.arange(N)/fs_prueba

plt.figure(8)
plt.plot(t,wav_prueba)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.title('Audio "Esta es una prueba..."')
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud []")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

K = np.arange(30, 50, 2)
L = N // K

wav_prueba_para_pot = wav_prueba.astype(np.float64)
potx = np.mean(wav_prueba_para_pot**2)

for l in L:
    freq, PSD = sig.welch(wav_prueba,fs_prueba,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2g}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=34
L = N//K
freq, PSD = sig.welch(wav_prueba,fs_prueba,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice_low = np.searchsorted(potencia, 0.005 * potencia[-1])
indice_up = np.searchsorted(potencia, 0.995 * potencia[-1])
BW = freq[indice_up] - freq[indice_low]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(9, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma del audio de prueba")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([-10,BW+100])
plt.grid()
plt.title("Periodograma del audio de prueba")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(BW,color="green",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

#%% Silbido
fs_silbido, wav_silbido = sio.wavfile.read('files/silbido.wav')

N = len(wav_silbido)
t = np.arange(N)/fs_silbido

plt.figure(10)
plt.plot(t,wav_silbido)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.title('Silbido')
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud []")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

K = np.arange(30, 50, 2)
L = N // K

wav_silbido_para_pot = wav_silbido.astype(np.float64)
potx = np.mean(wav_silbido_para_pot**2)

for l in L:
    freq, PSD = sig.welch(wav_silbido,fs_silbido,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2g}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=36
L = N//K
freq, PSD = sig.welch(wav_silbido,fs_silbido,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice_low = np.searchsorted(potencia, 0.005 * potencia[-1])
indice_up = np.searchsorted(potencia, 0.995 * potencia[-1])
BW = freq[indice_up] - freq[indice_low]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(11, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma del silbido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([freq[indice_low]-200,freq[indice_up]+200])
plt.grid()
plt.title("Periodograma del silbido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

#%% la cucaracha
fs_cucaracha, cucaracha_data = sio.wavfile.read('files/la cucaracha.wav')

N = len(cucaracha_data)
t = np.arange(N)/fs_cucaracha

plt.figure(12)
plt.plot(t,cucaracha_data)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.title('Silbido la cucaracha')
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud []")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

K = np.arange(30, 50, 2)
L = N // K

cucaracha_data_para_pot = cucaracha_data.astype(np.float64)
potx = np.mean(cucaracha_data_para_pot**2)

for l in L:
    freq, PSD = sig.welch(cucaracha_data,fs_cucaracha,"hann",l,detrend=False)
    res = freq[1] - freq[0]
    pot_welch = np.sum(PSD) * (freq[1] - freq[0])
    sesgo = 100 * (pot_welch - potx)/potx
    print(f"K={N//l}, Potencia={pot_welch:.2g}, sesgo (%)={sesgo:.4f}%, var = {np.var(PSD)}")

K=44
L = N//K
freq, PSD = sig.welch(cucaracha_data,fs_cucaracha,"hann",L)

res = freq[1] - freq[0]
potencia = np.cumsum(PSD) * res
indice_low = np.searchsorted(potencia, 0.005 * potencia[-1])
indice_up = np.searchsorted(potencia, 0.995 * potencia[-1])
BW = freq[indice_up] - freq[indice_low]

print(f"Ancho de banda 99%: {BW:.2f} Hz")

plt.figure(13, figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(freq,PSD,":.")
plt.grid()
plt.title("Periodograma de la cucaracha")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

plt.subplot(1, 2, 2)
plt.plot(freq,PSD,":.")
plt.xlim([freq[indice_low]-200,freq[indice_up]+200])
plt.grid()
plt.title("Periodograma de la cucaracha")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [$U^2/Hz$]")
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(freq[indice_low],color="green",linewidth=1)
plt.axvline(freq[indice_up],color="green",linewidth=1)

import sounddevice as sd

sd.play(cucaracha_data, fs_cucaracha) 
