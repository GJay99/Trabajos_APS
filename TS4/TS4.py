# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:24:09 2026

@author: ECyT
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win 

N=1000
fs=1000

def mi_fft(x, fs=fs):
    x_fft = np.fft.fft(x, axis=1)
    
    x_fft = x_fft[:,:N//2]
    
    freq = np.fft.fftfreq(N, 1/fs)[:N//2]
    x_fft_abs = np.abs(x_fft)
    
    return freq, x_fft_abs

omega_cero = np.pi/2
f_r = np.random.uniform(-2,2,size=200)

omega_uno = omega_cero + f_r * 2 * np.pi/N
nn = np.arange(0,N,1)
s = np.sqrt(2)*np.sin(np.outer(omega_uno,nn))
Ps = np.var(s)
SNR = 3
sigma = np.sqrt(Ps/np.power(10,SNR/10)) 

n_a = np.random.normal(scale=sigma,size=s.shape)
    
x = s + n_a

#%% Rectangular 3dB
freq, x_fft_abs = mi_fft(x)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(1)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_rect = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]


indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_rect = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp rectang:", np.mean(a_1_rect))
print("Media estimador de fre rectang:",np.mean(omega_rect))
"""
espa_1_rect = np.mean(a_1_rect)
sa_1_rect=espa_1_rect-np.sqrt(2)
print(sa_1_rect)
espa_1_rect = np.var(a_1_rect)
sa_1_rect=espa_1_rect-np.sqrt(2)
print(sa_1_rect)
#pot = np.mean(x**2)
#print("Potencia blackmanh:", pot)

#par = np.sum(x_fft_abs**2, axis=1) / (2)
#par = np.mean(par)

#print("Potencia Parseval:", par)
"""
#%% Flattop 3dB

flat=win.flattop(N)
x_flat=flat*x

freq, x_fft_abs = mi_fft(x_flat)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(2)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_flat = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]


indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_flat = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp flattop:",np.mean(a_1_flat))
print("Media estimador de fre flattop:",np.mean(omega_flat))

#%% Blackmanharris 3dB

blackh=win.blackmanharris(N)
x_blackh=x*blackh

freq, x_fft_abs = mi_fft(x_blackh)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(3)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_blackh = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_blackh = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp blackmanh:",np.mean(a_1_blackh))
print("Media estimador de fre blackmanh:",np.mean(omega_blackh))

#%% Histogramas 3dB

plt.figure(5)
plt.hist(a_1_rect, bins=20, color="red")
plt.hist(a_1_flat, bins=20, color="blue")
plt.hist(a_1_blackh, bins=20, color="gold", alpha=0.7)

plt.figure(6)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2)
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2)
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2)

"""
plt.figure(5)

plt.hist(a_1_rect, bins=20, alpha=0.5, color="red", label="Rectangular")
plt.hist(a_1_flat, bins=20, alpha=0.5, color="blue", label="Flattop")
plt.hist(a_1_blackh, bins=20, alpha=0.5, color="gold", label="Blackman-Harris")

plt.xlabel("Amplitud")
plt.ylabel("Cantidad de muestras")
plt.title("Histograma de amplitudes - SNR = 3 dB")
plt.legend()
plt.grid(alpha=0.2)


plt.figure(6)

plt.hist(omega_rect, bins=20, alpha=0.5, color="red", label="Rectangular")
plt.hist(omega_flat, bins=20, alpha=0.5, color="blue", label="Flattop")
plt.hist(omega_blackh, bins=20, alpha=0.5, color="gold", label="Blackman-Harris")

plt.xlabel("Frecuencia")
plt.ylabel("Cantidad de muestras")
plt.title("Histograma de frecuencias - SNR = 3 dB")
plt.legend()
plt.grid(alpha=0.2)

plt.show()
"""




#%% Para SNR=10

SNR = 3
sigma = np.sqrt(Ps/np.power(10,SNR/10)) 

n_a = np.random.normal(scale=sigma,size=s.shape)
    
x = s + n_a
#%% Rectangular 10dB

freq, x_fft_abs = mi_fft(x)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(7)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_rect = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_rect = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp rectang:", np.mean(a_1_rect))
print("Media estimador de fre rectang:",np.mean(omega_rect))

#%% Flattop 10dB

flat=win.flattop(N)
x_flat=flat*x

freq, x_fft_abs = mi_fft(x_flat)
x_fft_abs = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs)
plt.figure(8)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_flat = x_fft_abs[:,freq==(omega_cero/(2*np.pi))*N]


indice_maximo = np.argmax(x_fft_abs, axis=1)
omega_flat = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp flattop:",np.mean(a_1_flat))
print("Media estimador de fre flattop:",np.mean(omega_flat))

#%% Blackmanharris 10dB

blackh=win.blackmanharris(N)
x_blackh=x*blackh

freq, x_fft_abs = mi_fft(x_blackh)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(9)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_blackh = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_blackh = freq[indice_maximo]*2*np.pi/N
print("Media estimador de amp blackmanh:",np.mean(a_1_blackh))
print("Media estimador de fre blackmanh:",np.mean(omega_blackh))

#%% Histogramas 10dB

plt.figure(11)
plt.hist(a_1_rect, bins=20, color="red")
plt.hist(a_1_flat, bins=20, color="blue")
plt.hist(a_1_blackh, bins=20, color="gold", alpha=0.7)

plt.figure(12)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2)
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2)
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2)