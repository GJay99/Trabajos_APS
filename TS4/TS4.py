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
print(f"Caso SNR = {SNR}")
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
omega_rect = freq[indice_maximo]*2*np.pi/fs
print("\nVentana rectangular:")
print(f"Media est_a1:{np.mean(a_1_rect)} y Varianza:{np.var(a_1_rect)}")
sesgo_a1 = np.mean(a_1_rect) - np.sqrt(2)
print(f"Sesgo:{sesgo_a1}")
print(f"Media est_omega:{np.mean(omega_rect)} y Varianza:{np.var(omega_rect)}")
sesgo_omega = np.mean(omega_rect-omega_uno)
print(f"Sesgo:{sesgo_omega}")

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
omega_flat = freq[indice_maximo]*2*np.pi/fs
print("\nVentana flat-top:")
print(f"Media est_a1:{np.mean(a_1_flat)} y Varianza:{np.var(a_1_flat)}")
sesgo_a1 = np.mean(a_1_flat) - np.sqrt(2)
print(f"Sesgo:{sesgo_a1}")
print(f"Media est_omega:{np.mean(omega_flat)} y Varianza:{np.var(omega_flat)}")
sesgo_omega = np.mean(omega_flat-omega_uno)
print(f"Sesgo:{sesgo_omega}")

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
omega_blackh = freq[indice_maximo]*2*np.pi/fs
print("\nVentana Blackman-Harris:")
print(f"Media est_a1:{np.mean(a_1_blackh)} y Varianza:{np.var(a_1_blackh)}")
print(f"Media est_omega:{np.mean(omega_blackh)} y Varianza:{np.var(omega_blackh)}")
sesgo_omega = np.mean(omega_flat-omega_uno)
print(f"Sesgo:{sesgo_omega}")

#%% Hamming 3dB

hamm = win.hamming(N)
x_hamm = x * hamm

freq, x_fft_abs = mi_fft(x_hamm)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(4)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_hamm = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_hamm = freq[indice_maximo]*2*np.pi/fs
print("\nVentana Hamming:")
print(f"Media est_a1:{np.mean(a_1_hamm)} y Varianza:{np.var(a_1_hamm)}")
print(f"Media est_omega:{np.mean(omega_hamm)} y Varianza:{np.var(omega_hamm)}")

#%% Histogramas 3dB

plt.figure(5)
plt.hist(a_1_rect, bins=20, color="red")
plt.hist(a_1_flat, bins=20, color="blue")
plt.hist(a_1_blackh, bins=20, color="gold", alpha=0.7)
plt.hist(a_1_hamm, bins=20, color="green", alpha=0.7)


plt.figure(6)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2)
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2)
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2)
plt.hist(omega_hamm, bins=20, color="green", histtype="step", linewidth=2)

#%% Para SNR=10

SNR = 10
print(f"\n\nCaso SNR = {SNR}")
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
omega_rect = freq[indice_maximo]*2*np.pi/fs
print("\nVentana rectangular:")
print(f"Media est_a1:{np.mean(a_1_rect)} y Varianza:{np.var(a_1_rect)}")
print(f"Media est_omega:{np.mean(omega_rect)} y Varianza:{np.var(omega_rect)}")

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
omega_flat = freq[indice_maximo]*2*np.pi/fs

print("\nVentana Flat-Top:")
print(f"Media est_a1:{np.mean(a_1_flat)} y Varianza:{np.var(a_1_flat)}")
print(f"Media est_omega:{np.mean(omega_flat)} y Varianza:{np.var(omega_flat)}")

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
omega_blackh = freq[indice_maximo]*2*np.pi/fs

print("\nVentana Blackman-Harris:")
print(f"Media est_a1:{np.mean(a_1_blackh)} y Varianza:{np.var(a_1_blackh)}")
print(f"Media est_omega:{np.mean(omega_blackh)} y Varianza:{np.var(omega_blackh)}")

#%% Hamming 10dB

hamm = win.hamming(N)
x_hamm = x * hamm

freq, x_fft_abs = mi_fft(x_hamm)
x_fft_abs_norm = 2*x_fft_abs/N
x_fft_abs_traspuesta = np.transpose(x_fft_abs_norm)
plt.figure(10)
plt.plot(freq,x_fft_abs_traspuesta,":.")
plt.xlim(240,260)
a_1_hamm = x_fft_abs_norm[:,freq==(omega_cero/(2*np.pi))*N]

indice_maximo = np.argmax(x_fft_abs_norm, axis=1)
omega_hamm = freq[indice_maximo]*2*np.pi/fs
print("\nVentana Hamming:")
print(f"Media est_a1:{np.mean(a_1_hamm)} y Varianza:{np.var(a_1_hamm)}")
print(f"Media est_omega:{np.mean(omega_hamm)} y Varianza:{np.var(omega_hamm)}")

#%% Histogramas 10dB

plt.figure(11)
plt.hist(a_1_rect, bins=20, color="red")
plt.hist(a_1_flat, bins=20, color="blue")
plt.hist(a_1_blackh, bins=20, color="gold", alpha=0.7)
plt.hist(a_1_hamm, bins=20, color="green", alpha=0.7)

plt.figure(12)
plt.hist(omega_rect, bins=20, color="red", histtype="step", linewidth=2)
plt.hist(omega_flat, bins=20, color="blue", histtype="step", linewidth=2)
plt.hist(omega_blackh, bins=20, color="gold", histtype="step", linewidth=2)
plt.hist(omega_hamm, bins=20, color="green", histtype="step", linewidth=2)