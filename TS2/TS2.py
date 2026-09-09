#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:01:09 2026

@author: gonza
"""

#%% imports
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.stats import kstest

#%% definiciones
fs=1000
N=1000

#%% funciones
def mi_funcion_seno(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    tt=np.zeros(nn)
    xx=np.zeros(nn)
    for n in np.arange(0,nn,1):
        tt[n]= n * (1/fs)
        xx[n]= vmax * np.sin(2*np.pi*ff * tt[n] + ph) + dc
    return tt, xx

def mi_fft(x,ref=0):
    x_fft = fft(x)
    x_fft = x_fft[:N//2]

    freq = fftfreq(N,1/fs)
    freq = freq[:N//2]

    x_fft_abs = np.abs(x_fft)
    
    return freq, x_fft_abs,

#%% a
t,s = mi_funcion_seno(vmax = np.sqrt(2),ff=fs/N)

n_bits = 4
k_n = 1
Vfs = 2
q = (2*Vfs)/(np.pow(2,n_bits))
P_q = q**2/12
P_n = k_n * P_q
sigma = np.sqrt(P_n)
n = np.random.normal(scale=sigma,size=len(t))

sn = s+n

sn_limitada = np.clip(sn, -Vfs, Vfs)
sn_cuantizada = np.round(sn_limitada/q)*q

plt.figure(1)
plt.plot(t,sn_cuantizada,color = "blue",label="$Sn_q$ = $Q_{B,V_f}$ (ADC out)")
plt.plot(t,sn,":.",color = "green",label="$S_n$ = s+n (ADC in)")
plt.plot(t,s,linestyle="--",marker=" ",color = "yellow",label="S (analog)")
plt.title(f"Señal muestreada por un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.3f}V")
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.legend()

freq, sn_fft_abs = mi_fft(sn)
ref = np.max(sn_fft_abs)

sn_fft_dB = 20 * np.log10(sn_fft_abs/ref)

freq, sn_cuantizada_fft_abs = mi_fft(sn_cuantizada)
sn_cuantizada_fft_dB = 20 * np.log10(sn_cuantizada_fft_abs/ref)

freq, n_fft_abs = mi_fft(n)
n_fft_dB = 20 * np.log10(n_fft_abs/ref)

n_q = sn_cuantizada - sn
freq, n_q_fft_abs = mi_fft(n_q)
n_q_fft_dB = 20 * np.log10(n_q_fft_abs/ref)

plt.figure(2)
plt.plot(freq,sn_cuantizada_fft_dB,color="blue",label="$Sn_q$ = $Q_{B,V_f}$ (ADC out)")
plt.plot(freq,n_fft_dB,linestyle ="--", color="red")
plt.axhline(np.mean(n_fft_dB), color="red", linestyle="--", label=f"n: {np.mean(n_fft_dB):.2f}dB (piso analog.)")
plt.plot(freq,n_q_fft_dB,linestyle ="--",color="cyan")
plt.axhline(np.mean(n_q_fft_dB), color="cyan", linestyle='--', label=f"n_q: {np.mean(n_q_fft_dB):.2f}dB (piso digital)")

plt.legend()
plt.title(f"FFT de Señal muestreada por un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.3f}V")
plt.ylabel("|X|")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(3)
plt.hist(n_q)
plt.title(f"Ruido de cuantización para un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.3f}V")
plt.ylabel("Cantidad de muestras")
plt.xlabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(-q/2, linestyle="--",color="red")
plt.axvline(q/2, linestyle="--",color="red")
plt.axhline(100, linestyle="--",color="red")


correlacion = np.correlate(n_q,n_q,"full")/N
lags = np.arange(-(N-1), N)

plt.figure(4)
plt.plot(lags, correlacion,":.")
plt.xlabel("Retardo [muestras]")
plt.ylabel("Autocorrelación")

a = -q/2
b = q/2

resultado = kstest(n_q, 'uniform', args=(a, b-a))

alpha = 0.05
if resultado.pvalue < alpha:
    print("La señal no es compatible con una distribución uniforme.")
else:
    print("No hay evidencia suficiente para afirmar que la señal no sea uniforme.")

#%% b
n_bits = 16
k_n = 10
q = (2*Vfs)/(np.pow(2,n_bits))
P_q = q**2/12
P_n = k_n * P_q
sigma = np.sqrt(P_n)
n = np.random.normal(scale=sigma,size=len(t))

sn = s+n

sn_limitada = np.clip(sn, -Vfs, Vfs)
sn_cuantizada = np.round(sn_limitada/q)*q

plt.figure(5)
plt.plot(t,sn," .",color = "green",label="$S_n$ = s+n (ADC in)")
plt.plot(t,sn_cuantizada,color = "blue",label="$Sn_q$ = $Q_{B,V_f}$ (ADC out)")
plt.title(f"Señal muestreada por un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.2e}V")
plt.xlabel("tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.legend()

freq, sn_fft_abs = mi_fft(sn)
ref = np.max(sn_fft_abs)

sn_fft_dB = 20 * np.log10(sn_fft_abs/ref)

freq, sn_cuantizada_fft_abs = mi_fft(sn_cuantizada)
sn_cuantizada_fft_dB = 20 * np.log10(sn_cuantizada_fft_abs/ref)

freq, n_fft_abs = mi_fft(n)
n_fft_dB = 20 * np.log10(n_fft_abs/ref)

n_q = sn_cuantizada - sn
freq, n_q_fft_abs = mi_fft(n_q)
n_q_fft_dB = 20 * np.log10(n_q_fft_abs/ref)

plt.figure(6)
plt.plot(freq,sn_cuantizada_fft_dB,color="blue",label="$Sn_q$ = $Q_{B,V_f}$ (ADC out)")
plt.plot(freq,n_fft_dB,linestyle ="--", color="red")
plt.axhline(np.mean(n_fft_dB), color="red", linestyle="--", label=f"n: {np.mean(n_fft_dB):.2f}dB (piso analog.)")
plt.plot(freq,n_q_fft_dB,linestyle ="--",color="cyan")
plt.axhline(np.mean(n_q_fft_dB), color="cyan", linestyle='--', label=f"n_q: {np.mean(n_q_fft_dB):.2f}dB (piso digital)")

plt.legend()
plt.title(f"FFT de Señal muestreada por un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.2e}V")
plt.ylabel("|X|")
plt.xlabel("frecuencia [Hz]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(7)
plt.hist(n_q)
plt.title(f"Ruido de cuantización para un ADC de {n_bits} bits, $V_r$ = ±{Vfs:.1f}V y q = {q:.2e}V")
plt.ylabel("Cantidad de muestras")
plt.xlabel("Amplitud [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)
plt.axvline(-q/2, linestyle="--",color="red")
plt.axvline(q/2, linestyle="--",color="red")
plt.axhline(100, linestyle="--",color="red")


correlacion = np.correlate(n_q,n_q,"full")/N
lags = np.arange(-(N-1), N)

plt.figure(8)
plt.plot(lags, correlacion,":.")
plt.xlabel("Retardo [muestras]")
plt.ylabel("Autocorrelación")
