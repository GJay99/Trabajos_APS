#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:02:51 2026

@author: gonza
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle
import scipy.io as sio

fs_ecg = 1000

sio.whosmat('ECG_TP4.mat')
mat_struct=sio.loadmat('./ECG_TP4.mat')

ecg_one_lead_audio =mat_struct ['ecg_lead'] * 1.0
ecg_one_lead_audio = ecg_one_lead_audio.flatten()

ws1 = 0.1
wp1 = 1
wp2 = 35
ws2 = 40
wp = [wp1,wp2]
ws = [ws1,ws2]

G_p = 3
G_s = 40

fs = 1000
f_nq = fs/2

#%% Diseño IIR
sos_butter = signal.iirdesign(wp, ws, G_p, G_s, ftype='butter',output='sos',fs=fs)
f_butter, H_butter = signal.freqz_sos(sos_butter,fs=fs)

sos_ellip = signal.iirdesign(wp, ws, G_p, G_s, ftype='ellip',output='sos',fs=fs)
f_ellip, H_ellip = signal.freqz_sos(sos_ellip,fs=fs)

sos_cheby = signal.iirdesign(wp, ws, G_p, G_s, ftype='cheby2',output='sos',fs=fs)
f_cheby, H_cheby = signal.freqz_sos(sos_cheby,fs=fs)

modulo_butter = np.abs(H_butter)                 
fase_butter = np.unwrap(np.angle(H_butter))
retardo_butter = -np.gradient(np.unwrap(np.angle(H_butter)), f_butter)

modulo_ellip = np.abs(H_ellip)                 
fase_ellip = np.unwrap(np.angle(H_ellip))
retardo_ellip = -np.gradient(np.unwrap(np.angle(H_ellip)), f_ellip)   

modulo_cheby = np.abs(H_cheby)                 
fase_cheby = np.unwrap(np.angle(H_cheby))
retardo_cheby = -np.gradient(np.unwrap(np.angle(H_cheby)), f_cheby) 


plt.figure(1, figsize=(9,9))

# ---------------- MÓDULO ----------------
plt.subplot(2,1,1)

plt.plot(f_butter, 20*np.log10(np.maximum(modulo_butter, 1e-12)),label="Butter")
plt.plot(f_ellip, 20*np.log10(np.maximum(modulo_ellip, 1e-12)),label="Cauer")
plt.plot(f_cheby, 20*np.log10(np.maximum(modulo_cheby, 1e-12)),label="Chebyshev II")

plt.title("Respuesta en frecuencia")
plt.ylabel("Módulo (dB)")

# Límites de las bandas
plt.axvline(ws1, color="red", linestyle="--", label="ws")
plt.axvline(wp1, color="orange", linestyle="--", label="wp")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.axhline(-G_p, color="orange", linestyle=":", label="Límite Gp")
plt.axhline(-G_s, color="red", linestyle=":", label="Límite Gs")

# Límites verticales del gráfico
plt.ylim(-80, 5)
y_min, y_max = plt.ylim()

# Banda de rechazo inferior
plt.fill_between([0, ws1], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

# Banda pasante: zona no permitida por debajo de -Gp
plt.fill_between([wp1, wp2], y_min, -G_p,color="red", alpha=0.12,hatch="xxx", edgecolor="red",label="Plantilla")

# Banda de rechazo superior
plt.fill_between([ws2, fs_ecg/2], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

plt.xlim(0, ws2+50)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()


# ---------------- FASE ----------------
plt.subplot(2,1,2)

plt.plot(f_butter, fase_butter, label="Butter")
plt.plot(f_ellip, fase_ellip, label="Cauer")
plt.plot(f_cheby, fase_cheby, label="Chebyshev II")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Fase (rad)")
plt.xlim(0, ws2+50)

plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()

plt.tight_layout()
# ---------------- RETARDO ----------------
plt.figure(2,figsize=(9,9))

plt.title("Retardo de grupo")

plt.plot(f_butter, retardo_butter, label="Butter")
plt.plot(f_ellip, retardo_ellip, label="Cauer")
plt.plot(f_cheby, retardo_cheby, label="Chebyshev II")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Retardo (ms)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, ws2+50)

plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()

#%% Diseño FIR
ws1=0.1
wp1=1
wp2 = 35
ws2 = 40

taps_firls = 1001
taps_win = 1001
taps_PMR= 1001

freqs = [0.0,ws1,wp1,wp2+2,wp2+5,fs_ecg/2]
gains = [0.0,0.0,1.0,1.0,0.0,0.0]

ff = np.append(np.logspace(np.log10(0.001),np.log10(1),2000),np.linspace(1.001,35,2000))
ff = np.append(ff,np.logspace(np.log10(35.001),np.log10(40),2000))
ff = np.append(ff,np.linspace(41,fs_ecg/2,2000))

b_ls = signal.firls(taps_firls, freqs,gains,weight=[500,1.0,500],fs=fs_ecg)
b_win_hamming=signal.firwin2(taps_win,freqs, gains, window="hamming", fs=fs_ecg)
b_PMR = signal.remez(taps_PMR,freqs,[0,1,0],weight=[20,1.0,20],fs=fs_ecg)

f_ls, H_ls = signal.freqz(b_ls, worN = ff,fs=fs_ecg)
f_hamming, H_hamming = signal.freqz(b_win_hamming,worN=ff, fs=fs_ecg)
f_PMR, H_PMR = signal.freqz(b_PMR,worN=ff, fs=fs_ecg)

# Módulos
modulo_ls = np.abs(H_ls)
modulo_hamming = np.abs(H_hamming)
modulo_PMR = np.abs(H_PMR)

# Fases
fase_ls = np.unwrap(np.angle(H_ls))
fase_hamming = np.unwrap(np.angle(H_hamming))
fase_PMR = np.unwrap(np.angle(H_PMR))

# Retardos de grupo (ms)
retardo_ls = np.full_like(f_ls, (taps_firls - 1) / 2 * 1000 / fs_ecg)
retardo_hamming = np.full_like(f_hamming, (taps_win - 1) / 2 * 1000 / fs_ecg)
retardo_PMR = np.full_like(f_PMR, (taps_PMR - 1) / 2 * 1000 / fs_ecg)

plt.figure(3, figsize=(9,9))

# ---------------- MÓDULO ----------------
plt.subplot(2,1,1)

plt.plot(f_ls, 20*np.log10(np.maximum(modulo_ls, 1e-12)),label="Cuadrados mínimos")
plt.plot(f_hamming, 20*np.log10(np.maximum(modulo_hamming, 1e-12)),label="Hamming")
plt.plot(f_PMR, 20*np.log10(np.maximum(modulo_PMR, 1e-12)),label="Parks-Mc Clellan-Remez")

plt.title("Respuesta en frecuencia")
plt.ylabel("Módulo (dB)")

# Límites de las bandas
plt.axvline(ws1, color="red", linestyle="--", label="ws")
plt.axvline(wp1, color="orange", linestyle="--", label="wp")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.axhline(-G_p, color="orange", linestyle=":", label="Límite Gp")
plt.axhline(-G_s, color="red", linestyle=":", label="Límite Gs")

# Límites verticales del gráfico
plt.ylim(-100, 5)
y_min, y_max = plt.ylim()

# Banda de rechazo inferior
plt.fill_between([0, ws1], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

# Banda pasante: zona no permitida por debajo de -Gp
plt.fill_between([wp1, wp2], y_min, -G_p,color="red", alpha=0.12,hatch="xxx", edgecolor="red",label="Plantilla")

# Banda de rechazo superior
plt.fill_between([ws2, fs_ecg/2], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

plt.xlim(0, ws2+50)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()


# ---------------- FASE ----------------
plt.subplot(2,1,2)

plt.plot(f_ls, fase_ls, label="Cuadrados mínimos")
plt.plot(f_hamming, fase_hamming, label="Hamming")
plt.plot(f_PMR, fase_PMR, label="Parks-Mc Clellan-Remez")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Fase (rad)")
plt.xlim(0, ws2+50)

plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()

plt.tight_layout()
# ---------------- RETARDO ----------------
plt.figure(4,figsize=(9,9))

plt.plot(f_ls, retardo_ls, label="Cuadrados mínimos")
plt.plot(f_hamming, retardo_hamming, label="Hamming")
plt.plot(f_PMR, retardo_PMR, label="Parks-Mc Clellan-Remez")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Retardo (ms)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, ws2+50)

plt.grid(True)
plt.legend()

#%%Todo junto
plt.figure(5, figsize=(9,9))

# ---------------- MÓDULO ----------------
plt.subplot(2,1,1)

plt.plot(f_ls, 20*np.log10(np.maximum(modulo_ls, 1e-12)),label="Cuadrados mínimos")
plt.plot(f_hamming, 20*np.log10(np.maximum(modulo_hamming, 1e-12)),label="Hamming")
plt.plot(f_PMR, 20*np.log10(np.maximum(modulo_PMR, 1e-12)),label="Parks-Mc Clellan-Remez")
plt.plot(f_butter, 20*np.log10(np.maximum(modulo_butter, 1e-12)),label="Butter")
plt.plot(f_ellip, 20*np.log10(np.maximum(modulo_ellip, 1e-12)),label="Cauer")
plt.plot(f_cheby, 20*np.log10(np.maximum(modulo_cheby, 1e-12)),label="Chebyshev II")

plt.title("Respuesta en frecuencia")
plt.ylabel("Módulo (dB)")

# Límites de las bandas
plt.axvline(ws1, color="red", linestyle="--", label="ws")
plt.axvline(wp1, color="orange", linestyle="--", label="wp")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.axhline(-G_p, color="orange", linestyle=":", label="Límite Gp")
plt.axhline(-G_s, color="red", linestyle=":", label="Límite Gs")

# Límites verticales del gráfico
plt.ylim(-100, 5)
y_min, y_max = plt.ylim()

# Banda de rechazo inferior
plt.fill_between([0, ws1], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

# Banda pasante: zona no permitida por debajo de -Gp
plt.fill_between([wp1, wp2], y_min, -G_p,color="red", alpha=0.12,hatch="xxx", edgecolor="red",label="Plantilla")

# Banda de rechazo superior
plt.fill_between([ws2, fs_ecg/2], -G_s, y_max,color="red", alpha=0.12,hatch="xxx", edgecolor="red")

plt.xlim(0, ws2+50)
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()


# ---------------- FASE ----------------
plt.subplot(2,1,2)

plt.plot(f_ls, fase_ls, label="Cuadrados mínimos")
plt.plot(f_hamming, fase_hamming, label="Hamming")
plt.plot(f_PMR, fase_PMR, label="Parks-Mc Clellan-Remez")
plt.plot(f_butter, fase_butter, label="Butter")
plt.plot(f_ellip, fase_ellip, label="Cauer")
plt.plot(f_cheby, fase_cheby, label="Chebyshev II")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Fase (rad)")
plt.xlim(0, ws2+50)

plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.grid(True)
plt.legend()

plt.tight_layout()
# ---------------- RETARDO ----------------
plt.figure(6,figsize=(9,9))

plt.plot(f_ls, retardo_ls, label="Cuadrados mínimos")
plt.plot(f_hamming, retardo_hamming, label="Hamming")
plt.plot(f_PMR, retardo_PMR, label="Parks-Mc Clellan-Remez")
plt.plot(f_butter, retardo_butter, label="Butter")
plt.plot(f_ellip, retardo_ellip, label="Cauer")
plt.plot(f_cheby, retardo_cheby, label="Chebyshev II")

plt.axvline(ws1, color="red", linestyle="--")
plt.axvline(wp1, color="orange", linestyle="--")
plt.axvline(wp2, color="orange", linestyle="--")
plt.axvline(ws2, color="red", linestyle="--")

plt.ylabel("Retardo (ms)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, ws2+50)

plt.grid(True)
plt.legend()

#%% Implementacion IIR

y_b = signal.sosfilt(sos_butter,ecg_one_lead_audio)
y_e = signal.sosfilt(sos_ellip,ecg_one_lead_audio)
y_c = signal.sosfilt(sos_cheby,ecg_one_lead_audio)
x = np.arange(0,len(ecg_one_lead_audio)+1,1)

plt.figure(7)
plt.title("Filtrado zona de interes 4000-5500")
plt.plot(x[4000:5500],y_b[4000:5500],label='Señal filtrada por SOS mediante Butterworth')
plt.plot(x[4000:5500],y_e[4000:5500],label='Señal filtrada por SOS mediante Cauer')
plt.plot(x[4000:5500],y_c[4000:5500],label='Señal filtrada por SOS mediante Chebyshev')
plt.plot(x[4000:5500],ecg_one_lead_audio[4000:5500],label='Señal de ECG')
plt.grid()
plt.legend()


plt.figure(8)
plt.title("Filtrado zona de interes 10000-11000")
plt.plot(x[10000:11000],y_b[10000: 11000],label='Señal filtrada por SOS mediante Butterworth')
plt.plot(x[10000:11000],y_e[10000: 11000],label='Señal filtrada por SOS mediante Cauer')
plt.plot(x[10000:11000],y_c[10000: 11000],label='Señal filtrada por SOS mediante Chebyshev')
plt.plot(x[10000:11000],ecg_one_lead_audio[10000: 11000],label='Señal de ECG')
plt.grid()
plt.legend()

#%% Implementación FIR

ecg_hamming = signal.lfilter(b_win_hamming,1.0, ecg_one_lead_audio)
ecg_ls = signal.lfilter(b_ls,1.0, ecg_one_lead_audio)
ecg_PMR = signal.lfilter(b_PMR,1.0, ecg_one_lead_audio)

plt.figure(9)
plt.title("Filtrado zona de interes 4000-5500")
plt.plot(x[4000:5500],ecg_hamming[4000:5500],label='Señal filtrada por IIR mediante Ventana de Hamming')
plt.plot(x[4000:5500],ecg_ls[4000:5500],label='Señal filtrada por IIR mediante cuadrados mínimos')
plt.plot(x[4000:5500],ecg_PMR[4000:5500],label="Señal filtrada por IIR mediante Parks-Mc Clellan-Remez")
plt.plot(x[4000:5500],ecg_one_lead_audio[4000:5500],label='Señal de ECG')
plt.grid()
plt.legend()


plt.figure(10)
plt.title("Filtrado zona de interes 10000-11000")
plt.plot(x[10000:11000],ecg_hamming[10000: 11000],label='Señal filtrada por IIR mediante Ventana de Hamming')
plt.plot(x[10000:11000],ecg_ls[10000: 11000],label='Señal filtrada por IIR mediante cuadrados mínimos')
plt.plot(x[10000:11000],y_c[10000: 11000],label="Señal filtrada por IIR mediante Parks-Mc Clellan-Remez")
plt.plot(x[10000:11000],ecg_PMR[10000: 11000],label='Señal de ECG')
plt.grid()
plt.legend()

retraso_ls = (taps_firls-1)//2
retraso_win = (taps_win-1)//2
retraso_PMR = (taps_PMR-1)//2

plt.figure(11)
plt.title("Filtrado zona de interes 4000-5500 (retraso corregido)")
plt.plot(x[4000:5500],ecg_hamming[4000+retraso_win:5500+retraso_win],label='Señal filtrada por IIR mediante Ventana de Hamming')
plt.plot(x[4000:5500],ecg_ls[4000+retraso_ls:5500+retraso_ls],label='Señal filtrada por IIR mediante cuadrados mínimos')
plt.plot(x[4000:5500],ecg_PMR[4000+retraso_PMR:5500+retraso_PMR],label="Señal filtrada por IIR mediante Parks-Mc Clellan-Remez")
plt.plot(x[4000:5500],ecg_one_lead_audio[4000:5500],label='Señal de ECG')
plt.grid()
plt.legend()


plt.figure(12)
plt.title("Filtrado zona de interes 10000-11000 (retraso corregido)")
plt.plot(x[10000:11000],ecg_hamming[10000+retraso_win: 11000+retraso_win],label='Señal filtrada por IIR mediante Ventana de Hamming')
plt.plot(x[10000:11000],ecg_ls[10000+retraso_ls: 11000+retraso_ls],label='Señal filtrada por IIR mediante cuadrados mínimos')
plt.plot(x[10000:11000],ecg_PMR[10000+retraso_PMR: 11000+retraso_PMR],label="Señal filtrada por IIR mediante Parks-Mc Clellan-Remez")
plt.plot(x[10000:11000],ecg_one_lead_audio[10000: 11000],label='Señal de ECG')
plt.grid()
plt.legend()

plt.show()
