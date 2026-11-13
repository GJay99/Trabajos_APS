# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 21:06:19 2026

@author: Joaco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import scipy.io as sio

sio.whosmat('ECG_TP4.mat')
mat_struct=sio.loadmat('./ECG_TP4.mat')

ecg_one_lead_audio =mat_struct ['ecg_lead'] * 1.0
ecg_one_lead_audio = ecg_one_lead_audio.flatten()
s = ecg_one_lead_audio

#%% Mediana
b_intermedio = signal.medfilt(s,201)
x_sombrero_int = s - b_intermedio

b_sombrero = signal.medfilt(b_intermedio,601)
x_sombrero = s - b_sombrero

plt.figure(1)
plt.plot(s[8000:13000],label="Señal original")
plt.plot(x_sombrero_int[8000:13000],label="Señal intermedia")
plt.plot(x_sombrero[8000:13000],label="Señal Filtrada")
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(s[3000:6000],label="Señal original")
plt.plot(x_sombrero_int[3000:6000],label="Señal intermedia")
plt.plot(x_sombrero[3000:6000],label="Señal Filtrada")
plt.legend()
plt.grid()

plt.legend()
plt.grid()

#%% Spline cúbica

ecg_QRS = mat_struct ['qrs_detections']
ecg_QRS  = ecg_QRS.flatten()

prueba = ecg_QRS[ecg_QRS<6000]

plt.figure(3)
plt.plot(ecg_one_lead_audio[:6000])
plt.vlines(x=prueba, ymin=np.min(ecg_one_lead_audio), ymax=np.max(ecg_one_lead_audio), color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='QRS')


zonasmuertas_start = ecg_QRS-60
zonasmuertas_end = ecg_QRS-40

zonasmuertas = np.array([x for par in zip(zonasmuertas_start, zonasmuertas_end) for x in par])

pq_hat = np.array([np.mean(ecg_one_lead_audio[a:b]) for a,b in zip(zonasmuertas_start,zonasmuertas_end)]).astype(int)
pq_n_hat = np.array([np.mean([a,b]) for a,b in zip(zonasmuertas_start,zonasmuertas_end)])

plt.figure(4)
plt.plot(ecg_one_lead_audio)
plt.plot(pq_n_hat, pq_hat, marker = ".", linestyle = " ")


prueba = zonasmuertas[zonasmuertas<6000]

plt.figure(5)
plt.plot(ecg_one_lead_audio[:6000])
plt.vlines(x=prueba, ymin=np.min(ecg_one_lead_audio), ymax=np.max(ecg_one_lead_audio), color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='QRS')


inter = interpolate.CubicSpline(pq_n_hat, pq_hat)

x = np.arange(0,len(ecg_one_lead_audio),1)
linea_base = inter(x)

ecg_filtrado_spline = ecg_one_lead_audio -linea_base

plt.figure(6)
plt.plot(ecg_one_lead_audio,label="original")
plt.plot(ecg_filtrado_spline, label = "filtrado")
plt.legend()
plt.grid()
plt.show()

plt.figure(7)
plt.plot(ecg_one_lead_audio[8000: 13000],label="original")
plt.plot(ecg_filtrado_spline[8000: 13000], label = "filtrado")
plt.legend()
plt.grid()
plt.show()

plt.figure(8)
plt.plot(ecg_one_lead_audio[3000: 6000],label="original")
plt.plot(ecg_filtrado_spline[3000: 6000], label = "filtrado")
plt.legend()
plt.grid()
plt.show()

plt.figure(9)
plt.plot(ecg_one_lead_audio)
plt.plot(linea_base)
plt.grid()
plt.show()