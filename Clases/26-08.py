#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:02:58 2026

@author: gonza
"""
#%% imports
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
#%%declaraciones
fs=1000
N=1000

vmax=1
dc=0
k=4
ph=0

#%%seno y fft
nn=np.arange(N)

xx = vmax * np.sin(2*np.pi*(k*fs/N)*nn/fs + ph) + dc

fftxx = fft(xx)
fftxx = fftxx[:N//2]
print(fftxx[k])

freq = fftfreq(N,1/fs)
freq = freq[:N//2]

fftxx_abs = np.abs(fftxx)
     
plt.figure(1)
plt.plot(freq,fftxx_abs)
plt.title("Modulo FFT seno")

fftxx_db = 20 * np.log10(fftxx_abs)

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(freq,fftxx_db)
plt.title("Modulo FFT seno en dB")
#fftxx_filtrado = np.where(np.abs(fftxx)>0.001,fftxx,0) Esto al final dijo que no se hace, pero es interesante
ph = np.angle(fftxx)

plt.subplot(2,1,2)
plt.plot(freq,ph)
plt.title("Fase FFT seno")

#%%seno + ruido y fft
Px = np.var(xx)
SNR = 10
tt = np.arange(0,N)/fs

sigma = np.sqrt(Px/np.power(10,SNR/10))
rr = np.random.normal(scale=sigma,size=len(tt))

Pn = np.var(rr)
SNR_real = 10 * np.log10(Px/Pn)
print(SNR_real)

zz = xx + rr

plt.figure(3)
plt.plot(tt,zz)
plt.title("Seno + Ruido")


fftzz = fft(zz)
fftzz = fftzz[:N//2]

freq = fftfreq(N,1/fs)
freq = freq[:N//2]

print(fftxx[4])

fftzz_abs = np.abs(fftzz)
fftzz_db = 20 * np.log10(fftzz_abs)

plt.figure(4) 
plt.subplot(2,1,1)
plt.plot(freq,fftzz_db)
plt.title("Modulo FFT seno + ruido en dB")


ph = np.angle(fftzz)

plt.subplot(2,1,2)
plt.plot(freq,ph)
plt.title("Fase FFT seno + ruido en dB")

#%% Cuantización
Vmax = max(zz)
Vmin = min(zz)

n_bits = 2
Vfs = max(Vmax,np.abs(Vmin))/0.9
q = (2*Vfs)/(np.pow(2,n_bits))
print(q)
zz_cuantizada = np.round(zz/q)*q
plt.figure(5)
plt.plot(tt,zz_cuantizada,linestyle=" ", marker = ".")
plt.show()