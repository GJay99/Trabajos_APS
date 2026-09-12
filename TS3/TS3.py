# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:36:25 2026

@author: ECyT
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

#%%
N=1000
k0=N/4
fs=1000
f0=k0*fs/N

#%%

def mi_funcion_sen( vmax = np.sqrt(2), dc = 0, ff = 1, ph=0, nn = N, fs = fs):
    tt=np.arange(0.0, nn/fs, 1/fs)
    xx=vmax*np.sin(2*np.pi*ff*tt +ph)+dc
    return tt, xx

def mi_fft (xx, N, fs):
    fftxx = fft(xx)
    freq = fftfreq(N,1/fs)
    fftxx_abs = np.abs(fftxx)
    return freq, fftxx_abs

#%% k0=N/4

tt1, sen1=mi_funcion_sen(ff=f0)

k0=N/4+0.25
f0=k0*fs/N
tt2, sen2=mi_funcion_sen(ff=f0)

k0=N/4+0.5
f0=k0*fs/N
tt3, sen3=mi_funcion_sen(ff=f0)

#%%

freq1, fftsen1_abs=mi_fft(sen1, N, fs)
freq2, fftsen2_abs=mi_fft(sen2, N, fs)
freq3, fftsen3_abs=mi_fft(sen3, N, fs)
freq_graf = freq3[:N//2]

ref=np.max(fftsen1_abs)
fftsen1_abs_dB=20*np.log10(fftsen1_abs/ref)
fftsen2_abs_dB=20*np.log10(fftsen2_abs/ref)
fftsen3_abs_dB=20*np.log10(fftsen3_abs/ref)

plt.figure(1)
plt.plot(freq_graf, fftsen1_abs_dB[:N//2],":.", color="red")
plt.title("FFT normalizada Seno k0=N/4")
plt.figure(2)
plt.plot(freq_graf, fftsen2_abs_dB[:N//2],":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25")
plt.figure(3)
plt.plot(freq_graf, fftsen3_abs_dB[:N//2],":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5")

Pot_sen1=np.sum(sen1**2)/N
Pot_fft_sen1=np.sum(fftsen1_abs**2)/N**2
print(Pot_sen1, Pot_fft_sen1)

Pot_sen2=np.sum(sen2**2)/N
Pot_fft_sen2=np.sum(fftsen2_abs**2)/N**2
print(Pot_sen2, Pot_fft_sen2)

Pot_sen3=np.sum(sen3**2)/N
Pot_fft_sen3=np.sum(fftsen3_abs**2)/N**2
print(Pot_sen3, Pot_fft_sen3)

#%%

ceros=np.zeros(9*N)
sen1_0=np.concatenate((sen1,ceros))
sen2_0=np.concatenate((sen2,ceros))
sen3_0=np.concatenate((sen3,ceros))

freq1, fftsen1_0_abs=mi_fft(sen1_0, 10*N, fs)
freq2, fftsen2_0_abs=mi_fft(sen2_0, 10*N, fs)
freq3, fftsen3_0_abs=mi_fft(sen3_0, 10*N, fs)
freq_graf = freq3[:10*N//2]

ref=np.max(fftsen1_0_abs)
fftsen1_0_abs_dB=20*np.log10(fftsen1_0_abs/ref)
fftsen2_0_abs_dB=20*np.log10(fftsen2_0_abs/ref)
fftsen3_0_abs_dB=20*np.log10(fftsen3_0_abs/ref)

plt.figure(4)
plt.plot(freq_graf, fftsen1_0_abs_dB[:10*N//2], ":.", color="red")
plt.title("FFT normalizada Seno k0=N/4 (zero padding)")
plt.xlim(240, 261)
plt.ylim(-80, 5)
plt.figure(5)
plt.plot(freq_graf, fftsen2_0_abs_dB[:10*N//2],":.", color="green")
plt.title("FFT normalizada Seno k0=N/4+0.25 (zero padding)")
plt.xlim(240, 261)
plt.ylim(-80, 5)
plt.figure(6)
plt.plot(freq_graf, fftsen3_0_abs_dB[:10*N//2], ":.", color="blue")
plt.title("FFT normalizada Seno k0=N/4+0.5 (zero padding)")
plt.xlim(240, 261)
plt.ylim(-80, 5)

Pot_sen1=np.sum(sen1**2)/N
Pot_fft_sen1=np.sum(fftsen1_0_abs**2)/N * (1/(10*N))
print(Pot_sen3, Pot_fft_sen3)
print(Pot_sen1, Pot_fft_sen1)

Pot_sen2=np.sum(sen2**2)/N
Pot_fft_sen2=np.sum(fftsen2_0_abs**2)/N * (1/(10*N))
print(Pot_sen3, Pot_fft_sen3)
print(Pot_sen2, Pot_fft_sen2)

Pot_sen3=np.sum(sen3**2)/N
Pot_fft_sen3=np.sum(fftsen3_0_abs**2)/N * (1/(10*N))
print(Pot_sen3, Pot_fft_sen3)

plt.show()


