# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

#%% imports
import matplotlib.pyplot as plt
import numpy as np

#%% definiciones
fs = 1000   # Hz
N = 1000    # Muestras

#%% funciones

def mi_funcion_seno(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    tt=np.zeros(nn)
    xx=np.zeros(nn)
    for n in np.arange(0,nn,1):
        tt[n]= n * (1/fs)
        xx[n]= vmax * np.sin(2*np.pi*ff * tt[n] + ph) + dc
    return tt, xx

#%% implementacion

plt.figure(1)

tt,xx = mi_funcion_seno()
plt.plot(tt, xx, linestyle=" ", marker=".")
plt.title("Seno ff=1 Hz")
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(2)

tt,xx = mi_funcion_seno(ff=500)
plt.plot(tt, xx, linestyle=" ", marker=".")
plt.title("Seno ff=500 Hz")
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(3)

tt,xx = mi_funcion_seno(ff=999)
plt.plot(tt, xx, linestyle=" ", marker=".")
plt.title("Seno ff=999 Hz")
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(4)

tt,xx = mi_funcion_seno(ff=1001)
plt.plot(tt, xx, linestyle=" ", marker=".")
plt.title("Seno ff=1001 Hz")
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)

plt.figure(5)

tt,xx = mi_funcion_seno(ff=2001)
plt.plot(tt, xx, linestyle=" ", marker=".")
plt.title("Seno ff=2001 Hz")
plt.xlabel("tiempo [s]")
plt.ylabel("voltaje [V]")
plt.grid()
plt.axhline(0,color="black")
plt.axvline(0,color="black",linewidth=1)