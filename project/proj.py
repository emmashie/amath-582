import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import xarray as xr 
import numpy.ma as ma
import cmocean.cm as cmo

plt.ion()
#plt.style.use('ggplot')

fdir = os.path.join('/data2','enuss','funwave_lab_setup','jonswap_test','postprocessing')
eta_df = xr.open_dataset(os.path.join(fdir, 'compiled_output', 'eta.nc'))
t = 500
eta = eta_df['eta'].values[t:,:,:]
time = eta_df['time'].values[t:]*0.2 # time in seconds

depFile = os.path.join(fdir,'compiled_output', 'dep.out')
dep = np.loadtxt(depFile)
[n,m] = dep.shape

# discretization
dx = 0.05
dy = 0.1

# x and y field vectors 
x = np.arange(0,m*dx,dx)
y = np.arange(0,n*dy,dy)

n = np.max(y)
T = 0.2*len(time)
k_pos = np.linspace(0, n/2, int(len(y)/2))
k_neg = np.linspace(-n/2, 0, int(len(y)/2))
k = (2*np.pi/n)*np.append(k_pos, k_neg)
ks = np.fft.fftshift(k)

[tt, yy] = np.meshgrid(y, time)

j = 1540

j = j+100
fig, ax = plt.subplots(figsize=(10,4))
p = ax.pcolormesh(yy[:300,:], tt[:300,:], eta[:300,:,j], cmap=cmo.deep, shading='gouraud')
ax.set_xticks(time[:300][::25])
ax.set_xticklabels(np.arange(0,60,1)[::5])
ax.set_ylabel('Alongshore (m)')
ax.set_xlabel('Time (s)')
fig.colorbar(p, ax=ax, label='$\eta$ (m)')
p.set_clim((-0.1,0.1))
fig.savefig('plots/eta_hov_brk.png')



t = 0
spec_avg = np.mean(np.abs(np.fft.fftshift(np.fft.fft(eta[t:-1,:,j], axis=1))), axis=0)
spec_avg2 = np.mean(np.abs(np.fft.fftshift(np.fft.fft(eta[t:-1,:,j+60], axis=1))), axis=0)
spec_avg3 = np.mean(np.abs(np.fft.fftshift(np.fft.fft(eta[t:-1,:,j+100], axis=1))), axis=0)


fig, ax = plt.subplots()
ax.loglog(ks, spec_avg, color='k', label='Offshore Edge')
ax.loglog(ks, spec_avg2, '--', color='k', label='Mid Region')
ax.loglog(ks, spec_avg3, linestyle='dotted', color='k', label='Breaking Region')
ax.set_ylabel('$S_{\eta \eta}$ (m$^3$)')
ax.set_xlabel('L$^{-1}$ (m$^{-1}$)')
ax.legend(loc='best')
fig.savefig('plots/wavenumber_spec3.png')
