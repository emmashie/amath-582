import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
import xarray as xr 
import numpy.ma as ma
plt.ion()

fdir = os.path.join('/data2','enuss','funwave_lab_setup','jonswap_test','postprocessing')
eta_df = xr.open_dataset(os.path.join(fdir, 'compiled_output', 'eta.nc'))
t = 500
eta = eta_df['eta'].values[t:,:,:]
time = eta_df['time'].values[t:]*0.2 # time in seconds

del eta_df

mask = xr.open_dataset(os.path.join(fdir, 'compiled_output', 'mask.nc'))['mask'].values[t:,:,:]

eta = ma.masked_where(mask==0, eta)

del mask 

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

i = 300
j = 1600

test_spec = np.mean(np.fft.fftshift(np.fft.fft(eta[:,:,j], axis=1)), axis=0)

spec = np.zeros(eta.shape)
for i in range(len(time)):
	for j in range(len(x)):
		spec[i,:,j] =  np.fft.fftshift(np.fft.fft(eta[i,:,j]))

spec_avg = np.mean(spec, axis=0)