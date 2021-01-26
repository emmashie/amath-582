import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
plt.ion()

datapath = '../data/'
filename = 'subdata.csv'

subdata = pd.read_csv(os.path.join(datapath, filename), header=None)
subdata = subdata.applymap(lambda s: np.complex(s.replace('i', 'j')))
data = subdata.values

L = 10
n = 64
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (2*np.pi/(2*L))*np.append(k_pos, k_neg)
ks = np.fft.fftshift(k)

x2 = np.linspace(-L,L,n+1)
x = x2[:n]
y = x
z = x
[X, Y, Z] = np.meshgrid(x,y,z)
[Kx,Ky,Kz] = np.meshgrid(ks,ks,ks)

Un = np.asarray([np.reshape(data[:,i], (n,n,n)) for i in range(data.shape[-1])])
#M = np.max(np.abs(Un))


### finding the frequency
Un_fft = np.fft.fftshift(np.fft.fftn(Un, axes=[1,2,3]), axes=[1,2,3])
Un_avg = np.mean(Un_fft, axis=0, dtype=complex)
Un_avg_normalized = np.abs(Un_avg)/np.max(np.abs(Un_avg))

threshold = 0.7 

ind = np.where(Un_avg_normalized>=threshold)

kx = ks[ind[0]]
ky = ks[ind[1]]
kz = ks[ind[2]]

kx_avg = np.mean(kx)
ky_avg = np.mean(ky)
kz_avg = np.mean(kz)

kx_centered = (Kx - kx_avg)**2
ky_centered = (Ky - ky_avg)**2
kz_centered = (Kz - kz_avg)**2

tau = 0.1

gaussian_filter = np.exp(-tau * kx_centered)*np.exp(-tau * ky_centered)*np.exp(-tau * kz_centered)


### filtering the data with a gaussian filter at the frequency we found prior
Un_fft_filtered = np.asarray([Un_fft[i,:,:,:]*gaussian_filter for i in range(len(Un))])

Un_filtered = np.fft.ifftn(np.fft.ifftshift(Un_fft_filtered, axes=[1,2,3]), axes=[1,2,3])
Un_filtered_normalized = np.asarray([np.abs(Un_filtered[i,:,:,:])/np.max(np.abs(Un_filtered[i,:,:,:])) for i in range(len(Un_filtered))])

## find locations at each time step that have the max signal 
#sub_ind = np.squeeze(np.asarray([np.asarray(np.argwhere(Un_filtered[i,:,:,:]==np.max(Un_filtered[i,:,:,:]))) for i in range(len(Un_filtered))]))
sub_ind = np.asarray([np.mean(np.asarray(np.argwhere(Un_filtered_normalized[i,:,:,:]>=threshold)), axis=0) for i in range(len(Un_filtered_normalized))])


sub_x = np.asarray([x[int(sub_ind[i,0])] for i in range(len(sub_ind))])
sub_y = np.asarray([y[int(sub_ind[i,1])] for i in range(len(sub_ind))])
sub_z = np.asarray([z[int(sub_ind[i,2])] for i in range(len(sub_ind))])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(sub_x, sub_y, sub_z, label='parametric curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Submarine Path')


if 0:
	fig = go.Figure(data = go.Isosurface(
					x=Kx.flatten(),
					y=Ky.flatten(),
					z=Kz.flatten(),
					value=gaussian_filter.flatten()))

	fig.show()


