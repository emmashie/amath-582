import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
plt.ion()

# definte data path and filename
datapath = '../data/'
filename = 'subdata.csv'

# load in data and reformat complex numbers for python
subdata = pd.read_csv(os.path.join(datapath, filename), header=None)
subdata = subdata.applymap(lambda s: np.complex(s.replace('i', 'j')))
data = subdata.values

# define spatial and frequency domains 
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
[X, Y, Z] = np.meshgrid(x,y,z, indexing='ij')
[Kx,Ky,Kz] = np.meshgrid(ks,ks,ks, indexing='ij')

### NOTE: Matlab convention is (Y, X, Z) in meshgrid versus (X, Y, Z) here, 
### so x and y are actually flipped throughout the code 
### variables are not switched, but x-y labels for plotting and submarine
### coordinate reporting are switched to be consistent with Matlab


# reshape data into 4-d time-spatial data array
Un = np.asarray([np.reshape(data[:,i], (n,n,n), order='F') for i in range(data.shape[-1])])

# find the frequency of the submarine
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


if 0: # plotting code for isosurface of normalized and averaged data
	fig = go.Figure(data = go.Isosurface(
					x=Kx.flatten(),
					y=Ky.flatten(),
					z=Kz.flatten(),
					value=Un_avg_normalized.flatten(),
					isomin=0.2,
    				isomax=1,
    				surface_count=15,
    				opacity=0.3))
	fig.update_layout(scene = dict(
                    xaxis_title='ky', 
                    yaxis_title='kx',
                    zaxis_title='kz'))
	fig.show()

kx_centered = (Kx - kx_avg)**2
ky_centered = (Ky - ky_avg)**2
kz_centered = (Kz - kz_avg)**2

# creating gaussian filter
tau = 0.1

gaussian_filter = np.exp(-tau * kx_centered)*np.exp(-tau * ky_centered)*np.exp(-tau * kz_centered)

if 0: # plotting gaussian filter
	fig = go.Figure(data = go.Isosurface(
					x=Kx.flatten(),
					y=Ky.flatten(),
					z=Kz.flatten(),
					value=gaussian_filter.flatten(),
					isomin=0.1,
    				isomax=1,
    				surface_count=15,
    				opacity=0.3))
	fig.update_layout(scene = dict(
                    xaxis_title='ky',
                    yaxis_title='kx',
                    zaxis_title='kz'))
	fig.show()

# filtering the data with a gaussian filter at the frequency we found prior
Un_fft_filtered = np.asarray([Un_fft[i,:,:,:]*gaussian_filter for i in range(len(Un))])

Un_filtered = np.fft.ifftn(np.fft.ifftshift(Un_fft_filtered, axes=[1,2,3]), axes=[1,2,3])
Un_filtered_normalized = np.asarray([np.abs(Un_filtered[i,:,:,:])/np.max(np.abs(Un_filtered[i,:,:,:])) for i in range(len(Un_filtered))])

# find locations at each time step that have signal above threshold 
sub_ind = np.asarray([np.mean(np.asarray(np.argwhere(Un_filtered_normalized[i,:,:,:]>=threshold)), axis=0) for i in range(len(Un_filtered_normalized))])

# finding the center of mass of the signal 
sub_x = np.asarray([x[int(sub_ind[i,0])] for i in range(len(sub_ind))])
sub_y = np.asarray([y[int(sub_ind[i,1])] for i in range(len(sub_ind))])
sub_z = np.asarray([z[int(sub_ind[i,2])] for i in range(len(sub_ind))])

if 0: # plotting submarine path
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot(sub_y, sub_x, sub_z, color='black')
	ax.plot(sub_y[-1], sub_x[-1], sub_z[-1], 'o', color='tab:blue')
	ax.plot(sub_y[0], sub_x[0], sub_z[0], 'o', color='tab:green')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_xlim((-L,L))
	ax.set_ylim((-L,L))
	ax.set_zlim((-L,L))
	ax.locator_params(nbins=5)
	fig.savefig('sub_path.png')

if 0: # plotting aircraft path
	plt.style.use('ggplot')
	fig, ax = plt.subplots()
	ax.plot(sub_y, sub_x, color='black')
	ax.plot(sub_y[-1], sub_x[-1], 'o', color='tab:blue')
	ax.plot(sub_y[0], sub_x[0], 'o', color='tab:green')
	ax.set_xlabel('X', fontsize=14)
	ax.set_ylabel('Y', fontsize=14)
	ax.set_xticks(np.arange(-L, L)[::3])
	ax.set_yticks(np.arange(-L, L)[::3])
	ax.set_xticklabels(np.arange(-L, L)[::3], fontsize=14)
	ax.set_yticklabels(np.arange(-L, L)[::3], fontsize=14)	
	ax.set_xlim((-L,L))
	ax.set_ylim((-L,L))	
	fig.savefig('air_path.png')

if 0:
	# write x, y, z coordinates to csv for table generation
	f = open('xyz_coords.csv', 'w')
	subx = sub_x[::4]
	suby = sub_y[::4]
	subz = sub_z[::4]
	for i in range(len(subx)-1):
		f.write("%0.1f," % (subx[i]))
	f.write("%0.1f\n" % (subx[-1]))
	for i in range(len(suby)-1):
		f.write("%0.1f," % (suby[i]))
	f.write("%0.1f\n" % (suby[-1]))	
	for i in range(len(subz)-1):
		f.write("%0.1f," % (subz[i]))
	f.write("%0.1f\n" % (subz[-1]))	
	f.close()



