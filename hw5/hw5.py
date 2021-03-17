import numpy as np 
import matplotlib.pyplot as plt
import skvideo.io
import os
import hw5_functions as f 
from scipy import linalg 
plt.ion()
#plt.style.use('ggplot')

### race video
filepath = os.path.join('../data','monte_carlo_low.mp4')
mc = skvideo.io.vread(filepath)  
meta_mc = skvideo.io.ffprobe(filepath)
T = 6.322983
time = np.linspace(0, T, len(mc))

X, X1, X2, U, Sdiag, V, nimgs, nx, ny = f.video_svd(mc, T)
D, Phi = f.mode_appoximation(X2, U, Sdiag, V, modes=3)
foreground, background = f.reconstruct_DMD(D, X, X1, Phi, time, nx, ny, nimgs)

mc_grey = np.reshape(X, (nx,ny,nimgs), order='F')

n = 50
fig, ax = plt.subplots(ncols=3, figsize=(11,3))
ax[0].imshow(foreground[:,:,n], cmap='Greys')
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title('Foreground')

ax[1].imshow(background[:,:,n], cmap='Greys')
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('Background')

ax[2].imshow(mc_grey[:,:,n], cmap='Greys')
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Original')
fig.tight_layout()
fig.savefig('monte_carlo.png')

### ski video
filepath = os.path.join('../data','ski_drop_low.mp4')
ski = skvideo.io.vread(filepath)
meta_ski = skvideo.io.ffprobe(filepath)
T = 7.574233
time = np.linspace(0, T, len(ski))

X, X1, X2, U, Sdiag, V, nimgs, nx, ny = f.video_svd(ski, T)
D, Phi = f.mode_appoximation(X2, U, Sdiag, V, modes=3)
foreground, background = f.reconstruct_DMD(D, X, X1, Phi, time, nx, ny, nimgs)

ski_grey = np.reshape(X, (nx,ny,nimgs), order='F')

n = 50
fig, ax = plt.subplots(ncols=3, figsize=(11,3), sharex=True, sharey=True)
ax[0].imshow(foreground[:,:,n], cmap='Greys')
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title('Foreground')
ax[0].set_xlim(232,725)

ax[1].imshow(background[:,:,n], cmap='Greys')
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('Background')
ax[1].set_xlim(232,725)

ax[2].imshow(ski_grey[:,:,n], cmap='Greys')
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Original')
ax[2].set_xlim(232,725)
fig.tight_layout()
fig.savefig('ski.png')



