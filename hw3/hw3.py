import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import hw3_functions as funct
plt.ion()
plt.style.use('ggplot')

# define data path 
datapath = '../data/'

########### TEST 1 #############
# define filenames
cam11_file = 'cam1_1.mat'
cam21_file = 'cam2_1.mat'
cam31_file = 'cam3_1.mat'

# load mat files into python
cam11 = sio.loadmat(os.path.join(datapath, cam11_file))
cam21 = sio.loadmat(os.path.join(datapath, cam21_file))
cam31 = sio.loadmat(os.path.join(datapath, cam31_file))

# pull camera data from mat files
vid11 = cam11['vidFrames1_1']
vid21 = cam21['vidFrames2_1']
vid31 = cam31['vidFrames3_1']

# put camera data into greyscale
vid11_grey = np.asarray([funct.rgb2grey(vid11[:,:,:,i]) for i in range(len(vid11[0,0,0,:]))])
vid21_grey = np.asarray([funct.rgb2grey(vid21[:,:,:,i]) for i in range(len(vid21[0,0,0,:]))])
vid31_grey = np.asarray([funct.rgb2grey(vid31[:,:,:,i]) for i in range(len(vid31[0,0,0,:]))])

# find indices of mass from each frame
mindx11, mindy11 = funct.find_ind_and_plot(vid11_grey, 'plots/vid11_animation/fig', xmin=250, xmax=400, plot=False)
mindx21, mindy21 = funct.find_ind_and_plot(vid21_grey, 'plots/vid21_animation/fig', xmin=200, xmax=400, plot=False)
mindx31, mindy31 = funct.find_ind_and_plot(vid31_grey, 'plots/vid31_animation/fig', xmin=200, xmax=600, plot=False)

# plot 
funct.plot_positions(mindx11, mindy11, mindx21, mindy21, mindx31, mindy31, '1')
funct.plot_positions(mindx11, mindy11, mindx21[10:], mindy21[10:], mindx31, mindy31, '1_shifted')

minlen = np.min([len(mindx11), len(mindx21[10:]), len(mindx31)])

X = np.zeros((6, minlen))
X[0,:] = mindx11[:minlen]
X[1,:] = mindy11[:minlen]
X[2,:] = mindx21[10:10+minlen]
X[3,:] = mindy21[10:10+minlen]
X[4,:] = mindx31[:minlen]
X[5,:] = mindy31[:minlen]

X = X - np.expand_dims(np.mean(X, axis=-1), axis=-1)

## python vs. matlab differences in svd function: https://stackoverflow.com/questions/50930899/svd-command-in-python-v-s-matlab
[m, n] = X.shape
U, Sdiag, VH = np.linalg.svd(X) 
V = VH.T

pca1 = np.matmul(U.T, X)
percentage1 = Sdiag**2/np.sum(Sdiag**2)

########### TEST 2 #############
# define filenames
cam12_file = 'cam1_2.mat'
cam22_file = 'cam2_2.mat'
cam32_file = 'cam3_2.mat'

# load mat files into python
cam12 = sio.loadmat(os.path.join(datapath, cam12_file))
cam22 = sio.loadmat(os.path.join(datapath, cam22_file))
cam32 = sio.loadmat(os.path.join(datapath, cam32_file))

# pull camera data from mat files
vid12 = cam12['vidFrames1_2']
vid22 = cam22['vidFrames2_2']
vid32 = cam32['vidFrames3_2']

# put camera data into greyscale
vid12_grey = np.asarray([funct.rgb2grey(vid12[:,:,:,i]) for i in range(len(vid12[0,0,0,:]))])
vid22_grey = np.asarray([funct.rgb2grey(vid22[:,:,:,i]) for i in range(len(vid22[0,0,0,:]))])
vid32_grey = np.asarray([funct.rgb2grey(vid32[:,:,:,i]) for i in range(len(vid32[0,0,0,:]))])

# find indices of mass from each frame
mindx12, mindy12 = funct.find_ind_and_plot(vid12_grey, 'plots/vid12_animation/fig', xmin=250, xmax=500, plot=False)
mindx22, mindy22 = funct.find_ind_and_plot(vid22_grey, 'plots/vid22_animation/fig', xmin=200, xmax=450, plot=False)
mindx32, mindy32 = funct.find_ind_and_plot(vid32_grey, 'plots/vid32_animation/fig', xmin=225, xmax=600, plot=False)

# plot 
funct.plot_positions(mindx12, mindy12, mindx22, mindy22, mindx32, mindy32, '2')

# svd calculation
minlen = np.min([len(mindx12), len(mindx22), len(mindx32)])

X = np.zeros((6, minlen))
X[0,:] = mindx12[:minlen]
X[1,:] = mindy12[:minlen]
X[2,:] = mindx22[:minlen]
X[3,:] = mindy22[:minlen]
X[4,:] = mindx32[:minlen]
X[5,:] = mindy32[:minlen]

X = X - np.expand_dims(np.mean(X, axis=-1), axis=-1)

[m, n] = X.shape
U, Sdiag, VH = np.linalg.svd(X) 
V = VH.T
#Xrank1_2 = np.matmul(np.expand_dims(U[:,0]*Sdiag[0],axis=-1), np.expand_dims(V[:,0], axis=-1).T)

pca2 = np.matmul(U.T, X)
percentage2 = Sdiag**2/np.sum(Sdiag**2)

########### TEST 3 #############
# define filenames
cam13_file = 'cam1_3.mat'
cam23_file = 'cam2_3.mat'
cam33_file = 'cam3_3.mat'

# load mat files into python
cam13 = sio.loadmat(os.path.join(datapath, cam13_file))
cam23 = sio.loadmat(os.path.join(datapath, cam23_file))
cam33 = sio.loadmat(os.path.join(datapath, cam33_file))

# pull camera data from mat files
vid13 = cam13['vidFrames1_3']
vid23 = cam23['vidFrames2_3']
vid33 = cam33['vidFrames3_3']

# put camera data into greyscale
vid13_grey = np.asarray([funct.rgb2grey(vid13[:,:,:,i]) for i in range(len(vid13[0,0,0,:]))])
vid23_grey = np.asarray([funct.rgb2grey(vid23[:,:,:,i]) for i in range(len(vid23[0,0,0,:]))])
vid33_grey = np.asarray([funct.rgb2grey(vid33[:,:,:,i]) for i in range(len(vid33[0,0,0,:]))])

# find indices of mass from each frame
mindx13, mindy13 = funct.find_ind_and_plot(vid13_grey, 'plots/vid13_animation/fig', xmin=250, xmax=450, plot=False)
mindx23, mindy23 = funct.find_ind_and_plot(vid23_grey, 'plots/vid23_animation/fig', xmin=200, xmax=450, ymin=200, ymax=415, plot=False, restricty=True)
mindx33, mindy33 = funct.find_ind_and_plot(vid33_grey, 'plots/vid33_animation/fig', xmin=225, xmax=600, plot=False)

# plot 
funct.plot_positions(mindx13, mindy13, mindx23, mindy23, mindx33, mindy33, '3')

# svd calculation
minlen = np.min([len(mindx13), len(mindx23), len(mindx33)])

X = np.zeros((6, minlen))
X[0,:] = mindx13[:minlen]
X[1,:] = mindy13[:minlen]
X[2,:] = mindx23[:minlen]
X[3,:] = mindy23[:minlen]
X[4,:] = mindx33[:minlen]
X[5,:] = mindy33[:minlen]

X = X - np.expand_dims(np.mean(X, axis=-1), axis=-1)

[m, n] = X.shape
U, Sdiag, VH = np.linalg.svd(X) 
V = VH.T
pca3 = np.matmul(U.T, X)
percentage3 = Sdiag**2/np.sum(Sdiag**2)


########### TEST 4 #############
# define filenames
cam14_file = 'cam1_4.mat'
cam24_file = 'cam2_4.mat'
cam34_file = 'cam3_4.mat'

# load mat files into python
cam14 = sio.loadmat(os.path.join(datapath, cam14_file))
cam24 = sio.loadmat(os.path.join(datapath, cam24_file))
cam34 = sio.loadmat(os.path.join(datapath, cam34_file))

# pull camera data from mat files
vid14 = cam14['vidFrames1_4']
vid24 = cam24['vidFrames2_4']
vid34 = cam34['vidFrames3_4']

# put camera data into greyscale
vid14_grey = np.asarray([funct.rgb2grey(vid14[:,:,:,i]) for i in range(len(vid14[0,0,0,:]))])
vid24_grey = np.asarray([funct.rgb2grey(vid24[:,:,:,i]) for i in range(len(vid24[0,0,0,:]))])
vid34_grey = np.asarray([funct.rgb2grey(vid34[:,:,:,i]) for i in range(len(vid34[0,0,0,:]))])

# find indices of mass from each frame
mindx14, mindy14 = funct.find_ind_and_plot(vid14_grey, 'plots/vid14_animation/fig', xmin=300, xmax=500, plot=False)
mindx24, mindy24 = funct.find_ind_and_plot(vid24_grey, 'plots/vid24_animation/fig', xmin=215, xmax=400, plot=False)
mindx34, mindy34 = funct.find_ind_and_plot(vid34_grey, 'plots/vid34_animation/fig', xmin=200, xmax=600, plot=False)

# plot 
funct.plot_positions(mindx14, mindy14, mindx24, mindy24, mindx34, mindy34, '4')

# svd calculation
minlen = np.min([len(mindx14), len(mindx24), len(mindx34)])

X = np.zeros((6, minlen))
X[0,:] = mindx14[:minlen]
X[1,:] = mindy14[:minlen]
X[2,:] = mindx24[:minlen]
X[3,:] = mindy24[:minlen]
X[4,:] = mindx34[:minlen]
X[5,:] = mindy34[:minlen]

X = X - np.expand_dims(np.mean(X, axis=-1), axis=-1)

[m, n] = X.shape
U, Sdiag, VH = np.linalg.svd(X) 
V = VH.T
pca4 = np.matmul(U.T, X)
percentage4 = Sdiag**2/np.sum(Sdiag**2)


### analysis figures ###
fig, ax = plt.subplots(figsize=(12,7), nrows=2, ncols=2, sharex=False, sharey=True)
ax[0,0].plot(mindx11-np.mean(mindx11), '--',  color='tab:blue')
ax[0,0].plot(mindy11-np.mean(mindy11),label='Camera 1', color='tab:blue')
ax[0,0].plot(mindx21-np.mean(mindx21), '--', color='tab:green')
ax[0,0].plot(mindy21-np.mean(mindy21), label='Camera 2', color='tab:green')
ax[0,0].plot(mindx31-np.mean(mindx31), '--', color='tab:red')
ax[0,0].plot(mindy31-np.mean(mindy31), label='Camera 3', color='tab:red')
ax[0,0].legend(loc='best')
ax[0,0].set_title('Ideal Case')

ax[0,1].plot(mindx12-np.mean(mindx12), '--',  color='tab:blue')
ax[0,1].plot(mindy12-np.mean(mindy12),label='Camera 1', color='tab:blue')
ax[0,1].plot(mindx22-np.mean(mindx22), '--', color='tab:green')
ax[0,1].plot(mindy22-np.mean(mindy22), label='Camera 2', color='tab:green')
ax[0,1].plot(mindx32-np.mean(mindx32), '--', color='tab:red')
ax[0,1].plot(mindy32-np.mean(mindy32), label='Camera 3', color='tab:red')
ax[0,1].set_title('Noisy Case')

ax[1,0].plot(mindx13-np.mean(mindx13), '--',  color='tab:blue')
ax[1,0].plot(mindy13-np.mean(mindy13),label='Camera 1', color='tab:blue')
ax[1,0].plot(mindx23-np.mean(mindx23), '--', color='tab:green')
ax[1,0].plot(mindy23-np.mean(mindy23), label='Camera 2', color='tab:green')
ax[1,0].plot(mindx33-np.mean(mindx33), '--', color='tab:red')
ax[1,0].plot(mindy33-np.mean(mindy33), label='Camera 3', color='tab:red')
ax[1,0].set_title('Horizontal Displacement Case')

ax[1,1].plot(mindx14-np.mean(mindx14), '--',  color='tab:blue')
ax[1,1].plot(mindy14-np.mean(mindy14),label='Camera 1', color='tab:blue')
ax[1,1].plot(mindx24-np.mean(mindx24), '--', color='tab:green')
ax[1,1].plot(mindy24-np.mean(mindy24), label='Camera 2', color='tab:green')
ax[1,1].plot(mindx34-np.mean(mindx34), '--', color='tab:red')
ax[1,1].plot(mindy34-np.mean(mindy34), label='Camera 3', color='tab:red')
ax[1,1].set_title('Horizontal Displacement and Rotation Case')
fig.savefig('plots/positions.png')


fig, ax = plt.subplots(figsize=(7,8.5), nrows=4, sharex=True)
ax[0].plot(pca1[0,:], label='Mode 1', color='tab:purple')
ax[0].plot(pca1[1,:], label='Mode 2', color='tab:cyan')
ax[0].plot(pca1[2,:], label='Mode 3', color='tab:gray')
ax[0].set_title('Ideal Case')
ax[0].legend(loc='best')
ax[0].set_ylabel('Position', fontsize=14)

ax[1].plot(pca2[0,:], label='Mode 1', color='tab:purple')
ax[1].plot(pca2[1,:], label='Mode 2', color='tab:cyan')
ax[1].plot(pca2[2,:], label='Mode 3', color='tab:gray')
ax[1].set_title('Noisy Case')
#ax[1].legend(loc='best')
ax[1].set_ylabel('Position', fontsize=14)

ax[2].plot(pca3[0,:], label='Mode 1', color='tab:purple')
ax[2].plot(pca3[1,:], label='Mode 2', color='tab:cyan')
ax[2].plot(pca3[2,:], label='Mode 3', color='tab:gray')
ax[2].set_title('Horizontal Displacement Case')
#ax[2].legend(loc='best')
ax[2].set_ylabel('Position', fontsize=14)

ax[3].plot(pca4[0,:], label='Mode 1', color='tab:purple')
ax[3].plot(pca4[1,:], label='Mode 2', color='tab:cyan')
ax[3].plot(pca4[2,:], label='Mode 3', color='tab:gray')
ax[3].set_title('Horizontal Displacement and Rotation Case')
ax[3].set_xlabel('Frame', fontsize=14)
ax[3].set_ylabel('Position', fontsize=14)
#ax[3].legend(loc='best')
fig.savefig('plots/PCA_comparison_modes.png')


fig, ax = plt.subplots()
ax.plot(np.arange(1,7,1), percentage1, '-^', label='Ideal Case')
ax.plot(np.arange(1,7,1), percentage2, '-^', label='Noisy Case')
ax.plot(np.arange(1,7,1), percentage3, '-^', label='Horizontal Displacement Case')
ax.plot(np.arange(1,7,1), percentage4, '-^', label='Horizontal Displacement and Rotation Case')
ax.legend(loc='best')
ax.set_xlabel('Mode')
ax.set_ylabel('Proportion of Variance')
fig.savefig('plots/variance.png')






