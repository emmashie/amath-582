#!/usr/bin/env python
""" functions used for AMATH 582 hw3
"""
import numpy as np 
import matplotlib.pyplot as plt

def norm(x):
	return x/np.nanmax(x)

def rgb2grey(rgb):
    """ convert rgb data to greyscale 
    """
    grey = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]   
    return norm(grey)

def plotim_w_max(vid, indx, indy, avg_indx, avg_indy, path):
	fig, ax = plt.subplots()
	p = ax.pcolormesh(vid[::-1,:], cmap='Greys')
	#fig.colorbar(p, ax=ax)
	ax.scatter(indx, indy, color='tab:blue')
	ax.scatter(avg_indx, avg_indy, color='tab:red')
	ax.get_xaxis().set_visible(False)	
	ax.get_yaxis().set_visible(False)
	fig.savefig(path)
	plt.close()

def find_ind_and_plot(vid, path, xmin=250, xmax=400, ymin=0, ymax=500, plot=True, restricty=False):
	sindx = []
	sindy = []
	for i in range(len(vid)):
		ind = np.where(vid[i,:,:]>0.95)
		subindx = np.where((ind[-1]<xmax)&(ind[-1]>xmin))
		if restricty==True:
			subindy = np.where((ind[0]<ymax)&(ind[0]>ymin))
			subind = np.intersect1d(subindx, subindy)
		if restricty==False:
			subind=subindx
		indx = ind[-1][subind]
		indy = len(vid[i,:,:]) - ind[0][subind]
		avg_indx = int(np.mean(indx))
		avg_indy = int(np.mean(indy))
		sindx.append(avg_indx)
		sindy.append(avg_indy)
		if plot==True:
			plotim_w_max(vid[i,:,:], indx, indy, avg_indx, avg_indy, path+'%05d' % i)
	return sindx, sindy

def plot_positions(mindx11, mindy11, mindx21, mindy21, mindx31, mindy31, test_num):
	fig, ax = plt.subplots(figsize=(8,5))
	ax.plot(mindx11, '--',  color='tab:blue')
	ax.plot(mindy11,label='Camera 1', color='tab:blue')
	ax.plot(mindx21, '--', color='tab:green')
	ax.plot(mindy21, label='Camera 2', color='tab:green')
	ax.plot(mindx31, '--', color='tab:red')
	ax.plot(mindy31, label='Camera 3', color='tab:red')
	ax.legend(loc='best')
	ax.set_ylabel('X & Y position', fontsize=16)
	ax.set_xlabel('Frame', fontsize=16)
	fig.savefig('plots/test%s_positions.png' % test_num)	