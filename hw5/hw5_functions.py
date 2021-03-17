#!/usr/bin/env python
""" functions used for AMATH 582 hw3
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy import linalg 

def norm(x):
	return x/np.nanmax(x)

def rgb2grey(rgb):
    """ convert rgb data to greyscale 
    """
    grey = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]   
    return norm(grey)

def video_svd(video, duration):
	time = np.linspace(0, duration, len(video))
	dt = np.diff(time)[0]
	video_grey = np.asarray([rgb2grey(video[i,:,:,:]) for i in range(len(video))])
	[nimgs, nx, ny] = np.shape(video_grey)
	X = np.reshape(video_grey, (nimgs, nx*ny), order='F').T 
	X1 = X[:,:-1]
	X2 = X[:,1:]
	U, Sdiag, VH = np.linalg.svd(X1, full_matrices=False) 
	V = VH.T
	return X, X1, X2, U, Sdiag, V, nimgs, nx, ny 

def mode_appoximation(X2, U, Sdiag, V, modes):
	Atilde = U[:,:modes].T@X2@V[:,:modes]@np.diag(1/Sdiag[:modes])
	[D, eV] = linalg.eig(Atilde)
	Phi = X2@V[:,:modes]@np.diag(1/Sdiag[:modes])@eV
	return D, Phi

def reconstruct_DMD(D, X, X1, Phi, time, nx, ny, nimgs):
	omega = np.log(D)/np.diff(time)[0]
	b = np.linalg.lstsq(Phi, X1[:,0], rcond=None)[0]
	xmodes = np.asarray([b*np.exp(omega*time[i]) for i in range(len(time))])
	X_lr = Phi@xmodes.T
	X_sp = X - np.abs(X_lr)
	R = X - np.abs(X_lr)
	R[R>=0] = 0
	X_lr_R = R + np.abs(X_lr)
	X_sp_R = X_sp - R
	foreground = np.reshape(X_sp_R, (nx, ny, nimgs), order='F')
	background = np.reshape(X_lr_R, (nx, ny, nimgs), order='F')
	return foreground, background

