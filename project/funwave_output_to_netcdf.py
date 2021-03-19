import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import pandas as pd 
import xarray as xr

rundir = 'jonswap_test'
fdir = os.path.join('/data2','enuss','funwave_lab_setup',rundir,'output')
savedir = os.path.join('/data2', 'enuss', 'funwave_lab_setup', rundir, 'postprocessing', 'compiled_output')
if not os.path.exists(savedir):
    os.makedirs(savedir)

depFile = os.path.join(fdir,'dep.out')
dep = np.loadtxt(depFile)
[n,m] = dep.shape

# discretization
dx = 0.05
dy = 0.1

# x and y field vectors 
x = np.arange(0,m*dx,dx)
y = np.arange(0,n*dy,dy)

def funwave_to_netcdf(flist, x, y, time, fpath, name):
	var = np.zeros((fnum,len(y),len(x)))
	for i in range(fnum):
		var_i = pd.read_csv(os.path.join(fdir,flist[i]), header=None, delim_whitespace=True)
		var_i = np.asarray(var_i)
		var[i,:,:] = var_i
		del var_i
	dim = ["time", "y", "x"]
	coords = [time, y, x]
	dat = xr.DataArray(var, coords=coords, dims=dim, name=name)
	dat.to_netcdf(fpath)

##### ETA #####
flist = [file for file in glob.glob(os.path.join(fdir,'eta_*'))]
fnum = len(flist)
time = np.arange(0,fnum)
fpath = os.path.join(savedir, 'eta.nc')
funwave_to_netcdf(flist, x, y, time, fpath, 'eta')

##### U #####
flist = [file for file in glob.glob(os.path.join(fdir, 'u_*'))]
fnum = len(flist)
time = np.arange(0,fnum)
fpath = os.path.join(savedir, 'u.nc')
funwave_to_netcdf(flist, x, y, time, fpath, 'u')

##### V #####
flist = [file for file in glob.glob(os.path.join(fdir, 'v_*'))]
fnum = len(flist)
time = np.arange(0,fnum)
fpath = os.path.join(savedir, 'v.nc')
funwave_to_netcdf(flist, x, y, time, fpath, 'v')

##### MASK #####
flist = [file for file in glob.glob(os.path.join(fdir, 'mask_*'))]
fnum = len(flist)
time = np.arange(0,fnum)
fpath = os.path.join(savedir, 'mask.nc')
funwave_to_netcdf(flist, x, y, time, fpath, 'mask')






