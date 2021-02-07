import numpy as np
import matplotlib.pyplot as plt
import os
#import librosa 
import scipy.io as sio
import scipy.io.wavfile as wav
#plt.ion()

# definte data path and filename
datapath = '../data/'
floydfile = 'floyd.mat'
gnrfile = 'GNR.mat'

#floyd, floyd_sr = librosa.load(os.path.join(datapath, floydfile), sr=None)

floyd_fs = 44100.0
gnr_fs = 48000.0

#fs, floyd_test = sio.wavfile.read(os.path.join(datapath, 'Floyd.wav'))
floyd = sio.loadmat(os.path.join(datapath, floydfile))['y']/floyd_fs
floyd = np.squeeze(floyd)
floyd = floyd[:-1]
Tf = len(floyd)/floyd_fs 
time_floyd = np.linspace(0, Tf, len(floyd))

gnr = sio.loadmat(os.path.join(datapath, gnrfile))['y']/gnr_fs
gnr = np.squeeze(gnr)
Tgnr = len(gnr)/gnr_fs
time_gnr = np.linspace(0, Tgnr, len(gnr))

# define time and frequency domains 
n = len(floyd)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tf))*np.append(k_pos, k_neg)
floyd_ks = np.fft.fftshift(k)

n = len(gnr)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tgnr))*np.append(k_pos, k_neg)
gnr_ks = np.fft.fftshift(k)


if 1:
	plot = 'floyd'
	if plot=='floyd':
		fig, ax = plt.subplots()
		ax.plot(time_floyd, floyd)
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Amplitude')
	if plot=='gnr':
		fig, ax = plt.subplots()
		ax.plot(time_gnr, gnr)
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Amplitude')

def filter(a, t, tau):
	return np.exp(-a*(t-tau)**2)

a = 1
floyd_tau = np.arange(0, Tf, 1)
floyd_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_floyd, floyd_tau[i])*floyd))) for i in range(len(floyd_tau))])


gnr_tau = np.arange(0, Tgnr, 1)
gnr_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_gnr, gnr_tau[i])*gnr))) for i in range(len(gnr_tau))])

[floyd_Ks, floyd_Tau] = np.meshgrid(floyd_ks, floyd_tau)
[gnr_Ks, gnr_Tau] = np.meshgrid(gnr_ks, gnr_tau)


if 1:
	n = 10
	fig, ax = plt.subplots()
	p = ax.pcolormesh(floyd_Tau[:,::n], floyd_Ks[:,::n], np.log(floyd_spec[:,::n]), shading='gouraud')
	fig.colorbar(p, ax=ax)
	ax.set_ylabel('Frequency (k)')
	ax.set_xlabel('Time (s)')
	fig.savefig('floyd_spec.png')

	fig, ax = plt.subplots()
	p = ax.pcolormesh(gnr_Tau[:,::n], gnr_Ks[:,::n], np.log(gnr_spec[:,::n]), shading='gouraud')
	fig.colorbar(p, ax=ax)
	ax.set_ylabel('Frequency (k)')
	ax.set_xlabel('Time (s)')
	fig.savefig('gnr_spec.png')





