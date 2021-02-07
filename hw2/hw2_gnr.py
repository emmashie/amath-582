import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import scipy.io.wavfile as wav
plt.ion()
plt.style.use('ggplot')

# definte data path and filename
datapath = '../data/'
gnrfile = 'GNR.mat'

gnr_fs = 48000.0

gnr = sio.loadmat(os.path.join(datapath, gnrfile))['y']/gnr_fs
first_bar = 48000*2
#gnr = np.squeeze(gnr)[:first_bar]
gnr = np.squeeze(gnr)[:-2] #remove last time-step to keep even time series
Tgnr = len(gnr)/gnr_fs
time_gnr = np.linspace(0, Tgnr, len(gnr))


n = len(gnr)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tgnr))*np.append(k_pos, k_neg)
gnr_ks = np.fft.fftshift(k)

def filter(a, t, tau):
	return np.exp(-a*(t-tau)**2)

a = 50
gnr_tau = np.arange(0, Tgnr, Tgnr/16)
gnr_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_gnr, gnr_tau[i])*gnr))) for i in range(len(gnr_tau))])

[gnr_Tau, gnr_Ks] = np.meshgrid(gnr_tau, gnr_ks)

notes_octave = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
freq_octave = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.5, 29.14, 30.87]

notes = []
freq = []
for i in range(8):
	for j in range(len(notes_octave)):
		notes.append(notes_octave[j])
		dum = 2**i
		freq.append(freq_octave[j]*dum)

if 0:
	fig, ax = plt.subplots()
	ax.plot(notes, freq, '^', color='black')
	ax.set_ylabel('Frequency (Hz)', fontsize=15)
	ax.set_ylim((0,4000))		
	ax.set_yticks(np.arange(0, 4000, 500))	
	ax.set_xticklabels(notes, fontsize=14)
	ax.set_yticklabels(np.arange(0, 4000, 500), fontsize=15)
	fig.tight_layout()	
	fig.savefig('notes2freq.png')

blim = 12*3 + 8
ulim = 12*5 + 3 + 5
fig, ax = plt.subplots(figsize=(15,8))
p = ax.pcolormesh(gnr_Tau, gnr_Ks, gnr_spec.T, shading='gouraud', cmap='cool')
ax.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax.set_yticks(freq[blim:ulim])
ax.set_yticklabels(notes[blim:ulim], fontsize=14)
ax.set_xticks(np.arange(0,np.floor(Tgnr),1))
ax.set_xticklabels(np.arange(0,np.floor(Tgnr),1), fontsize=14)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(p, ax=ax)
fig.tight_layout()
fig.savefig('gnr_spec_a50.png')








