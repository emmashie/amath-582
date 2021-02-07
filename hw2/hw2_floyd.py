import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import scipy.io.wavfile as wav
#plt.ion()
plt.style.use('ggplot')

# definte data path and filename
datapath = '../data/'
floydfile = 'floyd.mat'

floyd_fs = 44100.0

floyd = sio.loadmat(os.path.join(datapath, floydfile))['y']/floyd_fs
floyd = np.squeeze(floyd)[:-1] #remove last time-step to keep even time series
#first_bit = int(floyd_fs*15)
#floyd = floyd[:first_bit]
Tf = len(floyd)/floyd_fs 
time_floyd = np.linspace(0, Tf, len(floyd))

# define time and frequency domains 
n = len(floyd)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tf))*np.append(k_pos, k_neg)
floyd_ks = np.fft.fftshift(k)

def filter(a, t, tau):
	return np.exp(-a*(t-tau)**2)

a = 75
floyd_tau = np.arange(0, Tf, Tf/100)
floyd_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_floyd, floyd_tau[i])*floyd))) for i in range(len(floyd_tau))])

[floyd_Tau, floyd_Ks] = np.meshgrid(floyd_tau, floyd_ks)

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
	ax.set_ylabel('Frequency (Hz)', fontsize=14)
	ax.set_ylim((0,4000))		
	ax.set_yticks(np.arange(0, 4000, 500))	
	ax.set_xticklabels(notes, fontsize=14)
	ax.set_yticklabels(np.arange(0, 4000, 500), fontsize=14)
	fig.tight_layout()	
	fig.savefig('notes2freq.png')

# bass 
blim = 12*2
ulim = 12*3 + 4
# guitar solo
blim = 12*3 
ulim = 12*5 + 3 + 5
ind = np.where((floyd_ks>=np.min(freq[blim:ulim]))&(floyd_ks<=np.max(freq[blim:ulim])))
Tau = np.squeeze(floyd_Tau[ind,:])
Ks = np.squeeze(floyd_Ks[ind,:])
spec = np.squeeze(floyd_spec.T[ind,:])

fig, ax = plt.subplots(figsize=(15,8))
#p = ax.pcolormesh(floyd_Tau, floyd_Ks, floyd_spec.T, shading='gouraud', cmap='cool')
p = ax.pcolormesh(Tau, Ks, spec, shading='gouraud', cmap='cool')
ax.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax.set_yticks(freq[blim:ulim])
ax.set_yticklabels(notes[blim:ulim], fontsize=14)
ax.set_xticks(np.arange(0,np.floor(Tf),3))
ax.set_xticklabels(np.arange(0,np.floor(Tf),3), fontsize=15)
ax.set_xlabel('Time (s)', fontsize=14)
fig.colorbar(p, ax=ax)
fig.tight_layout()
fig.savefig('floyd_spec_a75_guitar.png')











