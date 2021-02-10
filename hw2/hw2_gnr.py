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
gnr = np.squeeze(gnr)
Tgnr = len(gnr)/gnr_fs
time_gnr = np.linspace(0, Tgnr, len(gnr))
### first bar only
first_bar = 48000*2
gnr_fb = np.squeeze(gnr)[:first_bar]
Tgnr_fb = len(gnr_fb)/gnr_fs
time_gnr_fb = np.linspace(0, Tgnr_fb, len(gnr_fb))


n = len(gnr)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tgnr))*np.append(k_pos, k_neg)
gnr_ks = np.fft.fftshift(k)
## frequencies for first bar only
n = len(gnr_fb)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tgnr_fb))*np.append(k_pos, k_neg)
gnr_ks_fb = np.fft.fftshift(k)

def filter(a, t, tau):
	return np.exp(-a*(t-tau)**2)

a = 50
gnr_tau = np.arange(0, Tgnr, Tgnr/32)
gnr_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_gnr, gnr_tau[i])*gnr))) for i in range(len(gnr_tau))])
# for first bar only
gnr_tau_fb = np.arange(0, Tgnr_fb, Tgnr_fb/16)
gnr_spec_fb = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_gnr_fb, gnr_tau_fb[i])*gnr_fb))) for i in range(len(gnr_tau_fb))])

[gnr_Tau, gnr_Ks] = np.meshgrid(gnr_tau, gnr_ks)
# for first bar only 
[gnr_Tau_fb, gnr_Ks_fb] = np.meshgrid(gnr_tau_fb, gnr_ks_fb)


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


## panel plot
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,7), constrained_layout=True)
gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,-1])
ax3 = fig.add_subplot(gs[1:,:])

ax1.plot(time_gnr, filter(a, time_gnr, gnr_tau[8]), color='slateblue', linewidth=2)
ax1.set_xticks(np.arange(0, np.floor(Tgnr),3))
ax1.set_xticklabels(np.arange(0, np.floor(Tgnr),3), fontsize=12)
ax1.set_xlabel('Time (s)')

blim = 12*3 + 8
ulim = 12*5 + 3 + 5
ind = [5,10,12,17,21,22]
freqlabels = [freq[blim:ulim][ind[i]] for i in range(len(ind))]
labels = [notes[blim:ulim][ind[i]] for i in range(len(ind))]
p = ax2.pcolormesh(gnr_Tau_fb, gnr_Ks_fb, gnr_spec_fb.T, shading='gouraud', cmap='cool')
p.set_clim((0.0003, np.max(gnr_spec_fb)))
ax2.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax2.set_yticks(freqlabels)
ax2.set_yticklabels(labels, fontsize=14)
ax2.set_xticks(np.arange(0,np.floor(Tgnr_fb),1))
ax2.set_xticklabels(np.arange(0,np.floor(Tgnr_fb),1), fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
fig.colorbar(p, ax=ax2)

p = ax3.pcolormesh(gnr_Tau, gnr_Ks, gnr_spec.T, shading='gouraud', cmap='cool')
p.set_clim((0.0003, np.max(gnr_spec)))
ax3.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax3.set_yticks(freq[blim:ulim])
ax3.set_yticklabels(notes[blim:ulim], fontsize=12)
ax3.set_xticks(np.arange(0,np.floor(Tgnr),1))
ax3.set_xticklabels(np.arange(0,np.floor(Tgnr),1), fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
fig.colorbar(p, ax=ax3)
fig.savefig('gnr_filter_fb_full_threshold.png')
