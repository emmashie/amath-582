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
#start = int(floyd_fs*28)
#end = int(floyd_fs*44)
#floyd = floyd[start:end]
Tf = len(floyd)/floyd_fs 
time_floyd = np.linspace(0, Tf, len(floyd))

# clip
start = int(floyd_fs*0)
end = int(floyd_fs*15)
floyd_c = floyd[start:end]
Tf_c = len(floyd_c)/floyd_fs 
time_floyd_c = np.linspace(start/floyd_fs, end/floyd_fs, len(floyd_c))


# define time and frequency domains 
n = len(floyd)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tf))*np.append(k_pos, k_neg)
floyd_ks = np.fft.fftshift(k)

# clip 
n = len(floyd_c)
k_pos = np.arange(0, n/2)
k_neg = np.arange(-n/2, 0)
k = (1/(Tf_c))*np.append(k_pos, k_neg)
floyd_ks_c = np.fft.fftshift(k)

def filter(a, t, tau):
	return np.exp(-a*(t-tau)**2)

def boxcar(k, ulim, blim):
	bx = np.zeros(len(k))
	bx[np.where((k>blim)&(k<ulim))[0]] = 1
	return bx

# filter data prior to building spectogram
#floyd = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(floyd))*boxcar(floyd_ks, 150, 60)))
#floyd_c = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(floyd_c))*boxcar(floyd_ks_c, 150, 60)))
floyd = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(floyd))*boxcar(floyd_ks, 700, 130)))
floyd_c = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(floyd_c))*boxcar(floyd_ks_c, 700, 130)))


a = 100
floyd_tau = np.arange(0, Tf, Tf/100)
floyd_spec = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_floyd, floyd_tau[i])*floyd))) for i in range(len(floyd_tau))])
# clip
floyd_tau_c = np.arange(0, Tf_c, Tf_c/60)
floyd_spec_c = np.asarray([np.fft.fftshift(np.abs(np.fft.fft(filter(a, time_floyd_c, floyd_tau_c[i])*floyd_c))) for i in range(len(floyd_tau_c))])

[floyd_Tau, floyd_Ks] = np.meshgrid(floyd_tau, floyd_ks)

#clip
[floyd_Tau_c, floyd_Ks_c] = np.meshgrid(floyd_tau_c, floyd_ks_c)


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
blim = 12*2 + 2
ulim = 12*3 + 4 - 3
# guitar solo
#blim = 12*2-2 #a sharp below the staff
blim = 12*3
ulim = 12*5 + 3 +1
ind = np.where((floyd_ks>=np.min(freq[blim:ulim]))&(floyd_ks<=np.max(freq[blim:ulim])))
Tau = np.squeeze(floyd_Tau[ind,:])
Ks = np.squeeze(floyd_Ks[ind,:])
spec = np.squeeze(floyd_spec.T[ind,:])

# clip
ind = np.where((floyd_ks_c>=np.min(freq[blim:ulim]))&(floyd_ks_c<=np.max(freq[blim:ulim])))
Tau_c = np.squeeze(floyd_Tau_c[ind,:])
Ks_c = np.squeeze(floyd_Ks_c[ind,:])
spec_c = np.squeeze(floyd_spec_c.T[ind,:])

#spec_rmean = np.asarray([np.convolve(spec[:,i], np.ones(int(len(Ks)/30))/int(len(Ks)/30), mode='same') for i in range(len(floyd_tau))]).T

## panel plot
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,7), constrained_layout=True)
gs = fig.add_gridspec(3,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,-1])
ax3 = fig.add_subplot(gs[1:,:])

ax1.plot(time_floyd, filter(a, time_floyd, floyd_tau[10]), color='slateblue', linewidth=2)
ax1.set_xticks(np.arange(0, np.floor(Tf),5))
ax1.set_xticklabels(np.arange(0, np.floor(Tf),5), fontsize=12)
ax1.set_xlim((0,15))
ax1.set_xlabel('Time (s)')
print('plotted ax1')

#blim = 12*2 + 2
#ulim = 12*3 + 4 - 3
#ind = [5,10,12,17,21,22]
#freqlabels = [freq[blim:ulim][ind[i]] for i in range(len(ind))]
#labels = [notes[blim:ulim][ind[i]] for i in range(len(ind))]
freqlabels = freq[blim:ulim]
labels = notes[blim:ulim]
#p = ax2.pcolorfast(floyd_tau_c, floyd_ks_c, floyd_spec_c.T, cmap='cool')
p = ax2.pcolormesh(Tau_c, Ks_c, spec_c, shading='gouraud', cmap='cool')
p.set_clim((0.0003, np.max(spec_c)))
#p.set_clim((0.0007, np.max(spec_c)))
ax2.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax2.set_yticks(freqlabels[::3])
ax2.set_yticklabels(labels[::3], fontsize=14)
ax2.set_xticks(np.arange(0,np.floor(Tf_c),3))
ax2.set_xticklabels(np.arange(0,np.floor(Tf_c),3), fontsize=12)
ax2.set_xlabel('Time (s)', fontsize=12)
fig.colorbar(p, ax=ax2)
print('plotted ax2')

p = ax3.pcolormesh(Tau, Ks, spec, shading='gouraud', cmap='cool')
p.set_clim((0.0003, np.max(spec)))
#p.set_clim((0.0007, np.max(spec_c)))
ax3.set_ylim((np.min(freq[blim:ulim]), np.max(freq[blim:ulim])))
ax3.set_yticks(freq[blim:ulim][::2])
ax3.set_yticklabels(notes[blim:ulim][::2], fontsize=12)
ax3.set_xticks(np.arange(0,np.floor(Tf),5))
ax3.set_xticklabels(np.arange(0,np.floor(Tf),5), fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
fig.colorbar(p, ax=ax3)
print('plotted ax3')

fig.savefig('floyd_filter_fb_full_guitar_bxfiltered_log.png')
print('saved figure')







