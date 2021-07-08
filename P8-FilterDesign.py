from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#1 - LowPass
num_taps = 51 # use odd number
cut_off = 3000 # HZ
sample_rate = 32000 # HZ
h = signal.firwin(num_taps, cut_off, nyq=sample_rate/2) # LowPass
# h = signal.firwin(num_taps, cut_off, nyq=sample_rate/2, pass_zero=False) HighPass
# h = signal.firwin(n, cutoff = [3000, 5000], , window = "hanning", nyq=sample_rate/2, window = 'blackmanharris', pass_zero = False) PassBand
# real taps = symmetric filter
# from -cutoff to cutoff (roughly)
# increase num_taps todecrease the transition ...
# ... zone but cmputation might be expensive
plt.figure() # impulse response
plt.plot(h, '.-') # taps in time domain
# plot the frequency response
H = np.abs(np.fft.fft(h, 1024)) # take the 1024-point FFT and magnitude
H = np.fft.fftshift(H) # make 0 Hz in the center
w = np.linspace(-sample_rate/2, sample_rate/2, len(H)) # x axis
plt.figure()
plt.plot(w, H, '.-')
plt.show()

#2 - PassBand
# Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
f0 = 10e3 # amount we will shift
Ts = 1.0/sample_rate # sample period
t = np.arange(0.0, Ts*len(h), Ts) # time vector. args are (start, stop, step)
exponential = np.exp(2j*np.pi*f0*t) # this is essentially a complex sine wave
h_band_pass = h * exponential # do the shift
# plot impulse response
plt.figure()
plt.plot(np.real(h_band_pass), '.-')
plt.plot(np.imag(h_band_pass), '.-')
plt.legend(['real', 'imag'], loc=1)
# plot the frequency response
H = np.abs(np.fft.fft(h_band_pass, 1024)) # take the 1024-point FFT and magnitude
H = np.fft.fftshift(H) # make 0 Hz in the center
w = np.linspace(-sample_rate/2, sample_rate/2, len(H)) # x axis
plt.figure()
plt.plot(w, H, '.-')
plt.xlabel('Frequency [Hz]')
plt.show()

#Example
H = np.hstack((np.zeros(20), np.arange(10)/10, np.zeros(20))) # 50 points used
w = np.linspace(-0.5, 0.5, 50)
plt.figure()
plt.plot(w, H, '.-')
h = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H)))
plt.figure()
plt.plot(np.real(h))
plt.plot(np.imag(h))
plt.legend(['real','imag'], loc=1)
plt.show()

H_fft = np.fft.fftshift(np.abs(np.fft.fft(h, 1024)))
plt.figure()
plt.plot(H_fft) # actual response
plt.show() 
# see that frequency response doesn't deacy to 0
# option 1: windowing, look at scipy.signal.firwin for list of parameters
window = np.hamming(len(h))
h = h * window
H_fft = np.fft.fftshift(np.abs(np.fft.fft(h, 1024)))
plt.figure()
plt.plot(H_fft) # actual response
plt.show() 

# option 2 : increase number of points in original function from 50 to e.g. 500
H = np.hstack((np.zeros(200), np.arange(100)/100, np.zeros(200))) # 50 points used
w = np.linspace(-0.5, 0.5, 500)
plt.figure()
plt.plot(w, H, '.-')
h = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H)))
plt.figure()
plt.plot(np.real(h))
plt.plot(np.imag(h))
plt.legend(['real','imag'], loc=1)
plt.show()
H_fft = np.fft.fftshift(np.abs(np.fft.fft(h, 1024)))
plt.figure()
plt.plot(H_fft) # actual response
plt.show()
