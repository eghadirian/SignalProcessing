import matplotlib.pyplot as plt
import numpy as np
import pywt

t = np.arange(0,20,0.2)
signal = np.sin(t)+np.sin(10*t)+np.sin(100*t)+np.sin(1000*t)

plt.figure()
(cA1, cD1) = pywt.dwt(signal, 'db2', 'smooth')
reconstructed_signal = pywt.idwt(cA1, cD1, 'db2', 'smooth')
plt.plot(signal, label='signal')
plt.plot(reconstructed_signal, label='reconstructed signal', linestyle='--')
plt.legend(loc='upper left')

plt.figure()
coeffs = pywt.wavedec(signal, 'db2', level=8)
reconstructed_signal = pywt.waverec(coeffs, 'db2')
plt.plot(signal[:1000], label='signal')
plt.plot(reconstructed_signal[:1000], label='reconstructed signal', linestyle='--')
plt.legend(loc='upper left')

plt.figure()
def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal
plt.plot(signal, color="b", alpha=0.5, label='original signal')
rec = lowpassfilter(signal, 0.4)
plt.plot(rec, 'k', label='DWT smoothing', linewidth=2)

plt.legend()
plt.show()