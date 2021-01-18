from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.signal.windows import kaiser
import sounddevice as sd
from sys import argv
from scipy.special import seterr
from scipy.io.wavfile import WavFileWarning

seterr(all='ignore')

fn = argv[1] if len(argv) > 1 else "trainall/070_M.wav"

try:
    sr, a = wavfile.read(fn, )
except WavFileWarning:
    pass
if len(a.shape)>1 and a.shape[1] > 1:
    a = a[:,0]

a1 = a #* kaiser(len(a), 30)

def proc(a):
    ff = np.abs(fft(a))
    n = len(ff)
    ff = ff[:n//2]
    ff /= n/2
    ff[0] /= 2
    x = np.arange(len(ff))/n*sr
    m = np.where(x>2000)[0][0]
    return ff, x, m


f1, x, m = proc(a)
#print(a.shape, a.dtype)

f2, _, _ = proc(a1)
f2[:np.where(x>80)[0][0]] = 0
f2 = f2[:m]
#f2 = f2[:np.where(x>800)[0][0]]

#f2 /= np.max(f2)

f3 = f2.copy()


for n in range(2, 5):
    d = decimate(f2, n)
    f3[:len(d)] *= d

f3 = f3[:m]

base_f = x[np.argmax(f3[:np.where(x>350)[0][0]])]


print("M" if base_f < 175 else "F", base_f)

#exit(0)

plt.subplot(311)
plt.stem(x[:m], f1[:m], markerfmt=' ')

plt.subplot(312)
plt.stem(x[:m], f2[:], markerfmt=' ')

plt.subplot(313)
plt.stem(x[:m], f3[:], markerfmt=' ')
plt.show()

#sd.play(a, sr, blocking=True)
