from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import decimate
from scipy.signal.windows import kaiser
import sounddevice as sd
from sys import argv
import sys
import os
import signal

# suppress warnings
sys.stderr = open(os.devnull, "w")

fn = argv[1] if len(argv) > 1 else "trainall/001_K.wav"

sr, a = wavfile.read(fn, )

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

f1[:np.where(x>80)[0][0]] = 0
f1 = f1[:m]

f3 = f1.copy()


for n in range(2, 5):
    d = decimate(f1, n)
    f3[:len(d)] *= d

f3 = f3[:m]

base_f = x[np.argmax(f3[:np.where(x>350)[0][0]])]


print("M" if base_f < 175 else "K", base_f)

# exit(0)

# plt.subplot(211)
# plt.stem(x[:m], f1[:m], markerfmt=' ')

# plt.subplot(212)
# plt.stem(x[:m], f3[:], markerfmt=' ')
# plt.show()

# sd.play(a, sr, blocking=True)
