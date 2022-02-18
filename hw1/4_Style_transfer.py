import librosa
import numpy as np

# read the file and transform them into Ma, Mb and Mc
filename = "./Audio/Synth.wav"
audio, sr = librosa.load(filename, sr=16000)
Ma = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=1024)
Ma_abs = abs(Ma)

filename = "./Audio/Piano.wav"
audio, sr = librosa.load(filename, sr=16000)
Mb = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=1024)
Mb_abs = abs(Mb)

filename = "./Audio/BlindingLights.wav"
audio, sr = librosa.load(filename, sr=16000)
Mc = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=1024)
Mc_abs = abs(Mc)
phase_c = Mc/(Mc_abs + 2.2204e-16)
Ma.shape, Mb.shape, Mc.shape

# find the T to make TMa be nearest to Mb
# TMa = Mb
# T = Mb right_inv(Ma)
# right_pinv(x) = x.T*(x.T x)^-1
right_pinv_Ma = np.linalg.pinv(Ma_abs)
T = np.dot(Mb_abs, right_pinv_Ma)
T.shape

# report the error between TMa and Mb
error_T = np.matmul(T, Ma_abs) - Mb_abs
value = np.square(np.linalg.norm(error_T))
print(value)

np.savetxt('./result/problem3t.csv', T, delimiter=',')

Md = np.dot(T, Mc_abs)
np.savetxt('./result/problem3md.csv', Md, delimiter=',')

signal_hat = librosa.istft(np.multiply(Md, phase_c), hop_length=256, center=False, win_length=1024)
librosa.output.write_wav("./result/problem3.wav", signal_hat, sr=16000)
