import numpy as np
import pandas as pd
import pywt
import librosa
import librosa.display
import soundfile
import matplotlib.pyplot as plt
from F2Coeffs import F2Coeffs


def read_wav_and_stft(file_name):
    audio, sr = librosa.load(file_name, sr=None)
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=256, center=False, win_length=2048)
    M = np.abs(spectrogram)
    phase = spectrogram / (M + 2.2204e-16)

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(M, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    # ax.set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return M, phase


def istft_and_save_wave(M, phase, filename='audio.wav'):
    audio = librosa.istft(M * phase, hop_length=256, center=False, win_length=2048)
    soundfile.write(filename, audio, samplerate=16000)
    return


def read_csv(filename):
    data = pd.read_csv(filename, header=None).values
    return data


def save_csv(data, filename=None):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)
    return


def F2image(F):
    Coeffs = F2Coeffs(F)
    image = pywt.waverec2(Coeffs, 'db1')
    return image


def show_and_save(image, num=None, filename='img.jpg'):
    plt.figure(num=num)
    plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.savefig(filename)
    return


def cal_rec_error(rec_image, src_img):
    assert rec_image.shape == src_img.shape
    error = rec_image - src_img
    error_total = (error * error).sum()
    error_mean = error_total / src_img.size
    return error_total, error_mean
