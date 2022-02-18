import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (read_csv, save_csv, read_wav_and_stft, istft_and_save_wave)
from nmf import (NMF_train, KL_divergence, separate_signals)

SRC_DIR = 'data/hw3_materials/problem2/'
SAVE_DIR = 'results/problem2/'
os.makedirs(SAVE_DIR, exist_ok=True)


def func2_1(M, B_init, W_init, title=None, save_path='KL.jpg'):
    n_iter_list = [0, 50, 100, 150, 200, 250]
    loss_list = []
    for n_iter in n_iter_list:
        print('n_iter = {}'.format(n_iter))
        B, W = NMF_train(M, B_init, W_init, n_iter)
        loss = KL_divergence(M, B @ W)
        print('KL_divergence = {:.3f}'.format(loss))
        loss_list.append(loss)
        B_init, W_init = B, W
    plt.figure()
    plt.plot(n_iter_list, loss_list, marker='o')
    plt.title('KL Divergence vs. Number of Iterations')
    plt.xlabel('number of iteration')
    plt.ylabel('KL divergence')
    plt.tight_layout()
    plt.savefig(save_path)
    return B, W


def main():
    music_file = os.path.join(SRC_DIR, 'music.wav')
    Bm_init_file = os.path.join(SRC_DIR, 'Bm_init.csv')
    Wm_init_file = os.path.join(SRC_DIR, 'Wm_init.csv')

    speech_file = os.path.join(SRC_DIR, 'speech.wav')
    Bs_init_file = os.path.join(SRC_DIR, 'Bs_init.csv')
    Ws_init_file = os.path.join(SRC_DIR, 'Ws_init.csv')

    # problem2.1
    print('***** music *****')
    M_music, phase_music = read_wav_and_stft(music_file)
    Bm_init = read_csv(Bm_init_file)
    Wm_init = read_csv(Wm_init_file)
    Wm_init = Wm_init[:, :-1]
    Bm, _ = func2_1(M_music, Bm_init, Wm_init, save_path=os.path.join(SAVE_DIR, 'KL-music.jpg'))
    print('***** speech *****')
    M_speech, phase_speech = read_wav_and_stft(speech_file)
    Bs_init = read_csv(Bs_init_file)
    Ws_init = read_csv(Ws_init_file)
    Ws_init = Ws_init[:, :-1]
    Bs, _ = func2_1(M_speech, Bs_init, Ws_init, save_path=os.path.join(SAVE_DIR, 'KL-speech.jpg'))

    # problem2.2
    music_file = os.path.join(SRC_DIR, 'mixed.wav')
    M_mixed, _ = read_wav_and_stft(music_file)
    W_init = np.concatenate([Ws_init, Wm_init])

    M_speech_rec, M_music_rec = separate_signals(M_mixed, W_init, Bs, Bm, n_iter=500)
    save_csv(M_speech_rec, filename=os.path.join(SAVE_DIR, 'M_speech_rec.csv'))
    save_csv(M_music_rec, filename=os.path.join(SAVE_DIR, 'M_music_rec.csv'))

    istft_and_save_wave(M_speech_rec, phase_speech, filename=os.path.join(SAVE_DIR, 'speech_rec.wav'))
    istft_and_save_wave(M_music_rec, phase_music, filename=os.path.join(SAVE_DIR, 'music_rec.wav'))
    return


if __name__ == '__main__':
    main()
    print('done!')
