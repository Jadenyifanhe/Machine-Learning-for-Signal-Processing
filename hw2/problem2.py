
import os
import wave
import numpy as np
import matplotlib.pylab as plt
from fobi_ica import FOBIICA


def read_wav(filename):
    with wave.open(filename, "rb") as f:
        params = f.getparams()
        str_data = f.readframes(params.nframes)

    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data, params


def show_wav(wave_data, fs):
    time = np.arange(0, len(wave_data)) * (1.0 / fs)

    plt.plot(time, wave_data)
    plt.show()
    return


def write_wav(wave_data, name='tmp.wav', params=None):
    wave_data = wave_data.astype(np.short)
    wave_data = wave_data.tobytes()

    with wave.open(name, 'wb') as f:
        f.setparams(params)
        f.writeframes(wave_data)
    return


def ica(waves, mode='fobi'):
    model = FOBIICA()
    ratio = 10**3

    waves_transformed = model.fit_transform(waves.T).T
    waves_audible = waves_transformed * ratio
    mixing_matrix = model.mixing_
    return waves_transformed, waves_audible, mixing_matrix


def main():
    waves = []
    mix_name_tmp =  'data/hw2_materials_f21/problem2/preprocessed/mix{}.wav'
    for i in range(1, 5):
        wave_data, params = read_wav(mix_name_tmp.format(i))
        waves.append(wave_data)

    waves = np.stack(waves, axis=0)
    print('range of original waves:')
    print('min: {}'.format(waves.min(axis=1)))
    print('max: {}'.format(waves.max(axis=1)))

    # show_wav(waves[0], params.framerate)

    mode = 'fobi'
    waves_transformed, waves_audible, mixing_matrix = ica(waves, mode=mode)

    print('range of transformed waves:')
    print('min: {}'.format(waves_transformed.min(axis=1)))
    print('max: {}'.format(waves_transformed.max(axis=1)))

    print('range of amplified waves:')
    print('min: {}'.format(waves_audible.min(axis=1)))
    print('max: {}'.format(waves_audible.max(axis=1)))

    save_dir = 'results/problem2/{}_ica'.format(mode)
    os.makedirs(save_dir, exist_ok=True)
    mixing_matrix_path = os.path.join(save_dir, 'mixing_matrix.csv')
    np.savetxt(mixing_matrix_path, mixing_matrix, delimiter=',')

    src_name_tmp = 'source{}.wav'
    for i, s in enumerate(waves_audible, 1):
        name = os.path.join(save_dir, src_name_tmp.format(i))
        write_wav(s, name, params)
    return


if __name__ == '__main__':
    main()
    print('done!')

