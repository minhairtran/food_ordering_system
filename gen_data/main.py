from playsound import playsound
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import torchaudio
import argparse
import torch
import json
import os

import augment

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', type=str, default='fpt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args

def load_script(args):
    script_json = os.path.join(args.data_dir, 'script.json')
    for meta in open(script_json):
        yield json.loads(meta)

def main():

    args = get_args()

    # mel spectrogram
    kwargs = {
        'n_fft': 512,
        'n_mels': 128,
    }
    wav_to_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    # spectrogram augmentation
    kwargs = {
        'freq_mask_param': 14,
        'time_mask_param': 20,
    }
    spec_augment = augment.SpectrogramAugmentation(**kwargs)

    # inverse melscale to linear scale
    kwargs = {
        'n_stft': 512 // 2 + 1,
        'n_mels': 128,
    }
    inverse_melscale = torchaudio.transforms.InverseMelScale(**kwargs)

    # convert spectrogram to waveform
    kwargs = {
        'n_fft': 512,
    }
    spec_to_wav = torchaudio.transforms.GriffinLim(**kwargs)

    scripts = load_script(args)

    for meta in scripts:

        dirname = os.path.join(args.data_dir, args.api, meta['folder'])
        wav_path = os.path.join(dirname, meta['filename'] + '.wav')
        mp3_path = os.path.join(dirname, meta['filename'] + '.mp3')
        # if 'com_heo_xi_muoi' not in wav_path:
        #     continue

        if not os.path.exists(wav_path):
            cmd = f'ffmpeg -i "{mp3_path}" -ac 1 -ar 16000 "{wav_path}" > /dev/null 2>&1'
            print(cmd)
            os.system(cmd)

        fs, data = wavfile.read(wav_path)
        data = torch.Tensor(data.copy())
        data = data / data.abs().max()

        print(wav_path)
        print('playing')
        playsound(wav_path)

        print('data', data.shape)

        x = wav_to_spec(data.clone())
        print('x', x.shape)

        y = spec_augment(x.clone().unsqueeze(0)).squeeze(0)
        print('y', y.shape)

        z = inverse_melscale(y.clone())
        print('z', z.shape)

        t = spec_to_wav(z.clone())
        print('t', t.shape)

        out_path = os.path.join(args.data_dir, 'test.wav')
        # wavfile.write(out_path, fs, t.detach().numpy())
        print('playing')
        playsound(out_path)

        if args.plot:
            plt.subplot(221)
            plt.plot(data, linewidth=0.2)

            plt.subplot(223)
            plt.imshow(x, origin='lower', aspect='auto', cmap='jet')


            plt.subplot(222)
            plt.plot(t, linewidth=0.2)

            plt.subplot(224)
            plt.imshow(y, origin='lower', aspect='auto', cmap='jet')

            plt.show()

if __name__ == '__main__':
    main()
