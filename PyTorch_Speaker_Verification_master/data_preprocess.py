"""
Copyright (c) 2019, HarryVolek
All rights reserved.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import pathlib
import random

import librosa
import numpy as np
from hparam import hparam as hp
import pywt
from scipy.fftpack import dct
import math
from python_speech_features import mfcc, logfbank, delta,base
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow


#new imports:
#import jax
#import jax.numpy as jnp
#import jaxlib
import matplotlib.pyplot as plt
#import lop
#import cr-sparse
#import cr.sparse as crs
#from cr.sparse import lop
#from cr.sparse import metrics
import metrics
import skimage.data
#from cr.sparse.dsp import time_values
#import time_values

# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """

    transform_kind = "Wavelet1.1"
    print("start text independent utterance feature extraction")
    os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d" % total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..." % i)
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = None
                        if transform_kind == "Fourier":
                            #global S
                            S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                                  win_length=int(hp.data.window * sr),
                                                  hop_length=int(hp.data.hop * sr))  # OUR TRANSFORM
                            S = np.abs(S) ** 2  # OUR TRANSFORM

                            basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels) # OUR TRANSFORM
                            S = np.log10(np.dot(basis, S) + 1e-6)  # log mel spectrogram of utterances
                            print(len(S))
                            print(type(S))
                        elif transform_kind == "Wavelet":
                            """
                            keep_positive_freq = False
                            freqs = librosa.cqt_frequencies(n_bins=40, fmin=librosa.note_to_hz('C1'))
                            basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=22050)
                            S = np.abs(np.fft.fftn(basis, axes=[-1]))
                            # Keep only the positive frequencies
                            if keep_positive_freq:
                                S = S[:, :(1 + S.shape[1] // 2)]"""
                            #fs = 1000.
                            #T = 2
                            #t = time_values(fs, T)
                            #n = t.size
                            #DWT_op = lop.dwt(n, wavelet='db', level=1)
                            #global S
                            approx,detail = pywt.dwt(utter_part, 'db8')
                            S = approx
                            k = 0
                            j = 0
                            feature = np.ndarray(shape=(40, 180))
                            app1 = np.ndarray(shape=(20, 180))
                            det1 = np.ndarray(shape=(20, 180))
                            val = math.floor(len(approx) / 180)
                            while k < len(approx) and j < 20:
                                app1[j][:] = approx[k: k + 180]
                                det1[j][:] = detail[k: k + 180]
                                k += val
                                j += 1
                            #app1 = app1.transpose()
                            #det1 = det1.transpose()
                            feature[:20, :] = app1
                            feature[20:, :] = det1
                            feature = base.hz2mel(feature)
                            feature = log_power_spectrum(feature)
                            #feature = dct(feature, type=2, axis=1, norm='ortho')
                            feature = dct(feature, type=2, axis=0, norm='forward')
                            #print(type(S))
                            S = feature

                            """S,_ = pywt.cwt(utter_part, np.arange(1, 258), 'morl')
                         
                            S = np.abs(S) ** 2  # OUR TRANSFORM

                            basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft,
                                                        n_mels=hp.data.nmels)  # OUR TRANSFORM
                            S = np.log10(np.dot(basis, S) + 1e-6)  # log mel spectrogram of utterances"""
                        elif transform_kind == "Wavelet1.1":
                            n_fft = hp.data.nfft
                            vector_length = 257
                            window = int(hp.data.window * sr)
                            hop_length = int(hp.data.hop * sr)
                            matrix = np.zeros((vector_length, int(math.ceil(len(utter_part) / hop_length))),
                                              dtype='float32')
                            j = 0
                            for start in range(0, len(utter_part), hop_length):
                                move = start + (1 + n_fft / 2) if start + (1 + n_fft / 2) < len(utter_part) else len(
                                    utter_part)
                                move = int(move)
                                part = utter_part[start: move]
                                if len(part) == 1 + n_fft / 2:
                                    part = pywt.dwt(part, 'db8')[0]
                                    part *= 10000
                                    part = adjust_the_vector(part, vector_length)
                                    matrix[:, j] = np.array(part)
                                j += 1
                            S = matrix
                            # freqs = librosa.cqt_frequencies(n_bins=40, fmin=librosa.note_to_hz('C1'))
                            # basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=hp.data.sr)
                            basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                            S = np.log10(np.dot(basis, S) + 1e-6)
                            for row_index in range(len(S)):
                                for col_index in range(len(S[row_index])):
                                    if np.isnan(S[row_index, col_index]):
                                        S[row_index, col_index] = 0

                        else:
                            raise Exception("Please choose a legal transform")

                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        #print(utterances_spec.shape)
        if i < train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)
        """ old code:
        if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join(hp.data.train_path, "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)
        """


def log_power_spectrum(feature):
    feature = np.absolute(feature) ** 2
    #feature = np.log10(feature)
    for i in range(40):
        for j in range(180):
            if feature[i,j] > 0:
                np.log10(feature[i,j])
    return feature

def adjust_the_vector(vector, length):
    if len(vector) == length:
        return vector

    super_vector = vector[:]
    while True:
        if len(super_vector) >= length:

            return super_vector[:length]

        else:
            super_vector = np.concatenate((super_vector, vector), axis=0)

if __name__ == "__main__":
    save_spectrogram_tisv()
