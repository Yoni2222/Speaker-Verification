import math
import random

import pywt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
import librosa
import os
import numpy as np
from torch.utils.data import DataLoader

from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.hparam import hparam as hp
import GUI.general_screen_functions as gf
from GUI.py_screen_files.verification import Ui_verificationWin
from functools import partial
import torch
import torch.nn.functional as F
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.train_speech_embedder import test
from GUI.controller_screen_files.id_verifyController import curr_id
from GUI.controller_screen_files.enrollmentController import ids_and_rec_paths
from torch.utils.data import Dataset
from numpy.linalg import norm
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification

path_to_verify = None
ids_and_rec_paths_ver = {}
ids_and_npy_paths_ver = {}
path_of_dir = None



def install_verificationWin(installer, main_window):
    gf.set_background(main_window)
    #installer.open_window(Ui_MainWindow)
    #set background for the screen
    # main_window.setStyleSheet(
    #     "#" + str(
    #         main_window.objectName()) + " { border-image: url(../GUI/images/sound.jpeg) 0 0 0 0 stretch stretch; }")

    upload_button = gf.get_object_by_name(main_window, "uploadRec")
    upload_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    path_box = gf.get_object_by_name(main_window, "chosen_path")
    func = partial(install_upload_btn_func, path_box)
    upload_button.clicked.connect(func)

    submit_button = gf.get_object_by_name(main_window, 'submit')
    submit_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_submit_btn_func, installer)
    submit_button.clicked.connect(func)

def install_upload_btn_func(path):
    file_path_string = gf.get_path_by_dialog()
    # while not file_path_string.endswith('.wav'):
    #    gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    #    file_path_string = gf.get_path_by_dialog()
    if not file_path_string.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        global path_to_verify
        path.setText(file_path_string)
        path_to_verify = file_path_string
        ids_and_rec_paths_ver[curr_id[0]] = path_to_verify
        print(path_to_verify)

def install_submit_btn_func(installer):
    print(curr_id[0])
    #ids_and_rec_paths_ver.clear()
    if path_to_verify is None:
        return None
    if not path_to_verify.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        global path_of_dir
        path_of_dir = "C:\\finalProject\\datasets\\timit\\GUI\\Users\\{}".format(curr_id[0])
        os.makedirs(path_of_dir, exist_ok=True)
        verification_result = verify()
        if verification_result > 0.8:
            gf.info_message("Verification Completed", "Your verification is successful!")
        else:
            gf.error_message("Verification Failed", "The identity does not match the identity provided at the enrollment!")


def verify():

    #global path_of_dir
    #os.makedirs(os.path.join(path_of_dir, "enrolled"), exist_ok = True)
    print("curr id: ", curr_id[0])
    print("ids and rec paths: ", ids_and_rec_paths[curr_id[0]])


    verifed_utter_features = pre_process(ids_and_rec_paths_ver[curr_id[0]])
    feature_path = os.path.join(path_of_dir, "verified")
    os.makedirs(feature_path, exist_ok = True)
    feature_path = os.path.join(feature_path, "{}_verification.npy".format(curr_id[0]))
    np.save(feature_path, verifed_utter_features)
    print("The second pre process is over!")

    #
    test_dataset = SpeakerDatasetPreprocessed()

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=hp.test.num_workers,
                             drop_last=False)
    embedder_net = SpeechEmbedder()
    print("line 99")
    #
    #
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net.eval()
    print("line 111")
    for batch_id, mel_db_batch in enumerate(test_loader):
        print("line 120")
        #enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 1), dim=1)
        enrollment_batch, verification_batch = mel_db_batch[0,:,:,:], mel_db_batch[1, :, :, :]
        print("line 122")
        #enrollment_batch = torch.reshape(enrollment_batch, (1, enrollment_batch.size(2), enrollment_batch.size(3)))
        enrollment_batch = torch.reshape(enrollment_batch, (1, enrollment_batch.size(1), enrollment_batch.size(2)))
        print("line 124")
        #verification_batch = torch.reshape(verification_batch, (1, verification_batch.size(2), verification_batch.size(3)))
        verification_batch = torch.reshape(verification_batch,(1, verification_batch.size(1), verification_batch.size(2)))
        print("line 126")
        perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
        print("line 128")
        unperm = list(perm)
        print("line 130")
        for i, j in enumerate(perm):
            unperm[j] = i
        #verification_batch = verification_batch[perm]
        print("line 135")
        enrollment_embeddings = embedder_net(enrollment_batch)
        print("line 137")
        verification_embeddings = embedder_net(verification_batch)
        print("line 139")
        #verification_embeddings = verification_embeddings[unperm]
        print("line 141")
        enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                              (1, 1, enrollment_embeddings.size(1)))

        verification_embeddings = torch.reshape(verification_embeddings,
                                                (1, 1, verification_embeddings.size(1)))



    vec_enroll = enrollment_embeddings[0, 0, :]
    vec_verif = verification_embeddings[0, 0, :]

    vec_enroll1 = vec_enroll.cpu().detach().numpy()
    vec_verif1 = vec_verif.cpu().detach().numpy()

    cosine = np.dot(vec_enroll1, vec_verif1) / (norm(vec_enroll1) * norm(vec_verif1))
    print(cosine)
    return cosine


def pre_process(utter_path):

    transform_type = "db8"
    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr  # lower bound of utterance length

    utterances_spec = []
    utter, sr = librosa.core.load(utter_path, hp.data.sr)  # load utterance audio
    intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
                # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
                # for vctk dataset use top_db=100

    i = 0
    for interval in intervals:
        if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
            utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
            S = None
            if transform_type == "db8":
                # global S
                print("i is ", i)

                S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                        win_length=int(hp.data.window * sr),
                                        hop_length=int(hp.data.hop * sr))  # OUR TRANSFORM

                S = np.abs(S) ** 2  # OUR TRANSFORM

                basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft,
                                            n_mels=hp.data.nmels)  # OUR TRANSFORM

                S = np.log10(np.dot(basis, S) + 1e-6)  # log mel spectrogram of utterances

                #print(len(S))
                #print(type(S))
            #utterances_spec.append(S[:, :hp.data.tisv_frame])  # first 180 frames of partial utterance
            #utterances_spec.append(S[:, -hp.data.tisv_frame:])  # last 180 frames of partial utterance
            utterances_spec.append(S[:, :180])

            utterances_spec.append(S[:, -180:])
            print(utterances_spec)
            utterances_spec = np.array(utterances_spec)
            print(utterances_spec.shape)
            i += 1

            #np.save(os.path.join(hp.data.train_path, "speaker%d.npy" % curr_id[0]), utterances_spec)

    return utterances_spec



class SpeakerDatasetPreprocessed(Dataset):

    def __init__(self, shuffle=False, utter_start=0):


        self.path_enrolled = os.path.join(path_of_dir, "enrolled")
        self.path_verified = os.path.join(path_of_dir, "verified")
        #self.file_list = os.listdir(self.path)
        self.file_list = [self.path_enrolled, self.path_verified]
        self.shuffle = shuffle
        #self.shuffle = False
        self.utter_start = 0

    def __len__(self):

        return 2

    def __getitem__(self, idx):
        np_file_list = []
        for file in self.file_list:
            np_file_list.append(os.path.join(file, os.listdir(file)[0]))
        #np_file_list = os.listdir(self.file_list)

        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]

        #utters = np.load(os.path.join(self.path, selected_file))  # load utterance spectrogram of selected speaker
        print("idx is ", idx)
        print(np_file_list[idx])
        utters = np.load(np_file_list[idx])

        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)  # select M utterances per speaker
            utterance = utters[utter_index]
        else:
            utterance = utters[self.utter_start: self.utter_start + 1]  # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]

        return utterance
