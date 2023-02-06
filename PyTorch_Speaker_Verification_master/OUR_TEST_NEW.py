import math
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import norm
from torch.utils.data import DataLoader

from typing import Any, Callable, Iterable, TypeVar
import torch
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master import db_manager
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.db_manager import DataBaseVendor
from torch.utils.data import IterDataPipe, IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler, Dataset
import torch.utils.data.graph_settings
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]

class SpeakerDatasetPreprocessed(Dataset):
    def __init__(self, original_after_process, test_after_process,  shuffle=False, utter_start=0):

        self.shuffle = shuffle
        self.utter_start = 0
        self.np_file_list = [original_after_process, test_after_process]

    def __len__(self):
        return 2

    def __getitem__(self, idx):

        if self.shuffle:
            selected_file = random.sample(self.np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = self.np_file_list[idx]

        #utters = np.load(os.path.join(self.path, selected_file))  # load utterance spectrogram of selected speaker

        utters = self.np_file_list[idx]
        utterance = utters[self.utter_start: self.utter_start + 1]  # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size
        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]

        return utterance



class TimeDetail:
    def __init__(self, union_ls, start_time, finish_time):
        self.union_ls = union_ls # type:#list[TimeDetail.Union]
        self.start_time = start_time
        self.finish_time = finish_time

    def get_accuracy(self, id):
        for union in self.union_ls:
            if union.id == id:
                return union.accuracy
        raise Exception("the id : " + str(id) + " , is not in the DB")

    def __str__(self):
        string = "TimeDetail:\n"
        for union in self.union_ls:
            string += str(union)
        string += "start_time : " + str(self.start_time) + "\n"
        string += "finish_time : " + str(self.finish_time)
        return string

    def is_none(self):
        try:
            if self.union_ls[0].accuracy == -2:
                return True
        except:
            print("line 74")
            return True
        print("line 76")
        return False

#returns a list
    def get_unions(self, amount=0):
        if amount < 0:
            raise Exception("can't give minus amount")
        if amount == 0 or amount >= len(self.union_ls):
            ls = self.union_ls[:]
            return ls
        ls =  self.union_ls[0:amount]
        return ls

    class Union:
        def __init__(self, id, accuracy):
            self.accuracy = accuracy
            self.id = id

        def __str__(self):
            string = "name : " + str(self.id) + ", similarity : " + str(self.accuracy) + "\n"
            return string

        @staticmethod
        def get_accuracy(item):
            return item.accuracy


class TimeLineDetails:
    def __init__(self, split_time):
        self.time_list = []  # type: list[TimeDetail]
        self.split_time = split_time
        self.start_time = 0
        self.end_time = 0

    def get_appreciation_by_seconds(self, seconds):
        for td in self.time_list:
            if seconds <= td.finish_time and seconds >= td.start_time:
                return td
        raise Exception("time is not valid !")

    def add_verification(self, union_ls, start_time=0, finish_time=0):
        count = len(self.time_list)
        if finish_time == 0:
            finish_time = (count+1)*self.split_time
        if start_time == 0:
            start_time = count * self.split_time
        td = TimeDetail(union_ls, start_time, finish_time)
        self.end_time = finish_time
        self.time_list.append(td)

    def get_time_accuracy(self, id):
        ls = []
        fs = []
        for td in self.time_list:
            ls.append([td.start_time, td.finish_time, round(td.get_accuracy(id[0]), 2)])
        return ls

    def get_none_index(self):
        invalid_indexes = []
        print(len(self.time_list))
        for i in range(len(self.time_list)):
            if self.time_list[i].is_none():
                invalid_indexes.append(i)
        invalid_indexes.reverse()
        return invalid_indexes

    def fix(self):
        invalid_indexes = self.get_none_index()

        print("len of invalid indexes is ", len(invalid_indexes), ", len of time list is ", len(self.time_list))
        if len(invalid_indexes) == len(self.time_list):
            raise Exception("it can't be fixed - all the items are None")

        while len(invalid_indexes) != 0:

            for i in invalid_indexes:
                if i == 0:

                    if not self.time_list[i+1].is_none():
                        self.time_list[i].union_ls = self.time_list[i+1].get_unions()
                    continue

                if i == len(self.time_list)-1:

                    if not self.time_list[i-1].is_none():
                        self.time_list[i].union_ls = self.time_list[i-1].get_unions()
                    continue

                first_union_ls = None
                second_union_ls = None
                if not self.time_list[i-1].is_none():

                    first_union_ls = self.time_list[i-1].get_unions()
                if not self.time_list[i+1].is_none():

                    second_union_ls = self.time_list[i+1].get_unions()
                if first_union_ls is None and second_union_ls is None:

                    continue
                if (first_union_ls is None and not second_union_ls is None) or (not first_union_ls is None and second_union_ls is None):

                    if not first_union_ls is None:

                        self.time_list[i].union_ls = first_union_ls
                    else:

                        self.time_list[i].union_ls = second_union_ls
                    continue

                dt = self.time_list.pop(i)

                diff = dt.finish_time - dt.start_time

                second_to_insert = TimeDetail(second_union_ls, dt.start_time, dt.start_time+diff/2)

                first_to_insert = TimeDetail(second_union_ls, dt.start_time + diff / 2, dt.finish_time)

                self.time_list.insert(i,first_to_insert)

                self.time_list.insert(i, second_to_insert)

            invalid_indexes = self.get_none_index()

    def __len__(self):
        return len(self.time_list)

    def __getitem__(self, item):
        return self.time_list[item]

    def __str__(self):
        string = "TimeLineDetails {"
        for item in self.time_list:
            string += "\n" + str(item) + "\n"
        string += "}"
        return string


class Verification:
    def __init__(self, model_path, DATA_VENDOR):
        print("line 191")
        self.dv = DATA_VENDOR
        self.model_path = model_path
        self.names_dict = dict()
        self.TLD = None
        self.__processed_picture = None
        self.__unprocessed_array = None
        print("line 197")
        i = 0
        for id in self.dv.id_test_dict:
            print("i is ", i)
            self.names_dict[id] = self.dv.get_wav_paths(id, "TEST")
            i += 1
        # for id in dv.id_train_dict:
        #     self.names_dict[id] = self.dv.get_wav_paths(id, "TRAIN")
        print("line 202")
        limit = len(self.names_dict)
        count = 1
        print("line 205")
        for key in self.names_dict.keys():
            print("process :", count, "/", limit)
            person_record_list = self.names_dict[key]
            for i in range(len(person_record_list)):
                person_record_list[i] = self.process_it(person_record_list[i])
                if person_record_list[i] is None:
                    print("person_record_list[i] is None !!")
            count += 1

    def fit(self, original_path, split_time):
        self.__processed_picture = self.process_it(original_path, 0)
        self.__unprocessed_array, _ = librosa.core.load(original_path, 16000)
        self.TLD = TimeLineDetails(split_time)
        self.path = original_path
        if split_time == 0:
            part, min_time, max_time = self.process_it(original_path, split_time, get_details=True)

            ls = self.get_max_cosin(part)
            #print("ls is ", ls)
            self.TLD.add_verification(ls, round(min_time,2), round(max_time,2))
        else:
            precess_parts = self.process_it(original_path, split_time)
            for i, (part, min_time, max_time) in enumerate(precess_parts):
                print("processing round ", i, " - (out of ", len(precess_parts), " rounds)")
                #print("part = ", part, " min_time = ", min_time, " max_time = ", max_time)
                ls = self.get_max_cosin(part)
                print("line 232")
                self.TLD.add_verification(ls, round(min_time, 2), round(max_time, 2))
        print("line 234")
        self.TLD.fix()

    def get_timeline(self, jump_x_time, trendlines=1):
        spot_times = [index_time*jump_x_time for index_time in range(int((self.TLD.end_time-self.TLD.start_time)/jump_x_time))]
        id_array_sound = [self.TLD.get_appreciation_by_seconds(time) for time in spot_times]
        names = [union.id for union in self.TLD.time_list[0].union_ls]
        name_to_index = dict(zip(names,[i+1 for i in range(len(names))]))

        for priority in range(trendlines):
            values = [name_to_index[td.union_ls[priority].id] for td in id_array_sound]
            plt.plot(spot_times, values)
        plt.show()
        return plt

    def get_plot_bars(self, accuracies):
        time_intervals = []
        values = []
        val = None
        for tup in accuracies:
            time_intervals.append(str(tup[0]) + "-" + str(tup[1]))
            val = tup[2]
            if tup[2] < 0 and tup[2] > -0.5:
                val = tup[2] + 0.5
            elif tup[2] < 0 and tup[2] <= -0.5:
                val = 1 + tup[2]
            values.append(val)

        """if self.path.endswith("benet2.wav"):
            time_intervals = ["0 - 3.0", "3.0 - 6.0", "6.0 - 9.0", "9.0 - 12.0", "12.0 - 15.0", "15.0 - 16.86"]
            values = [0.9, 0.88, 0.69, 0.62, 0.07, 0.04]
        elif self.path.endswith("michaeli.wav"):
            time_intervals = ["0 - 3.0", "3.0 - 6.0", "6.0 - 9.0", "9.0 - 12.0", "12.0 - 15.0", "15.0 - 18.0", "18.0 - 20.91"]
            values = [0.06, 0.15, 0.74, 0.92, 0.94, 0.07, 0.03]
        """

        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(time_intervals, values, color='maroon',
                width=0.4)

        plt.xlabel("Time Intervals")
        plt.ylabel("Accuracy Score")
        plt.title("The probability of participating in each segment")
        plt.show()

    def process_it(self, path, split_time=0,get_details=False):
        utter, sr = librosa.core.load(path, 16000)  # load utterance audio
        if split_time == 0:
            proccesed = self._pre_process(utter, sr)
            if proccesed is None:
                print("THE PROCCESSED IS NONE !!!!")
            if get_details:
                return proccesed, 0 , round(len(utter) / 16000, 2)
            return proccesed
        minimum_split_time = 0.5
        if split_time < minimum_split_time:
            raise Exception("this split time is too short..")
        minimum_frames = math.ceil(split_time*16000)
        utter_parts = []
        for i in range(0, len(utter), minimum_frames):
            minimum = i + minimum_frames if i + minimum_frames < len(utter) else len(utter)
            utter_parts.append((utter[i:minimum], round(i/16000, 2), round(minimum/16000, 2)))
        pre_processed_parts = []
        for part in utter_parts:
            processed_part = self._pre_process(part[0], sr)
            pre_processed_parts.append((processed_part, part[1], part[2]))
        return pre_processed_parts

    def load_processed_image(self, path):
        return np.load(path)

    def get_max_cosin(self, processed_original):
        minimum_cosin = -2
        index = 0
        id_accuracy_arr = []
        for id in self.names_dict.keys():
            print(index, end=", ")
            index += 1
            person_record_list = self.names_dict[id]
            sum_cosin = 0
            cnt_cosin = 0
            for i in range(0, 4):
                try:
                    a = self.get_cosin(processed_original, person_record_list[i])
                    sum_cosin += self.get_cosin(processed_original, person_record_list[i])
                    cnt_cosin += 1
                except:
                    pass

                # cosin_value = self.get_cosin(processed_original, person_record_list[i])
            if cnt_cosin == 0:
                avg_cosin = minimum_cosin
            else:
                avg_cosin = sum_cosin / cnt_cosin
            id_accuracy_arr.append(TimeDetail.Union(id, avg_cosin))
        id_accuracy_arr.sort(key=TimeDetail.Union.get_accuracy, reverse=True)
        return id_accuracy_arr
    def get_processed_picture(self):
        if self.__processed_picture is None:
            raise Exception("First Fit The Object")
        return self.__processed_picture

    def get_unprocessed_array(self):
        if self.__unprocessed_array is None:
            raise Exception("First Fit The Object")
        return self.__unprocessed_array

    def get_cosin(self, processed_original, processed_test):
        if processed_original is None or processed_test is None:
            print("NONE PROBLEM !!")
        sp = SpeakerDatasetPreprocessed(processed_original, processed_test)
        sp = DataLoader(sp, batch_size=2, shuffle=False, num_workers=0,
                   drop_last=False)
        embedder_net = SpeechEmbedder()
        embedder_net.load_state_dict(torch.load(self.model_path))
        embedder_net.eval()
        for batch_id, mel_db_batch in enumerate(sp):

            # enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1) / 1), dim=1)
            enrollment_batch, verification_batch = mel_db_batch[0, :, :, :], mel_db_batch[1, :, :, :]

            # enrollment_batch = torch.reshape(enrollment_batch, (1, enrollment_batch.size(2), enrollment_batch.size(3)))
            enrollment_batch = torch.reshape(enrollment_batch, (1, enrollment_batch.size(1), enrollment_batch.size(2)))
            verification_batch = torch.reshape(verification_batch,
                                               (1, verification_batch.size(1), verification_batch.size(2)))
            perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            enrollment_embeddings = embedder_net(enrollment_batch)
            verification_embeddings = embedder_net(verification_batch)
            enrollment_embeddings = torch.reshape(enrollment_embeddings,
                                                  (1, 1, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings,
                                                    (1, 1, verification_embeddings.size(1)))
        vec_enroll = enrollment_embeddings[0, 0, :]
        vec_verif = verification_embeddings[0, 0, :]
        vec_enroll1 = vec_enroll.cpu().detach().numpy()
        vec_verif1 = vec_verif.cpu().detach().numpy()
        cosine = np.dot(vec_enroll1, vec_verif1) / (norm(vec_enroll1) * norm(vec_verif1))
        return cosine

    def _pre_process(self, utter, sr):
        utter_min_len = (180 * 0.005 + 0.025) * 16000  # lower bound of utterance length
        intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection
        utterances_spec_arr = []
        utterances_spec = []
        for interval in intervals:
            if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficient long,
                utter_part = utter[interval[0]:interval[1]]  # save first and last 180 frames of spectrogram.
                S = librosa.core.stft(y=utter_part, n_fft=400,
                                      win_length=int(0.025 * sr),
                                      hop_length=int(0.005 * sr))  # OUR TRANSFORM
                S = np.abs(S) ** 2  # OUR TRANSFORM
                basis = librosa.filters.mel(sr=16000, n_fft=400,
                                            n_mels=40)  # OUR TRANSFORM
                S = np.log10(np.dot(basis, S) + 1e-6)  # log mel spectrogram of utterances
                utterances_spec.append(S[:, :180])
                utterances_spec.append(S[:, -180:])
                utterances_spec_arr = np.array(utterances_spec)

                # np.save(os.path.join(hp.data.train_path, "speaker%d.npy" % curr_id[0]), utterances_spec)

        return utterances_spec_arr


if __name__ == '__main__':
    model_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\PyTorch_Speaker_Verification_master\models\stft\final_epoch_800_batch_id_141.model"
    dv = DataBaseVendor(r"C:\finalProject\datasets\timit\data")
    enroll_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\PyTorch_Speaker_Verification_master\bibi_benet.wav"

    ver = Verification(model_path, dv)
    ver.fit(enroll_path, 10)
    print(ver.TLD.get_time_accuracy("bibi"))
    #ver.get_timeline(1,2)

    """
    
import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'C':20, 'C++':15, 'Java':30,
        'Python':35}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()
    """
