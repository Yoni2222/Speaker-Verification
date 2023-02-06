import os
import random
import shutil
#from DB_objects.details import Details
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.details import Details

class DataBaseVendor:
    def __init__(self, main_directory_path):
        self.main_directory_path = os.path.realpath(main_directory_path)
        if not os.path.exists(self.main_directory_path):
            raise Exception("there is no such a path : '" + self.main_directory_path + "'")
        self.id_train_dict = self.__get_ids_address("TRAIN")
        self.id_test_dict = self.__get_ids_address("TEST")

    def __get_dict(self, main_dataset_name):
        if main_dataset_name == "TRAIN":
            return self.id_train_dict
        if main_dataset_name == "TEST":
            return self.id_test_dict
        raise Exception("there is no match dict by the name: " + main_dataset_name)

    def get_ids(self, main_dataset_name):
        return list(self.__get_dict(main_dataset_name).keys())

    def convert_path(self, path):
        path = path[len(self.main_directory_path):]
        ls = path.split("\\")
        while ls[0] == "":
            ls = ls[1:]
        return ls

    def get_details(self, id, main_dataset_name):
        details_path = os.path.join(self.__get_dict(main_dataset_name)[id], "details.txt")
        return Details.from_file(details_path)

    def get_wav_paths(self, id, main_dataset_name):
        id_path = self.__get_dict(main_dataset_name)[id]
        ls = [f.name for f in os.scandir(id_path)]
        ls = [item for item in ls if item[-1:-4:-1][::-1].upper() == "WAV"]
        #ls = [item for item in ls if item[-1:-8:-1][::-1] == "WAV.wav"]
        return [os.path.join(id_path, item) for item in ls]

    def __get_ids_address(self, main_dataset_name):
        id_dict = dict([])
        if main_dataset_name is None:
            raise Exception("you have to choose main folder - should be 'TRAIN' or 'TEST'")
        _, dr_list = self.get_structure_names(main_dataset_name)
        for dr in dr_list:
            dr_path, id_list= self.get_structure_names(main_dataset_name, dr)
            for id in id_list:
                id_dict[id] = os.path.join(dr_path,id)
        return id_dict

    def get_structure_names(self, main_dataset_name=None, dr_name=None, ID=None, file_format=None):
        if main_dataset_name is None:
            return self.__get_main_dataset_list()
        elif dr_name is None:
            return self.__get_dr_list(main_dataset_name)
        elif ID is None:
            return self.__get_id_list(main_dataset_name, dr_name)
        elif file_format is None:
            return self.__get_files_list(main_dataset_name,dr_name,ID)
        else:
            return self.__get_files_list_by_format(main_dataset_name,dr_name,ID,file_format)

    def __get_main_dataset_list(self):
        path = self.main_directory_path
        return path, [f.name for f in os.scandir(path) if f.is_dir()]

    def __get_dr_list(self, main_dataset_name):
        if main_dataset_name not in self.__get_main_dataset_list()[1]:
            raise Exception("not database named: " + main_dataset_name)
        path = os.path.join(self.main_directory_path, main_dataset_name)
        return path, [f.name for f in os.scandir(path) if f.is_dir()]

    def __get_id_list(self, main_dataset_name, dr_name):
        if dr_name not in self.__get_dr_list(main_dataset_name)[1]:
            raise Exception("not DR named: " + dr_name)
        path = os.path.join(self.main_directory_path, main_dataset_name, dr_name)
        return path, [f.name for f in os.scandir(path) if f.is_dir()]

    def __get_files_list(self,main_dataset_name,dr_name, id):
        if id not in self.__get_id_list(main_dataset_name,dr_name)[1]:
            raise Exception("not ID named: " + id)
        path = os.path.join(self.main_directory_path, main_dataset_name, dr_name, id)
        return path, [f.name for f in os.scandir(path)]# if f.is_file()

    def __get_files_list_by_format(self,main_dataset_name,dr_name, id, file_format):
        path, ls = self.__get_files_list(main_dataset_name,dr_name,id)
        # get_structure_names by 3 last charecters
        return path, [item for item in ls if item[-1:-4:-1][::-1] == file_format]

    def get_path(self, *args):
        return os.path.join(self.main_directory_path, *args)

    def is_id_exist(self, id, main_dataset_name):
        return id in self.__get_dict(main_dataset_name).keys()

    def add_wav_to_existing_id(self, id, main_dataset_name, wav_file_path):
        id_path = self.__get_dict(main_dataset_name)[id]
        shutil.copy(wav_file_path, id_path)

    def add_id(self, id, main_dataset_name, details, wav_file):
        """
        :type details:Details
        :param id:
        :param main_dataset_name:
        :param details:
        :param wav_file:
        :return:
        """
        if self.is_id_exist(id, main_dataset_name):
            raise "the id is already existing"
        city = details["city"]
        _, dr_list = self.get_structure_names(main_dataset_name)
        if city not in dr_list:
            raise Exception("the city - " + str(city) + " is not one of the options : " + str(dr_list))
        new_dir_path = self.get_path(main_dataset_name, city, id)
        os.mkdir(new_dir_path)
        self.__get_dict(main_dataset_name)[id] = new_dir_path
        details.to_file(new_dir_path)
        self.add_wav_to_existing_id(id,main_dataset_name, r"C:\Users\Tomer\Desktop\Function_Num_Of_Peaks.xlsm")

    def insert_random_details(self):
        first_name_ls = DataBaseVendor.__get_list_from_file("first_names.txt", ",")
        last_name_ls = DataBaseVendor.__get_list_from_file("last_names.txt",",")
        address_ls = DataBaseVendor.__get_list_from_file("addresses.txt", ",")
        _, data_type_ls = self.get_structure_names()
        for data_type in data_type_ls:
            _, dr_name_ls = self.get_structure_names(data_type)
            for dr_name in dr_name_ls:
                dr_path, id_ls = self.get_structure_names(data_type, dr_name)
                for id in id_ls:
                    path = os.path.join(self.get_path(data_type, dr_name, id), "details.txt")
                    first_name = first_name_ls[random.randint(0, len(first_name_ls) - 1)]
                    last_name = last_name_ls[random.randint(0, len(last_name_ls) - 1)]
                    phone_number = "0"
                    for i in range(8):
                        phone_number += str(random.randint(0, 9))
                    address = address_ls[random.randint(0, len(address_ls) - 1)] + "-" + str(random.randint(1000, 9999))
                    details = first_name + "," + last_name + "," + id + "," + phone_number + ","\
                              + address + "," + dr_name
                    file = open(path, "w")
                    file.write(details)
                    file.close()

    @staticmethod
    def __get_list_from_file(file_path, sep):
        my_file = open(file_path, "r")
        text = my_file.read()
        my_file.close()
        return text.split(sep)


if __name__ == '__main__':

    db = DataBaseVendor(r"../timit")
    #x = db.convert_path(r"C:\Users\Tomer\PycharmProjects\finalProject1.1\Timit_Database\TRAIN\DR1\FCJF0\details.txt")
    #print(x)


