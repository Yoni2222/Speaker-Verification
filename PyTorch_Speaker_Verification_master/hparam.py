"""
Copyright (c) 2019, HarryVolek
All rights reserved.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import pathlib

import yaml


def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='./config/config.yaml'):
        super(Dotdict, self).__init__()
        CONFIG_PATH = "config\\config.yaml"
        CONFIG_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), CONFIG_PATH)
        print(CONFIG_PATH)

        hp_dict = load_hparam(CONFIG_PATH)

        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__
"""
    def __init__(self, file='/content/gdrive/My Drive/project/PyTorch_Speaker_Verification_master/config/config.yaml'):
        super(Dotdict, self).__init__()
        #CONFIG_PATH = "config/config.yaml"
        #CONFIG_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), CONFIG_PATH)
        #print(CONFIG_PATH)
        CONFIG_PATH = "/content/gdrive/My Drive/project/PyTorch_Speaker_Verification_master/config/config.yaml"
        hp_dict = load_hparam(CONFIG_PATH)
"""
        
hparam = Hparam()
