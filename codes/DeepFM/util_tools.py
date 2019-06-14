#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : util_tools.py
# PythonVersion: python3.5
# Date    : 2019/4/21 下午6:53
# Software: PyCharm

"""To define parameter dictionary and tf.log setting.
    Ref from: https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp/model
"""
import json
import logging


class Params():
    """
    Class that loads hyper-parameters from a json file.
    Example:
        ```
        params = Parmas(json_file)
        print(params.learning_rate)
        params.learning_rate = 0.5 # change the value of parameters in params dict
        ```
    """
    def __init__(self, json_file):
        self.update(json_file)

    def save(self, json_file):
        """Save parameters to a json file"""
        with open(json_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_file):
        """Load data from json file. and update properties of object"""
        with open(json_file) as f:
            params = json.load(f)
            self.__dict__.update(params)    # add and update values in object

    def print(self):
        """Print parameters and it's value"""
        logging.info("----------Parameters---------")
        for k, v in self.__dict__.items():
            logging.info('{} : {}'.format(k, v))
        logging.info("-----------------------------")

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_file):
    """
    Set the logger to log info in terminal and file 'log_file'

    Here the log file will be saved in `model_dir/train.log`
    Notes, you also can use tf.logging instead of python logging.
    :param log_file: log file name
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_file)
        # set logger format
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # logging to console
        stream_handle = logging.StreamHandler()
        stream_handle.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handle)


def save_dict_json(d, json_file):
    """
    Save a dict to json file
    :param d: dict of float-castable values(np.float, int, float,etc.)
    :param json_file: json file
    :return:
    """
    with open(json_file, 'w') as f:
        # convert dict values to float, not support np.array,np.float
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)