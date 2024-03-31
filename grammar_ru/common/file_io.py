import yaml
import pickle
import json
import os
import jsonpickle


class FileIO:

    @staticmethod
    def read_pickle(filename):
        with open(filename,'rb') as file:
            return pickle.load(file)

    @staticmethod
    def read_json(filename, as_obj = False):
        with open(filename,'r') as file:
            result = json.load(file)
            return result


    @staticmethod
    def read_text(filename):
        with open(filename,'r',encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def write_pickle(data, filename):
        with open(filename,'wb') as file:
            pickle.dump(data,file)

    @staticmethod
    def write_json(data, filename):
        with open(filename,'w') as file:
            json.dump(data,file,indent=1)

    @staticmethod
    def write_text(data, filename):
        with open(filename,'w',encoding='utf-8') as file:
            file.write(data)
