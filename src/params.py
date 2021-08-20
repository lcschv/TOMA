import json
from os import listdir
from os.path import isfile, join

class Params(object):
    def __init__(self):
        pass

    def get_indri_parameters(self, indri_config = "config/indri.json"):
        self.indri_config = indri_config
        self.indri_parameters = {}
        try:
            with open(self.indri_config) as f:
                self.indri_parameters = json.load(f)
        except FileNotFoundError:
            print("Indri Parameters file not found.")
            exit(1)
        return self.indri_parameters

    def get_order_parameters(self, qrels_config = "config/distance_order.json"):
        self.distance_order_config = qrels_config
        self.distance_order_parameters = {}
        try:
            with open(self.distance_order_config) as f:
                self.distance_order_parameters = json.load(f)
        except FileNotFoundError:
            print("Distance order parameters file not found.")
            exit(1)
        return self.distance_order_parameters

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def get_file_content(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        return content