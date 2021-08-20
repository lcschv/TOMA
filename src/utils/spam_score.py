import gzip
from os import listdir
from os.path import isfile, join
import pprint

class SpamScore(object):

    def __init__(self, qrels_path, spam_folder="data/waterloo-spam-cw12-decoded", collection=None):
        self.spam_folder = spam_folder
        self.qrels_path = qrels_path
        self.normalized_score_rank = {}
        if collection is None:
            self.files_to_unzip = self.get_list_files(spam_folder)
        self.dict_docs_spamscore_qrels = self.get_docs_qrels()
        if collection is not None:
            self.read_spam_score_09()
        else:
            self.read_spam_score()

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def add_spam_score_to_qrels(self):
        dict_to_use = self.dict_docs_spamscore_qrels
        new_qrels = open("data/processed_data/qrels_webtrack2009/qrels_final.txt", "w")
        with open(self.qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            docid = line.split()[2]
            if docid in dict_to_use:
                print(line + " " +str(dict_to_use[docid]), file=new_qrels)
            else:
                print("Something went wrong, docid not found in normalized_score_rank.")
                exit()
        new_qrels.close()


    def read_spam_score_09(self):
        file_spamrank_qrels_docs = open("data/processed_data/file_spamrank_qrels_webtrack2009_docs.txt", "w")
        file_pr = open(self.spam_folder, "r")
        while True:
            line = file_pr.readline()
            if not line:
                break
            parts = line.split()
            docid = parts[1]
            spam_score = parts[0]
            if docid in self.dict_docs_spamscore_qrels:
                # print(docid, spam_score)
                self.dict_docs_spamscore_qrels[docid] = spam_score

        # pprint.pprint(self.dict_docs_spamscore_qrels, file_spamrank_qrels_docs)
        file_spamrank_qrels_docs.close()



    def read_spam_score(self):
        # file_spamrank_qrels_docs = open("data/processed_data/file_spamrank_qrels_webtrack2014_docs.txt", "w")
        for file in self.files_to_unzip:
            with gzip.open(file, 'r') as fin:
                content = fin.readlines()
            content = [x.rstrip().decode('utf-8') for x in content]
            for line in content:
                parts = line.split()
                docid = parts[1]
                spam_score = parts[0]
                if docid in self.dict_docs_spamscore_qrels:
                    # print(docid, spam_score)
                    self.dict_docs_spamscore_qrels[docid] = spam_score

        # pprint.pprint(self.dict_docs_spamscore_qrels, file_spamrank_qrels_docs)
        # file_spamrank_qrels_docs.close()

    def get_docs_qrels(self):
        dict_docs_spamscore_qrels = {}
        with open(self.qrels_path) as f:
            content = f.readlines()

        content = [x.rstrip() for x in content]
        for line in content:
            docid = line.split()[2]
            if docid not in dict_docs_spamscore_qrels:
                dict_docs_spamscore_qrels[docid] = -2

        return dict_docs_spamscore_qrels
