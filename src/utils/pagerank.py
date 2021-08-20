import pandas as pd
from sklearn import preprocessing
import numpy as np
import math
import pprint
from pathlib import Path

class PageRank(object):

    def __init__(self, pagerank_path, qrels_path):
        self.pagerank_path = pagerank_path
        # self.pagerank_path = r"C:\Lucas\PhD\LLMS_multieval\data\pagerank.scoreOrder"
        self.qrels_path = qrels_path
        # self.qrels_path = r"C:\Lucas\PhD\LLMS_multieval\data\qrels_only_query.txt"
        self.normalized_pagerank = {}
        self.dict_docs_pagerank_qrels = self.get_docs_qrels()
        self.read_pagerank()
        self.normalize_pagerank()

    def add_pagerank_qrels(self, normalized=True):
        if normalized:
            dict_to_use = self.normalized_pagerank
            new_qrels = open("data/processed_data/qrels_webtrack2009/qrels_with_pr_normalized.txt", "w")
        else:
            new_qrels = open("data/processed_data/qrels_webtrack2009/qrels_with_pr_not_normalized.txt", "w")
            dict_to_use = self.dict_docs_pagerank_qrels

        with open (self.qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            docid = line.split()[2]
            if docid in dict_to_use:
                print(line + " " +str(dict_to_use[docid]), file=new_qrels)
            else:
                print("Something went wrong, docid not found in normalized_pagerank.")
                exit()

        new_qrels.close()


    def get_docs_qrels(self):
        dict_docs_pagerank_qrels = {}
        with open(self.qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            docid = line.split()[2]
            if docid not in dict_docs_pagerank_qrels:
                dict_docs_pagerank_qrels[docid] = -2

        return dict_docs_pagerank_qrels

    def read_pagerank(self):
        file_pr = open(self.pagerank_path, "r")
        file_pagerank_qrels_docs = open("data/processed_data/file_pagerank_qrels_docs_webtrack2009.txt", "w")
        self.max_pagerank = -1
        self.min_pagerank = 99999999
        while True:
            line = file_pr.readline()
            if not line:
                break
            docid = line.rstrip().split()[0]
            pagerank = float(line.rstrip().split()[1])
            # print(line)
            """Assing pagerank values to documents in qrels"""
            if docid in self.dict_docs_pagerank_qrels:
                self.dict_docs_pagerank_qrels[docid] = pagerank
                # print(docid)
            if pagerank > self.max_pagerank:
                self.max_pagerank = pagerank
                max_docid = docid
            if pagerank < self.min_pagerank:
                self.min_pagerank = pagerank
                min_docid = docid
        self.dict_docs_pagerank_qrels['max_pagerank'] = self.max_pagerank
        self.dict_docs_pagerank_qrels['min_pagerank'] = self.min_pagerank
        pprint.pprint(self.dict_docs_pagerank_qrels,file_pagerank_qrels_docs)

        file_pagerank_qrels_docs.close()
        file_pr.close()

    def normalize_pagerank(self):
        log_min_pagerank = math.log2(self.min_pagerank)
        log_max_pagerank = math.log2(self.max_pagerank)

        for docid, pr in self.dict_docs_pagerank_qrels.items():
            if docid not in self.normalized_pagerank:
                self.normalized_pagerank[docid] = 0
            #Enter in this if in case that the pagerank was not found.
            if pr < 0.0:
                self.normalized_pagerank[docid] = -2
            else:
                self.normalized_pagerank[docid] = (math.log2(pr) - log_min_pagerank) / (log_max_pagerank - log_min_pagerank)

# if __name__ == '__main__':
#     pagerank = PageRank("pagerank.scoreOrder", "qrels_only_query.txt")
#     pagerank.add_pagerank_qrels(normalized=False)
