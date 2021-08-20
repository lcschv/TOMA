import os
import re
from src.params import Params
import pprint
from pathlib import Path
import platform
import os


class Search(Params):

    def __init__(self):
        super(Search, self).__init__()
        if platform.system() == "Windows":
            print("Exiting because you are running on Windows. This is a reminder to not overwrite runs.")
            exit()
    def prepare_indri_queries(self, raw_queries = None):

        """
            This function generates the IndriRunQuery template. If you want to change the parameters
            of Indri (e.g. index path, memory, etc.) please change on config/indri.json
        """

        if raw_queries is None:
            raw_queries = ["data/processed_data/queries/qid_query.txt",
                         "data/processed_data/queries/qid_with_subtasks.txt"]

        self.indri_parameters = Params.get_indri_parameters(self)

        # Check if folder exists, otherwise create it.
        Path('data/processed_data/queries_indri/').mkdir(parents=True, exist_ok=True)

        # Generate the IndriRunQuery input file template.
        for query_file in raw_queries:
            arr_queries = self.get_queries(query_file)
            indri_out_file_name = query_file.split('/')[-1]
            self.write_to_file_query_indriformat(indri_out_file_name, arr_queries)

    def write_to_file_query_indriformat(self, indri_out_file_name, arr_queries):
        file_out = open("data/processed_data/queries_indri/"+indri_out_file_name, "w")
        print("<parameters>", file=file_out)
        for key, val in self.indri_parameters.items():
            print("<"+str(key)+">"+str(val)+"</"+str(key)+">", file=file_out)
        print(file=file_out)
        for query in arr_queries:
            print("<query>", file=file_out)
            print("<number>", query[0], "</number>", file=file_out)
            print("<text>", query[1], "</text>", file=file_out)
            print("</query>", file=file_out)
        print("</parameters>", file=file_out)

    def get_queries(self, query_file):
        content = self.get_file_content(query_file)
        arr_queries = []
        for line in content[1:]:
            qid, query = line.split('\t')
            query = self.clean_query(query)
            arr_queries += [(qid, query)]
        return arr_queries

    def clean_query(self, query):
        return re.sub('[^a-zA-Z0-9 \n\.]', '', query).replace('.','')

    def get_file_content(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        return content

    def search_griparameters(self):
        # Path('data/runs/lm_jel').mkdir(parents=True, exist_ok=True)
        # Path('data/runs/lm_dir').mkdir(parents=True, exist_ok=True)
        # Path('data/runs/bm25').mkdir(parents=True, exist_ok=True)
        for i in [0.1, 0.2, 0.04, 0.06, 0.08, 0.12, 0.14, 0.16, 0.18, 0.22, 0.24, 0.26]:
        # for i in range(1, 100, 2):
            lambda_val = i
            # lambda_val = (i + 1) / 100
            print("LM-Jel: ", lambda_val)
            cmd = "IndriRunQuery data/processed_data/queries_indri/qid_query.txt -rule=method:linear,collectionLambda:" + str(
                lambda_val) + " >data/runs/lm_jel/" + str(lambda_val) + ".txt"
            output = str(os.system(cmd))

        # for i in range(200, 10201, 400):
        #     print("LM-dir: ", i)
        #     cmd = "IndriRunQuery data/processed_data/queries_indri/qid_query.txt -rule=method:dir,mu:" + str(i) + " >data/runs/lm_dir/" + str(
        #         i) + ".txt"
        #     output = str(os.system(cmd))
        # for j in range(1, 302, 20):
        #     if j % 2 != 0 and j != 1:
        #         k1 = (j - 1) / 100
        #     else:
        #         k1 = j / 100
        #     for i in range(30, 91, 15):
        #         b = i / 100
        #         print("BM25 --", "k1:", k1, " b", b)
        #         cmd = "IndriRunQuery data/processed_data/queries_indri/qid_query.txt -baseline=tfidf,k1:" + str(k1) + ",b:" + str(
        #             b) + " >data/runs/bm25/" + str(k1) + "_" + str(b) + ".txt"
        #         output = str(os.system(cmd))


