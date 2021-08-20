import gzip
from collections import OrderedDict
import platform
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('agg')
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from src.params import *
from collections import Counter
import math
import pandas as pd
from operator import itemgetter
import numpy as np
from itertools import combinations
from itertools import permutations
from src.eval import Eval
import pprint

class DatasetStatistics(Params):
    def __init__(self):
        super(DatasetStatistics, self).__init__()
        # self.dataset_parameters = self.get_order_parameters(qrels_config="config/dataset.json")

    def plot_pagerank_distribution(self):
        dict_docid_aspects = {}
        file_pr = open(self.dataset_parameters['pagerank_file'], "r")
        pr_arr = []
        min_pagerank = 10000000
        fig = plt.figure()
        max_pagerank = -999999
        while True:
            line = file_pr.readline()
            if not line:
                break
            docid = line.rstrip().split()[0]
            pagerank = float(line.rstrip().split()[1])
            if pagerank > max_pagerank:
                max_pagerank = pagerank
            if pagerank < min_pagerank:
                min_pagerank = pagerank
            """Assing pagerank values to documents in qrels"""
            pr_arr += [pagerank]
        file_pr.close()
        x, y = [], []
        frequency_dict = (Counter(pr_arr))
        sm = sum(frequency_dict.values())
        frequencies = {key: float(value) / sm for (key, value) in frequency_dict.items()}
        sorted_frequencies = OrderedDict(sorted(frequencies.items(), key=lambda t: t[0]))
        log_min_pagerank = math.log2(min_pagerank)
        log_max_pagerank = math.log2(max_pagerank)
        for key,value in frequency_dict.items():
            x.append((math.log2(key) - log_min_pagerank) / (log_max_pagerank - log_min_pagerank))
            y.append(math.log2(value))
        plt.plot(x, y, 'ro', markersize=0.5)
        plt.title('PageRank distribution across documents of the collection.')
        plt.xlabel('Pagerank value')
        plt.ylabel('log2(number of documents)')
        fig.savefig('data/dataset_statistics/pagerank_distribution.png')

    def plot_spamscore_distribution(self):
        files_to_unzip = self.get_list_files(self.dataset_parameters['spam_score_folder'])
        pr_arr = []
        min_pagerank = 10000000
        fig = plt.figure()
        max_pagerank = -999999
        for file in files_to_unzip:
            with gzip.open(file, 'r') as fin:
                content = fin.readlines()
            content = [x.rstrip().decode('utf-8') for x in content]
            for line in content:
                parts = line.split()
                spam_score = int(parts[0])
                pr_arr += [spam_score]
        x, y = [], []
        frequency_dict = (Counter(pr_arr))
        sm = sum(frequency_dict.values())

        sorted_frequencies = OrderedDict(sorted(frequency_dict.items(), key=lambda t: t[0]))
        print(sorted_frequencies)
        for key, value in sorted_frequencies.items():
            y.append(math.log2(value))
            x.append(key)
        plt.plot(x, y, 'ro')
        plt.title('SpamScore distribution across documents of the collection.')
        plt.xlabel('SpamScore value')
        plt.ylabel('log2(number of documents)')
        fig.savefig('data/dataset_statistics/spamscore_distribution.png')

    def plot_histogram_qrels(self):
        df = pd.read_csv("data/qrels_decision/qrels_decision_3aspects.txt", sep=' ', names=['qid','it','docid','relevance','cred','correc'], header=None)
        for col in ['relevance','cred','correc']:
            frequency_dict = (Counter(df[col].values))
            print(col)
            print(frequency_dict)
            for k,v in frequency_dict.items():
                print(k,(v/sum(frequency_dict.values()))*100)
            # print(col,frequency_dict)
        # df.hist(column='relevance')
        # plt.show()

    def get_file_content(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        return content

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def get_total_number_documents_assessed(self):
        df = pd.read_csv("data/processed_data/qrels/qrels_2016_4aspects.txt", sep=' ',
                         names=['qid', 'it', 'docid', 'relevance', 'usefulness', 'popularity', 'credibility'],
                         header=None)
        print(df)
        print("unique:",df['docid'].nunique())


    """SHOWING LIMITATIONS OF BASELINE MEASURES"""

    def get_total_number_docs_misplaced(self):
        with open("data/processed_data/qrels/qrels_2016_4aspects.txt") as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        dict_qrels_list_tuples = {}
        num_aspects = 0
        aspects = [0,1,2,3]
        multi_comb = list(permutations(aspects, len(aspects)))

        for line in content:
            parts = line.split()
            if parts[0] not in dict_qrels_list_tuples:
                dict_qrels_list_tuples[parts[0]] = []
            tuples_weights = tuple([int(x) for x in parts[3:]])
            num_aspects = len(parts[3:])
            dict_qrels_list_tuples[parts[0]].append(tuples_weights)
        for qid in dict_qrels_list_tuples:
            misplaced = 0
            total = 0
            test = []
            for comb in multi_comb:
                test = dict_qrels_list_tuples[qid]
                for i in comb:
                    test.sort(key=itemgetter(i), reverse=True)

                for i in range(len(test)-1):
                    a = np.subtract(test[i], test[i+1])
                    total+=1
                    if min(a) >= 0:
                        continue
                    else:
                        misplaced+=1
            print(qid, "Misplaced: ", misplaced/total)
            dict_qrels_list_tuples[qid] = misplaced/total

    def get_number_zero_topics_by_measure(self):
        files = self.get_list_files("data/results/Decision/")
        dict_count = {'MM':0, 'CAM':0, 'chebyshev':0, 'euclidean':0, 'skyline':0}
        for file in files:
            df = pd.read_csv(file)
            num_qids = len(df['qid'])
            for column in df.columns[1:]:
                if column in dict_count:
                    dict_count[column] += int(list(df[column].values).count(0))
        for measure, valor in dict_count.items():
            dict_count[measure] = valor/(num_qids*len(files))
            print(measure, str(round(dict_count[measure]*100,2))+"%")

    def generate_individualIdealRankings(self, qrels_path='data/processed_data/qrels/qrels_discrete_aspects.txt',outpath='',num_aspects=4):
        with open(qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        num_aspects = num_aspects
        aspects = list(range(num_aspects))
        multi_comb = list(permutations(aspects, len(aspects)))
        dict_docs = {}

        for count, comb in enumerate(multi_comb):
            dict_qrels_list_tuples = {}
            for line in content:
                parts = line.split()
                if parts[0] not in dict_qrels_list_tuples:
                    dict_qrels_list_tuples[parts[0]] = []
                    dict_docs[parts[0]] = []
                tuples_weights = tuple([int(x) for x in parts[3:]]) + tuple([parts[2]])
                num_aspects = len(parts[3:])
                dict_qrels_list_tuples[parts[0]].append(tuples_weights)
                dict_docs[parts[0]].append(parts[2])
            "1	Q0	clueweb12-0105wb-74-23095	5	1.0	TOPIC_RUN2_TC"
            file_out = open(outpath+"individual/"+str(comb), 'w')
            for qid in dict_qrels_list_tuples:
                test = []

                test = dict_qrels_list_tuples[qid]
                # print(test)
                test.sort(key=itemgetter(num_aspects))
                # print(test)
                docs = dict_docs[qid]
                # exit()
                for i in comb:
                    test.sort(key=itemgetter(i), reverse=True)
                for score, doc in enumerate(test):
                    print(qid,'Q0',doc[num_aspects],score+1,len(test)-score,'run',doc, file=file_out)
            file_out.close()

    def generate_idealRuns_Single(self, qrels_path = "data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query.txt", out_path="data/runs_test_limitations/manhattan.txt", aspects=''):
        with open(qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        num_aspects = 0

        multi_comb = list(permutations(aspects, len(aspects)))
        dict_docs = {}
        dict_qrels_list_tuples = {}
        for line in content:
            parts = line.split()
            if parts[0] not in dict_qrels_list_tuples:
                dict_qrels_list_tuples[parts[0]] = []
                dict_docs[parts[0]] = []
            tuples_weights = tuple([int(x) for x in parts[3:]]) + tuple([parts[2]])
            num_aspects = len(parts[3:])
            dict_qrels_list_tuples[parts[0]].append(tuples_weights)
            dict_docs[parts[0]].append(parts[2])
        "1	Q0	clueweb12-0105wb-74-23095	5	1.0	TOPIC_RUN2_TC"
        file_out = open(out_path,'w')
        for qid in dict_qrels_list_tuples:
            test = []

            test = dict_qrels_list_tuples[qid]
            # print(test)
            test.sort(key=itemgetter(1))
            test.sort(key=itemgetter(0), reverse=True)
            # print(test)
            docs = dict_docs[qid]
            # exit()
            for score, doc in enumerate(test):
                print(qid,'Q0',doc[1],score+1,len(test)-score,'run',doc, file=file_out)
                # print(qid,'Q0',doc[1],score+1,len(test)-score,'run',doc)
        file_out.close()

    def plot_scores_by_topic(self, dict_parameters):
        # dict_parameters = {'qrels_arr':["data/qrels_decision/qrels_decision_rel.txt",
        #                                 "data/qrels_decision/qrels_decision_cred.txt",
        #                                 "data/qrels_decision/qrels_decision_correctness.txt",
        #                                 ],
        #                    'runs_folder':'data/runs_test_limitations_decision/individual/',
        #                    'topic_range':[1, 51],
        #                    'single_runs':{'chebyshev':("data/qrels_decision/qrels_decision_chebyshev.txt", "data/runs_test_limitations_decision/chebyshev.txt"),
        #                                   'euclidean':("data/qrels_decision/qrels_decision_euclidean.txt", "data/runs_test_limitations_decision/euclidean.txt"),
        #                                   'manhattan':("data/qrels_decision/qrels_decision_manhattan.txt", "data/runs_test_limitations_decision/manhattan.txt")}
        #                    }
        # dict_parameters = {'qrels_arr':["data/qrels_webtrack2014/qrels_only_relevance.txt",
        #                                 "data/qrels_webtrack2014/qrels_only_pagerank.txt",
        #                                 "data/qrels_webtrack2014/qrels_only_spam.txt",
        #                                 ],
        #                    'runs_folder':'data/run_test_limitations_webtrack2014/individual/',
        #                    'topic_range':[1, 51],
        #                    'single_runs':{'chebyshev':("data/qrels_webtrack2014/qrels_webtrack2014_chebyshev.txt", "data/run_test_limitations_webtrack2014/chebyshev.txt"),
        #                                   'euclidean':("data/qrels_webtrack2014/qrels_webtrack2014_euclidean.txt", "data/run_test_limitations_webtrack2014/euclidean.txt"),
        #                                   'manhattan':("data/qrels_webtrack2014/qrels_webtrack2014_manhattan.txt", "data/run_test_limitations_webtrack2014/manhattan.txt")}
        #                    }
        # dict_parameters = {'qrels_arr': ["/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_misinfo/qrels/misinfo-2020-qrels_first",
        #                                  "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_misinfo/qrels/misinfo-2020-qrels_second",
        #                                  "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_misinfo/qrels/misinfo-2020-qrels_third",
        #                                  ],
        #                    'runs_folder': '/science/image/cluster-homedirs/krn788/LLMS_multieval/data/run_test_limitations_misinformation/individual/',
        #                    # 'topic_range': [36, 85],
        #                    # 'single_runs': {'chebyshev': ("data/processed_data/qrels_chebyshev_iteration_qrels_only_query.txt",
        #                    #                               "data/runs_test_limitations_decision/chebyshev.txt"),
        #                    #                 'euclidean': ("data/processed_data/qrels_euclidean_iteration_qrels_only_query.txt",
        #                    #                               "data/runs_test_limitations_decision/euclidean.txt"),
        #                    #                 'manhattan': ("data/processed_data/qrels_manhattan_iteration_qrels_only_query.txt",
        #                    #                               "data/runs_test_limitations_decision/manhattan.txt")}
        #                    }
        eval = Eval()
        cam = []
        mm = []
        num_q = 0
        dict_results_by_run = {}
        count = 0
        count_all = 0
        count_cam = 0
        for run in self.get_list_files(dict_parameters['runs_folder']):
            # print(run)
            if run not in dict_results_by_run:
                dict_results_by_run[run] = {'cam':[], 'mm':[]}
            res = []
            for qrels in dict_parameters['qrels_arr']:
                # print(eval.run_trec_eval(qrels, run, "ndcg", queries=True))
                # exit()
                res += [eval.run_trec_eval(qrels, run, dict_parameters['metric'], queries=True)]
                # print(run, qrels.split('/')[-1], res)
            Q = sorted([int(x) for x in list(res[0].keys())])
            self.Q = Q

            # print(list(res[0].keys()))
            arr_res_cam = []
            arr_res_mm = []
            # for q in range(dict_parameters['topic_range'][0], dict_parameters['topic_range'][1]):
            for q in Q:
                results = [aspc[str(q)] for aspc in res]

                arr_res_cam += [np.mean(results)]
                count_all +=1
                if np.mean(results) == 0:
                    count_cam +=1
                if all(app > 0 for app in results):
                    sum = 0
                    for app in results:
                        sum += ((1/float(len(results)))/ app)
                    arr_res_mm += [1 / sum]
                else:
                    count+=1
                    arr_res_mm += [0.0]
            dict_results_by_run[run]['mm'] = arr_res_mm
            dict_results_by_run[run]['cam'] = arr_res_cam
        pprint.pprint(dict_results_by_run)
        self.print_runs_results_to_csv(dict_results_by_run, Q, dict_parameters['task'], dict_parameters['metric'])
        return
        dict_res = {}

        dict_res = {'mm':[], 'cam':[], 'chebyshev':[], 'manhattan':[], 'euclidean':[]}
        res_chebyshev = eval.run_trec_eval(dict_parameters['single_runs']['chebyshev'][0], dict_parameters['single_runs']['chebyshev'][1], "map", queries=True)
        res_manhattan = eval.run_trec_eval(dict_parameters['single_runs']['manhattan'][0], dict_parameters['single_runs']['manhattan'][1], "map", queries=True)
        res_euclidean = eval.run_trec_eval(dict_parameters['single_runs']['euclidean'][0], dict_parameters['single_runs']['euclidean'][1], "map", queries=True)
        # res_chebyshev = eval.run_trec_eval(dict_parameters['single_runs']['chebyshev'][0], dict_parameters['single_runs']['chebyshev'][1], "ndcg", queries=True)
        # res_manhattan = eval.run_trec_eval(dict_parameters['single_runs']['manhattan'][0], dict_parameters['single_runs']['manhattan'][1], "ndcg", queries=True)
        # res_euclidean = eval.run_trec_eval(dict_parameters['single_runs']['euclidean'][0], dict_parameters['single_runs']['euclidean'][1], "ndcg", queries=True)
        # exit()
        # res_chebyshev = eval.run_trec_eval(
        #     "data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query.txt",
        #     "data/runs_test_limitations/chebyshev.txt", "ndcg", queries=True)
        # res_manhattan = eval.run_trec_eval(
        #     "data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query.txt",
        #     "data/runs_test_limitations/manhattan.txt", "ndcg", queries=True)
        # res_euclidean = eval.run_trec_eval(
        #     "data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query.txt",
        #     "data/runs_test_limitations/euclidean.txt", "ndcg", queries=True)
        for qid, q in zip(Q, range(0, dict_parameters['topic_range'][1]- dict_parameters['topic_range'][0])):
            qid = str(qid)
            dict_res['mm'].append(np.mean([dict_results_by_run[run]['mm'][q] for run in dict_results_by_run]))
            dict_res['cam'].append(np.mean([dict_results_by_run[run]['cam'][q] for run in dict_results_by_run]))
            dict_res['chebyshev'].append(res_chebyshev[qid])
            dict_res['manhattan'].append(res_manhattan[qid])
            dict_res['euclidean'].append(res_euclidean[qid])
        pprint.pprint(dict_res)
        exit()
        for metric, arr in dict_res.items():
            plt.plot(range(1,len(arr)+1), arr, label=metric)
        plt.xlabel('Topics')
        plt.ylabel('Evaluation Measure Score')
        plt.legend(loc=1)
        plt.savefig('data/dataset_statistics/average_ideal_decision.png')

        print("CAM", count_cam / count_all)
        print("MM", count / count_all)
        from collections import Counter
        print(Counter(dict_res['chebyshev']))
        # print("CAM",np.mean(cam))
        # print("MM",np.mean(mm))

    def print_runs_results_to_csv(self, dict_results_by_run, Q, task, metric):
        # cam_file = open('/home/lucas/Metrics/LLMS_multieval/data/results_limitations/cam_WebTrack2009_results.csv', "w")
        # mm_file = open('/home/lucas/Metrics/LLMS_multieval/data/results_limitations/mm_WebTrack2009_results.csv', "w")
        if metric == 'map':
            cam_file = open('/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_limitations_ap/cam_'+task+'_results.csv', "w")
            mm_file = open('/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_limitations_ap/mm_'+task+'_results.csv', "w")
        else:
            cam_file = open(
                '/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_limitations/cam_' + task + '_results.csv',
                "w")
            mm_file = open(
                '/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_limitations/mm_' + task + '_results.csv',
                "w")
        # topics = list(range(1,len(dict_results_by_run[list(dict_results_by_run.keys())[0]]['mm'])+1))
        # topics = list(range(1,51))
        # print(len(topics))
        # print('runid',';'.join([str(x) for x in topics]),sep=';',file=cam_file)
        # print('runid', ';'.join([str(x) for x in topics]), sep=';', file=mm_file)

        print('runid', ';'.join([str(x) for x in Q]), sep=';', file=cam_file)
        print('runid', ';'.join([str(x) for x in Q]), sep=';', file=mm_file)

        for run in dict_results_by_run:
            print(run.split('/')[-1], ';'.join([str(x) for x in dict_results_by_run[run]['cam']]), sep=';', file=cam_file)
            print(run.split('/')[-1], ';'.join([str(x) for x in dict_results_by_run[run]['mm']]), sep=';', file=mm_file)
        mm_file.close()
        cam_file.close()



    def check_num_queries_zero_all(self):
        with open("data/qrels_decision/qrels_mapped.txt") as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        dict_check = {}
        for line in content:
            parts = line.split()
            if parts[0] not in dict_check:
                dict_check[parts[0]] = 0
            if not all(y == 0 for y in [int(x) for x in parts[3:]]):
                dict_check[parts[0]] += 1
        pprint.pprint(dict_check)


    def create_box_plot_format_csv(self, folder, folder_out):
        header = open(self.get_list_files(folder)[0]).readline().rstrip()
        df = pd.read_csv(self.get_list_files(folder)[0])
        metrics = header.split(',')[1:]
        header = "runid;"+';'.join([str(x) for x in df['qid'].values])
        files_out = [open(folder_out+metric,'w') for metric in metrics]
        for file_out in files_out:
            print(header, file=file_out)
        for file in self.get_list_files(folder):
            df = pd.read_csv(file)
            for metric, file_out in zip(metrics,files_out):
                print(file.split('/')[-1],';'.join([str(x) for x in df[metric].values]), sep=';', file=file_out)
        [f.close() for f in files_out]




if __name__ == '__main__':
    dt = DatasetStatistics()
    dt.generate_individualIdealRankings()
    dt.compute_cam_wham()