from src.eval import Eval
from pprint import pprint
import numpy as np
from trectools import TrecQrel, TrecRun, TrecEval
import os
class Results(Eval):
    def __init__(self, multiple_parameters=None):
        super(Results, self).__init__()
        if multiple_parameters is None:
            self.multiple_parameters = {
                'qrels_files': [
                    ('chebyshev', 'data/qrels_webtrack2014/qrels_webtrack2014_chebyshev.txt'),
                    ('euclidean', 'data/qrels_webtrack2014/qrels_webtrack2014_euclidean.txt'),
                    ('manhattan', 'data/qrels_webtrack2014/qrels_webtrack2014_manhattan.txt'),
                    # ('skyline', 'data/processed_data/mapped_qrels/qrels_skylineorder_qrels_only_query.txt'),
                    ('rel', 'data/qrels_webtrack2014/qrels_only_relevance.txt'),
                    ('cred', 'data/qrels_webtrack2014/qrels_only_pagerank.txt'),
                    ('correc', 'data/qrels_webtrack2014/qrels_only_spam.txt'),
                    # ('cred', 'data/processed_data/qrels/qrels_only_credibility.txt')
                ],
                'runs_folder': [
                    # ('Decision', 'data/decisionRuns/'),
                    ('WebTrack2014', 'data/webtrack2014Runs_extracted/'),
                    # ('WebTrack2014', 'data/webtrack2014Runs_extracted/'),
                    # ('TaskTrack2016', 'data/runs_track_newids/2016/'),
                    # ('TaskTrack2016', 'data/runs_track_newids/2016/'),
                ],
                'results_path':['data/results/WebTrack2014/'],
                'aspects':['rel','cred','correc'],
                'metrics': 'ndcg',
            }
        else:
            self.multiple_parameters = multiple_parameters

    def MM(self, arr):
        result = 0
        a = np.sum([(val * (1/float(len(arr)))) for val in arr])
        if 0 in arr:
            return 0.0
        else:
            summ = 0
            for val in arr:
                if val == 0:
                    summ += 0
                else:
                    summ += ((1/float(len(arr))) / val)
            result = 1 / summ
        return result

    def generate_results(self):
        for folder in self.multiple_parameters['runs_folder']:
            runs = self.get_list_files(folder[1])
            set_topics = self.get_set_of_topics(self.multiple_parameters['qrels_files'][0][1])
            # print(set_topics)
            for run in runs:
                print(run.split(os.sep)[-1])
                file_out = open(self.multiple_parameters['results_path']+run.split(os.sep)[-1]+'.csv', 'w')
                dict_result = {}
                print('qid', ','.join([qrels[0] for qrels in self.multiple_parameters['qrels_files']]),'CAM',"MM", sep=',',file=file_out)
                for qrels in self.multiple_parameters['qrels_files']:
                    if qrels[0] not in dict_result:
                        dict_result[qrels[0]] = {}
                    if qrels[0] == 'urbp':
                        r1 = TrecRun(run)
                        qrels_trectools = TrecQrel(qrels[1])
                        te = TrecEval(r1, qrels_trectools)
                        rbp_per_query, residuals = te.get_rbp(per_query=True)
                        rbp_per_query.fillna(0)
                        res = rbp_per_query.to_dict()['RBP(0.80)@1000']
                        res_fin = {}
                        for qid in set_topics:
                            if qid not in res_fin:
                                res_fin[qid] = 0
                            if int(qid) in res:
                                res_fin[qid] = res[int(qid)]
                        dict_result[qrels[0]] = res_fin
                    else:
                        dict_result[qrels[0]] = self.run_trec_eval(qrels[1], run, self.multiple_parameters['metrics'], queries=True)
                for qid in set_topics:
                    aspects = self.multiple_parameters['aspects']
                    cam = round(np.average([dict_result[aspect][qid] for aspect in aspects]),4)
                    mm = round(self.MM([dict_result[aspect][qid] for aspect in aspects]),4)
                    print(qid, ','.join([str(dict_result[qrels[0]][qid]) for qrels in self.multiple_parameters['qrels_files']]), cam, mm, sep=',', file=file_out)
                file_out.close()


    def get_set_of_topics(self, runs):
        Q = {}
        with open (runs) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            qid = line.split()[0]
            if qid not in Q:
                Q[int(qid)] = ''
        return [str(q) for q in sorted(list(Q.keys()))]