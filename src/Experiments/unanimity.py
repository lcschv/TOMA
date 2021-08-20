from src.eval import Eval
from pprint import pprint
from itertools import combinations
import pandas as pd
import os
import math
class Unanimity(Eval):
    def __init__(self):
        super(Unanimity, self).__init__()
        self.multiple_parameters = {
            'qrels': [
                ('rel', 'data/processed_data/qrels/qrels_only_rel.txt'),
                ('use', 'data/processed_data/qrels/qrels_only_usefulness.txt'),
                ('pop', 'data/processed_data/qrels/qrels_only_popularity.txt'),
                ('cred', 'data/processed_data/qrels/qrels_only_credibility.txt')
            ],
            'runs_folder': [
                ('ASL_track2015', 'data/runs_track_newids/2015/'),
                ('ASL_track2016', 'data/runs_track_newids/2016/')
            ],
            'results_folder':[
                ('ASL_track2015', 'data/results/TaskTrack2015/'),
                ('ASL_track2016', 'data/results/TaskTrack2016/')
            ],
            'simple_metrics': {"rel":{'metric':['Rprec'], 'qrels':'data/processed_data/qrels/qrels_only_rel.txt'},
                               "use":{'metric':['Rprec'], 'qrels':'data/processed_data/qrels/qrels_only_usefulness.txt'},
                               "pop":{'metric':['Rprec'], 'qrels':'data/processed_data/qrels/qrels_only_popularity.txt'},
                               "cred":{'metric': ['Rprec'], 'qrels': 'data/processed_data/qrels/qrels_only_credibility.txt'}
                               },
                'complex_metrics':['euclidean','skyline', 'MM', 'CAM']
        }

    def get_set_of_topics(self, runs):
        Q = {}
        with open (runs[0]) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            qid = line.split()[0]
            if qid not in Q:
                Q[int(qid)] = ''
        return [str(q) for q in sorted(list(Q.keys()))]

    def eval_all_runs_simple(self):
        #Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        for run in self.runs:
            for aspect, qrels in self.multiple_parameters['simple_metrics'].items():
                for metric in self.multiple_parameters['simple_metrics'][aspect]['metric']:
                    res = self.run_trec_eval(qrels['qrels'], run, metric, queries=True)
                    run_name = run.split(os.sep)[-1]
                    if run_name not in run_results:
                        run_results[run_name] = {}
                    if aspect not in run_results[run_name]:
                        run_results[run_name][aspect] = {}
                    if metric not in run_results[run_name][aspect]:
                        run_results[run_name][aspect][metric] = []
                    for qid in self.Q:
                        if qid in res:
                            run_results[run_name][aspect][metric] += [res[qid]]
                        else:
                            print("else", qid)
                            run_results[run_name] += [0.00]
        return run_results

    def load_results_complex_measures(self, runs):
        run_results = {}
        for run in runs:
            df = pd.read_csv(run)
            run_name = run.split(os.sep)[-1].replace('.csv','')
            for metric in self.multiple_parameters['complex_metrics']:
                res = list(df[metric].values)
                if run_name not in run_results:
                    run_results[run_name] = {}
                if metric not in run_results[run_name]:
                    run_results[run_name][metric] = res
        return run_results

    def get_run_pairs(self, list_of_runs):
        return list(combinations(list_of_runs, 2))

    def compute_unanimity_test(self, m1, pair_runs, set_topics):
        disagrement = 0
        delta_mij = 0
        delta_msimple = 0
        delta_m_complex_and_simple = 0
        for pair in pair_runs:
            # print(m1, m2, pair)
            run1 = pair[0].split(os.sep)[-1]
            run2 = pair[1].split(os.sep)[-1]
            for topic in range(len(set_topics)):

                # If there is a change +1 otherwise 0.5
                if self.run_results[run1][m1][topic] == self.run_results[run2][m1][topic]:
                    delta_mij += 0.5
                else:
                    delta_mij += 1.0

                # Computing the difference between one run to another for the same measure
                delta_m1 = self.run_results[run1][m1][topic] - self.run_results[run2][m1][topic]

                aspects = [x for x in self.run_results_simple[run1]]
                delta_simple = []
                for aspect in aspects:
                    for m in self.run_results_simple[run1][aspect]:
                        delta_simple += [self.run_results_simple[run1][aspect][m][topic] - self.run_results_simple[run2][aspect][m][topic]]
                # print("ArrayDelta:", delta_simple)
                #If the simple metrics agree for all cases that run1 > run2 or run1 < run2
                if all((val) >= 0 for val in delta_simple) or all((val) <= 0 for val in delta_simple):
                    delta_msimple +=1
                # print("delta_msimple", delta_msimple)

                # If the simple metrics and complex metrics agree for all cases that run1 > run2 or run1 > run2
                if (all((val) >= 0 for val in delta_simple) and delta_m1 >=0) or (all((val) <= 0 for val in delta_simple)and delta_m1 <=0):
                    delta_m_complex_and_simple +=1
                # print("delta_m1",delta_m1,"\tdelta_m_complex_and_simple",delta_m_complex_and_simple )
        R = float(len(pair_runs))
        # print("MC agrees with all simple metrics (mMs)",delta_m_complex_and_simple/R)
        # print("Different and Equal Complex (mij)",(delta_mij/R))
        # print("All simple measures found improvement (Ms)",(delta_msimple/R))
        PMI = math.log((delta_m_complex_and_simple/(R*len(set_topics)))/((delta_mij/(R*len(set_topics)))*(delta_msimple/(R*len(set_topics)))),2)

        return PMI

    def experiment_unanimity(self):

        # Get list of Runs
        for run_folder, results_folder in zip(self.multiple_parameters['runs_folder'], self.multiple_parameters['results_folder']):
            self.runs = self.get_list_files(run_folder[1])
            self.results_complex = self.get_list_files(results_folder[1])
            # Get set of Topics
            self.Q = self.get_set_of_topics(self.runs)
            self.run_results_simple = self.eval_all_runs_simple()
            posssible_runs_pair = self.get_run_pairs(self.runs)
            print(run_folder)
            self.run_results = self.load_results_complex_measures(self.results_complex)
            for metric in self.multiple_parameters['complex_metrics']:
                print(metric)
                PMI = self.compute_unanimity_test(metric, posssible_runs_pair, self.Q)
                print("PMI:", PMI)
