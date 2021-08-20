from src.eval import Eval
from pprint import pprint
from itertools import combinations
import pandas as pd
import os
class Intuitiveness(Eval):
    def __init__(self):
        super(Intuitiveness, self).__init__()
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
            'simple_metrics': {
                                "rel":{'metric':['Rprec'], 'qrels':'data/processed_data/qrels/qrels_only_rel.txt'},
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

    def compute_intuitiveness_test(self, m1, m2, pair_runs, set_topics):
        disagrement = 0
        correct_m1 = 0
        correct_m2 = 0
        total = 0
        for pair in pair_runs:
            # print(m1, m2, pair)
            run1 = pair[0].split(os.sep)[-1]
            run2 = pair[1].split(os.sep)[-1]
            for topic in range(len(set_topics)):
                # print(topic, self.run_results[run1][m1][topic])
                delta_m1 = self.run_results[run1][m1][topic] - self.run_results[run2][m1][topic]
                delta_m2 = self.run_results[run1][m2][topic] - self.run_results[run2][m2][topic]
                aspects = [x for x in self.run_results_simple[run1]]
                delta_simple = []
                if delta_m1 * delta_m2 < 0:
                    disagrement += 1
                    for aspect in aspects:
                        for m in self.run_results_simple[run1][aspect]:
                            delta_simple += [self.run_results_simple[run1][aspect][m][topic] - self.run_results_simple[run2][aspect][m][topic]]
                    if all((delta_m1*i) > 0 for i in delta_simple):
                        correct_m1 +=1
                    if all((delta_m2*i) > 0 for i in delta_simple):
                        correct_m2 +=1
                total +=1
        if disagrement == 0:
            return 'No disagreement', 'No disagreement', 0
        intuitiveness_m1 = correct_m1/disagrement
        intuitiveness_m2 = correct_m2/disagrement
        return intuitiveness_m1, intuitiveness_m2, round(disagrement/total*100,2)

    def experiment_intuitiveness(self):

        # Get list of Runs
        for run_folder, results_folder in zip(self.multiple_parameters['runs_folder'], self.multiple_parameters['results_folder']):
            print(run_folder)
            self.runs = self.get_list_files(run_folder[1])
            self.results_complex = self.get_list_files(results_folder[1])
            # Get set of Topics
            self.Q = self.get_set_of_topics(self.runs)
            self.run_results_simple = self.eval_all_runs_simple()
            posssible_runs_pair = self.get_run_pairs(self.runs)

            self.run_results = self.load_results_complex_measures(self.results_complex)
            pair_metrics = self.get_run_pairs(self.multiple_parameters['complex_metrics'])
            for pair in pair_metrics:
                c1, c2, disagreements = self.compute_intuitiveness_test(pair[0], pair[1], posssible_runs_pair, self.Q)
                print(pair,c1, c2, disagreements)
