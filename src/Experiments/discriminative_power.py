from src.eval import Eval
import numpy as np
import random
from random import choices
from itertools import combinations
import math
import pprint
import itertools
import matplotlib.pyplot as plt
from trectools import TrecQrel, TrecRun, TrecEval

class BootstrapTest(Eval):

    def __init__(self):
        super(BootstrapTest, self).__init__()

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

    def eval_all_runs(self):
        #Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        for run in self.runs:
            if self.bootstrap_parameters['qrels_id'] == 'urbp':
                r1 = TrecRun(run)
                qrels_trectools = TrecQrel(self.bootstrap_parameters['qrels_file'])
                te = TrecEval(r1, qrels_trectools)
                rbp_per_query, residuals = te.get_rbp(per_query=True)
                rbp_per_query.fillna(0)
                res = rbp_per_query.to_dict()['RBP(0.80)@1000']
                res_fin = {}
                for qid in self.Q:
                    if qid not in res_fin:
                        res_fin[qid] = 0
                    if int(qid) in res:
                        res_fin[qid] = float(res[int(qid)])
                # dict_result[qrels[0]] = res_fin
                res = res_fin
            else:
                res = self.run_trec_eval(self.bootstrap_parameters['qrels_file'], run,
                                                self.bootstrap_parameters['metric'], queries=True)
            if run not in run_results:
                run_results[run] = []
            for qid in self.Q:
                if qid in res:
                    run_results[run] += [res[qid]]
                else:
                    # print("else", qid)
                    run_results[run] += [0.00]
        return run_results

    def eval_all_runs_baselines(self, runs, qrels, metric, set_Q, queries=True):
        #Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        for run in runs:
            res = self.run_trec_eval(qrels, run, metric, queries=queries)
            if run not in run_results:
                run_results[run] = []
            for qid in set_Q:
                if qid in res:
                    run_results[run] += [res[qid]]
                else:
                    # print("else", qid)
                    run_results[run] += [0.00]
        return run_results


    def get_run_pairs(self, list_of_runs):
        return list(combinations(list_of_runs, 2))

    def create_bootstrap_samples(self, z_mean, z):
        pass

    def compute_t_test(self, arr):
        tz = 0.0
        if np.std(arr) != 0:
            tz = np.mean(arr)/(np.std(arr)/math.sqrt(len(arr)))
        else:
            tz = 0.0
        return tz

    def run_bootstrap_test(self, bootstrap_parameters):

        self.bootstrap_parameters = bootstrap_parameters

        # Get list of Runs
        self.runs = self.get_list_files(self.bootstrap_parameters['runs_folder'])

        # Get set of Topics
        self.Q = self.get_set_of_topics(self.runs)
        # pprint.pprint(self.Q)
        # print(len(self.Q))
        # Eval all runs
        run_results = self.eval_all_runs()
        # pprint.pprint(run_results)
        # Compute possible pairs of systems #
        possible_runs_pair = self.get_run_pairs(self.runs)
        dict_asls = {}

        for pair in possible_runs_pair:
            if pair not in dict_asls:
                dict_asls[pair] = {}
            # print(pair)
            # print("Y", run_results[pair[0]])
            # print("X", run_results[pair[1]])
            z = np.subtract(run_results[pair[0]], run_results[pair[1]])
            z_mean = np.mean(z)
            # print("z", z, "z_mean", z_mean)
            tz = self.compute_t_test(z)
            # print("Tz:", tz)
            count = 0
            arr_z_means = [z_mean]*len(z)
            # print('arr_z_means', arr_z_means)
            w = np.subtract(z, arr_z_means)
            # print("w", w)
            b_vals = []
            for b in range(self.bootstrap_parameters['B']):

                w_star_b = choices(list(w), k= len(w))
                t_b = self.compute_t_test(w_star_b)

                if math.fabs(t_b) >= math.fabs(tz):
                    count += 1
                b_vals += [(t_b, np.fabs(np.mean(w_star_b)))]

            ASL = count/float(self.bootstrap_parameters['B'])
            dict_asls[pair] = {"ASL": ASL, "b_vals": b_vals}

        return dict_asls


    def run_bootstrap_from_run_results(self, run_results, B, runs):

        # Compute possible pairs of systems #
        possible_runs_pair = self.get_run_pairs(runs)
        dict_asls = {}

        for pair in possible_runs_pair:
            if pair not in dict_asls:
                dict_asls[pair] = {}
            # print(pair)
            # print("Y", run_results[pair[0]])
            # print("X", run_results[pair[1]])
            # exit()
            z = np.subtract(run_results[pair[0]], run_results[pair[1]])
            z_mean = np.mean(z)
            # print("z", z, "z_mean", z_mean)
            tz = self.compute_t_test(z)
            # print("Tz:", tz)
            count = 0
            arr_z_means = [z_mean] * len(z)
            # print('arr_z_means', arr_z_means)
            w = np.subtract(z, arr_z_means)
            # print("w", w)
            b_vals = []
            for b in range(B):

                w_star_b = choices(list(w), k=len(w))
                t_b = self.compute_t_test(w_star_b)

                if math.fabs(t_b) >= math.fabs(tz):
                    count += 1
                b_vals += [(t_b, np.fabs(np.mean(w_star_b)))]

            ASL = count / float(B)
            dict_asls[pair] = {"ASL": ASL, "b_vals": b_vals}
        # pprint.pprint(dict_asls)
        return dict_asls


class DiscriminativePower(BootstrapTest):
    def __init__(self, multiple_parameters=None, multiple_parameters_baseline=None):
        # self.discriminative_power_params = self.get_order_parameters("config/discriminative.json")
        super(DiscriminativePower, self).__init__()

        if multiple_parameters is None:
            multiple_parameters = {
                'qrels_files': [
                    ('chebyshev', 'data/qrels_webtrack2013/qrels_webtrack2013_chebyshev.txt'),
                    ('euclidean', 'data/qrels_webtrack2013/qrels_webtrack2013_euclidean.txt'),
                    ('manhattan', 'data/qrels_webtrack2013/qrels_webtrack2013_manhattan.txt'),
                    # ('skyline', 'data/qrels_decision/qrels_decision_skyline.txt'),
                    # ('rel', 'data/qrels_webtrack2013/qrels_only_relevance.txt'),
                    # ('cred', 'data/qrels_webtrack2013/qrels_only_pagerank.txt'),
                    # ('correc', 'data/qrels_webtrack2013/qrels_only_spam.txt')
                ],
                'runs_folder': [
                    ('WebTrack 2013', 'data/webtrack2013Runs_extracted/'),
                    # ('ASL_track2016', 'data/runs_track_newids/2016/'),
                    # ('ASL_Topruns','data/quick_solution/')
                ],
                'metrics': ['ndcg'],
                'B': [10000]
            }
        if multiple_parameters_baseline is None:
            multiple_parameters_baseline = {
                'qrels_files': [
                    ('rel', 'data/qrels_webtrack2013/qrels_only_relevance.txt'),
                    ('cred', 'data/qrels_webtrack2013/qrels_only_pagerank.txt'),
                    ('correc', 'data/qrels_webtrack2013/qrels_only_spam.txt')
                ],
                'runs_folder': [
                    ('WebTrack 2013', 'data/webtrack2013Runs_extracted/'),
                    # ('ASL_Topruns','data/quick_solution/')
                ],
                'metrics': 'ndcg',
                'B': 10000
            }

        self.file_out = open(multiple_parameters['file_out'], 'w')
        ## This is to run for multiple parameters ##
        self.multiple_computation_discriminative_power(multiple_parameters)

        ## This is to run for baseline -- Additional work on results ##
        self.discriminative_power_baselines(multiple_parameters_baseline)
        self.file_out.close()
        # self.plot_ASL_curves(multiple_parameters)

    def multiple_computation_discriminative_power(self, multiple_parameters):
        """
            If you want to vary the different bootstrap_parameters changing
            qrels, runs folders, B value, etc., you can do it using this function
        """

        for run_folder in multiple_parameters['runs_folder']:
            print('\multicolumn{6}{l}{',run_folder[0], '} \\\ \midrule')
            print('\multicolumn{6}{l}{',run_folder[0], '} \\\ \midrule',file=self.file_out)
            for qrels in multiple_parameters['qrels_files']:  # This is an position of the array
                qrels_id = qrels[0]
                qrels_path = qrels[1]
                for metric in multiple_parameters['metrics']:
                    for B in multiple_parameters['B']:
                        bootstrap_parameters = {'qrels_file': qrels_path,
                                'runs_folder':run_folder[1],
                                'qrels_id':qrels_id,
                                'metric':metric,
                                'B': B}
                        sensitivity_arr = []
                        estimated_diff_arr = []
                        # for alpha in [0.05, 0.01]:
                        for alpha in [0.01]:
                            sensitivity, estimated_diff = self.compute_discriminative_power(bootstrap_parameters, alpha)
                            estimated_diff_arr += [round(estimated_diff,2)]
                            sensitivity_arr += [str(round(sensitivity*100,2))+r"\%"]
                        print(r'\multicolumn{1}{c|}{\textbf{',qrels_id+'('+metric+')}}','&', sensitivity_arr[0],'&', estimated_diff_arr[0],r'\\')
                        print(r'\multicolumn{1}{c|}{\textbf{',qrels_id+'('+metric+')}}','&', sensitivity_arr[0],'&', estimated_diff_arr[0],r'\\',file=self.file_out)
                        # print(r'\multicolumn{1}{c|}{\textbf{',qrels_id+'('+metric+')}}','&', sensitivity_arr[0],'&', estimated_diff_arr[0],'& &', sensitivity_arr[1],'&', estimated_diff_arr[1],r'\\')
            print('\midrule')

    def discriminative_power_baselines(self, multiple_parameters_baseline):
        """
                    If you want to vary the different bootstrap_parameters changing
                    qrels, runs folders, B value, etc., you can do it using this function
                """
        baseline_measures = {'CAM':self.compute_cam_run_results, 'MM':self.compute_mm_run_results}

        for run_folder in multiple_parameters_baseline['runs_folder']:
            print('\multicolumn{6}{l}{', run_folder[0], '} \\\ \midrule')
            print('\multicolumn{6}{l}{', run_folder[0], '} \\\ \midrule', file=self.file_out)
            for measure, function_measure in baseline_measures.items():
                runs = self.get_list_files(run_folder[1])
                topics = self.get_set_of_topics(runs)
                run_results = function_measure(runs, multiple_parameters_baseline['qrels_files'], multiple_parameters_baseline['metrics'], topics)
                self.dict_asls_by_pair = self.run_bootstrap_from_run_results(run_results, multiple_parameters_baseline['B'], runs)
                sensitivity_arr = []
                estimated_diff_arr = []
                # for alpha in [0.05, 0.01]:
                for alpha in [0.01]:
                    DIFF = []
                    val = 0
                    for pair in self.dict_asls_by_pair:
                        sorted_b_vals = sorted(self.dict_asls_by_pair[pair]['b_vals'], key=lambda x: x[0], reverse=True)
                        DIFF += [sorted_b_vals[int(multiple_parameters_baseline['B'] * alpha)][1]]
                        if float(self.dict_asls_by_pair[pair]['ASL']) < alpha:
                            val += 1
                    sensitivity = val / float(len(self.dict_asls_by_pair))
                    estimated_diff = max(DIFF)
                    estimated_diff_arr += [round(estimated_diff, 2)]
                    sensitivity_arr += [str(round(sensitivity * 100, 2)) + r"\%"]
                # print(r'\multicolumn{1}{c|}{\textbf{', measure + '(' + multiple_parameters_baseline['metrics'] + ')}}', '&', sensitivity_arr[0],
                #           '&', estimated_diff_arr[0], '& &', sensitivity_arr[1], '&', estimated_diff_arr[1], r'\\')
                print(r'\multicolumn{1}{c|}{\textbf{', measure + '(' + multiple_parameters_baseline['metrics'] + ')}}',
                      '&', sensitivity_arr[0],
                      '&', estimated_diff_arr[0], r'\\')
                print(r'\multicolumn{1}{c|}{\textbf{', measure + '(' + multiple_parameters_baseline['metrics'] + ')}}',
                      '&', sensitivity_arr[0],
                      '&', estimated_diff_arr[0], r'\\', file=self.file_out)

    def compute_cam_run_results(self, runs, qrels, metric, set_Q):
        #Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        run_results_cam = {}
        set_Q = sorted(set_Q)
        # print("topics", set_Q)
        for run in runs:
            for aspect_id, qrel in enumerate(qrels):
                # print(aspect_id, qrel)
                res = self.run_trec_eval(qrel[1], run, metric, queries=True)
                # print("Res:",res)
                if run not in run_results:
                    run_results[run] = {}
                if run not in run_results_cam:
                    run_results_cam[run] = []
                if aspect_id not in run_results[run]:
                    run_results[run][aspect_id] = []
                for qid in set_Q:
                    # print(qid)
                    if qid in res:
                        run_results[run][aspect_id] += [res[qid]]
                    else:
                        # print("else", qid)
                        run_results[run][aspect_id] += [0.00]
            # print("Results",run_results['data/runs_track_newids/2016/udelRun6C'][0])
            # exit()
            for qid in range(len(set_Q)):
                # print([run_results[run][aspect_id][qid] for aspect_id in range(len(qrels))])
                run_results_cam[run] += [np.sum([run_results[run][aspect_id][qid] for aspect_id in range(len(qrels))])/len(qrels)]
        return run_results_cam


    def compute_mm_run_results(self, runs, qrels, metric, set_Q):
        # Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        run_results_cam = {}
        set_Q = sorted(set_Q)
        # print("topics", set_Q)
        for run in runs:
            for aspect_id, qrel in enumerate(qrels):
                # print(aspect_id, qrel)
                res = self.run_trec_eval(qrel[1], run, metric, queries=True)
                # print("Res:",res)
                if run not in run_results:
                    run_results[run] = {}
                if run not in run_results_cam:
                    run_results_cam[run] = []
                if aspect_id not in run_results[run]:
                    run_results[run][aspect_id] = []
                for qid in set_Q:
                    # print(qid)
                    if qid in res:
                        run_results[run][aspect_id] += [res[qid]]
                    else:
                        # print("else", qid)
                        run_results[run][aspect_id] += [0.00]
            # print("Results",run_results['data/runs_track_newids/2016/udelRun6C'][0])
            # exit()
            for qid in range(len(set_Q)):
                # print([run_results[run][aspect_id][qid] for aspect_id in range(len(qrels))])
                a = np.sum([run_results[run][aspect_id][qid]*(1/len(qrels)) for aspect_id in range(len(qrels))])
                if 0 in [run_results[run][aspect_id][qid] for aspect_id in range(len(qrels))]:
                    run_results_cam[run] += [0]
                else:
                    summ = 0
                    for aspect_id in range(len(qrels)):
                        if run_results[run][aspect_id][qid] == 0:
                            summ += 0
                        else:
                            summ += (1/len(qrels))/run_results[run][aspect_id][qid]
                    run_results_cam[run] += [1/summ]
        return run_results_cam


    def compute_discriminative_power(self, bootstrap_parameters, alpha):
        DIFF = []
        val=0
        self.dict_asls_by_pair = self.run_bootstrap_test(bootstrap_parameters)
        for pair in self.dict_asls_by_pair:
            sorted_b_vals = sorted(self.dict_asls_by_pair[pair]['b_vals'], key=lambda x: x[0], reverse=True)
            DIFF += [sorted_b_vals[int(bootstrap_parameters['B']*alpha)][1]]
            if float(self.dict_asls_by_pair[pair]['ASL']) < alpha:
                val +=1
        sensitivity = val/float(len(self.dict_asls_by_pair))
        estimated_diff = max(DIFF)
        # print("Sensitivity:", sensitivity, "EstimatedDIff:", estimated_diff)
        return sensitivity, estimated_diff

    def plot_ASL_curves(self, multiple_parameters):
        for run_folder in multiple_parameters['runs_folder']:
            dict_asls_by_qrels = {}
            print(run_folder)
            for qrels in multiple_parameters['qrels_files']:  # This is an position of the array
                qrels_id = qrels[0]
                qrels_path = qrels[1]
                arr = []
                if qrels_id not in dict_asls_by_qrels:
                    dict_asls_by_qrels[qrels_id] = []
                for metric in multiple_parameters['metrics']:
                    for B in multiple_parameters['B']:
                        bootstrap_parameters = {'qrels_file': qrels_path,
                                                'runs_folder': run_folder[1],
                                                'metric': metric,
                                                'B': B}
                        self.dict_asls_by_pair = self.run_bootstrap_test(bootstrap_parameters)
                        for pair in self.dict_asls_by_pair:
                            arr += [self.dict_asls_by_pair[pair]['ASL']]
                        arr = sorted(arr, reverse=True)
                        dict_asls_by_qrels[qrels_id] = arr
            line_styles = ['.-', '.--', '.:', '.-.', '*-', '+--', 'x:', 'o-.']
            for i, (k, v) in enumerate(dict_asls_by_qrels.items()):
                plt.plot(range(1, len(v) + 1), v, line_styles[i], label=k, markersize=1.0, linewidth=0.5)
            plt.legend()  # To draw legend
            plt.xlabel('system pair sorted by ASL')
            plt.ylabel('achieved significance level (ASL)')
            plt.show()
            plt.savefig('data/dataset_statistics/'+run_folder[0]+'.png')
            plt.clf()
