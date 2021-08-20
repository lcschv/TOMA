from src.eval import Eval
import os
from os import listdir
from os.path import isfile, join
import shutil
from src.partial_order_relation.partial_order import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import math
import matplotlib


class PoolDownsampling(Eval):
    def __init__(self):
        super(PoolDownsampling, self).__init__()

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def pre_process(self, qrels_path, step=1, folder_new= 'data/downsample_qrels_2015/'):
        with open(qrels_path) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        number_values_to_remove = ((len(content)*step)/100)
        if not os.path.exists(folder_new+'qrels_removed/'):
            os.mkdir(folder_new+'qrels_removed/')
        else:
            shutil.rmtree(folder_new+'qrels_removed/')
            os.mkdir(folder_new + 'qrels_removed/')
        for i in range(0, 100, step):
            ### Make a copy to keep the original qrels without values removed ###
            if i == 0:
                new_qrels_file = open(folder_new + "qrels_removed/" + str(i), 'w')
                for line in content:
                    print(line, file=new_qrels_file)
                new_qrels_file.close()
                continue
            if len(content)-1 > round(number_values_to_remove):
                pos_to_remove = np.random.randint(len(content)-1, size=round(number_values_to_remove))
                content = np.delete(content, pos_to_remove)
            new_qrels_file = open(folder_new+"qrels_removed/"+str(i), 'w')
            for line in content:
                print(line, file=new_qrels_file)
            new_qrels_file.close()

    def compute_partial_order(self, folder_new='data/downsample_qrels_2015/', num_aspects=4):
        mult_qrels = self.get_list_files(folder_new+'qrels_removed/')
        dict_parameters_distance = {
            "qrels_file": '',
            "new_qrels_path": '',
            "from_qrels": False,
            "number_of_aspects": num_aspects,
            "greatest_label": [2, 2, 2, 2],
            "minimum_label": [0, 0, 0, 0],
            "mapping": "iteration",
            "distance_metric": ''
        }

        for folder in ['chebyshev', 'manhattan', 'euclidean', 'rel', 'use', 'pop', 'cred']:
            if not os.path.exists(folder_new + folder + '/'):
                os.mkdir(folder_new + folder + '/')
            else:
                shutil.rmtree(folder_new + folder + '/')
                os.mkdir(folder_new + folder + '/')

        """Generate qrels for partial order"""
        for qrels in mult_qrels:
            for metric in ['chebyshev', 'manhattan', 'euclidean']:
                dict_parameters_distance['qrels_file'] = qrels
                dict_parameters_distance['new_qrels_path'] = folder_new + metric+'/'+qrels.split('/')[-1]
                dict_parameters_distance['distance_metric'] = metric
                DistanceOrder(order_parameters=dict_parameters_distance)

            """Generate qrels for partial order"""
            with open(qrels) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]

            rel_file = open(folder_new + 'rel/'+qrels.split('/')[-1], 'w')
            use_file = open(folder_new + 'use/'+qrels.split('/')[-1], 'w')
            pop_file = open(folder_new + 'pop/'+qrels.split('/')[-1], 'w')
            cred_file = open(folder_new + 'cred/'+qrels.split('/')[-1], 'w')

            for line in content:
                parts = line.split()
                print(parts[0], parts[1], parts[2], parts[3], file=rel_file)
                print(parts[0], parts[1], parts[2], parts[4], file=use_file)
                print(parts[0], parts[1], parts[2], parts[5], file=pop_file)
                print(parts[0], parts[1], parts[2], parts[6], file=cred_file)

            rel_file.close()
            use_file.close()
            pop_file.close()
            cred_file.close()
        shutil.rmtree(folder_new+'qrels_removed/')

    def compute_partial_order_decision(self, folder_new='data/downsample_qrels_decision/', num_aspects=3):
        mult_qrels = self.get_list_files(folder_new+'qrels_removed/')
        dict_parameters_distance = {
            "qrels_file": '',
            "new_qrels_path": '',
            "from_qrels": False,
            "number_of_aspects": num_aspects,
            "greatest_label": [2, 2, 2],
            "minimum_label": [0, 0, 0],
            "mapping": "iteration",
            "distance_metric": ''
        }

        for folder in ['chebyshev', 'manhattan', 'euclidean', 'rel', 'cred', 'correctness']:
            if not os.path.exists(folder_new + folder + '/'):
                os.mkdir(folder_new + folder + '/')
            else:
                shutil.rmtree(folder_new + folder + '/')
                os.mkdir(folder_new + folder + '/')

        """Generate qrels for partial order"""
        for qrels in mult_qrels:
            for metric in ['chebyshev', 'manhattan', 'euclidean']:
                dict_parameters_distance['qrels_file'] = qrels
                dict_parameters_distance['new_qrels_path'] = folder_new + metric+'/'+qrels.split('/')[-1]
                dict_parameters_distance['distance_metric'] = metric
                DistanceOrder(order_parameters=dict_parameters_distance)

            """Generate qrels for partial order"""
            with open(qrels) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]

            rel_file = open(folder_new + 'rel/'+qrels.split('/')[-1], 'w')
            cred_file = open(folder_new + 'cred/'+qrels.split('/')[-1], 'w')
            correctness_file = open(folder_new + 'correctness/'+qrels.split('/')[-1], 'w')

            for line in content:
                parts = line.split()
                print(parts[0], parts[1], parts[2], parts[3], file=rel_file)
                print(parts[0], parts[1], parts[2], parts[4], file=cred_file)
                print(parts[0], parts[1], parts[2], parts[5], file=correctness_file)

            rel_file.close()
            cred_file.close()
            correctness_file.close()
        shutil.rmtree(folder_new+'qrels_removed/')


    def evaluate_downsampling(self,runs_folder, qrels_path_downsamples, metric='ndcg', results_folder = ''):
        runs = self.get_list_files(runs_folder)
        dict_run_results = {}
        qrels_names = {}
        qrels_ids = []
        for run in runs:
            run_name = run.split(os.sep)[-1]
            if run_name not in dict_run_results:
                dict_run_results[run_name] = {}
            list_approaches = os.listdir(qrels_path_downsamples)
            dict_run_results[run_name] = {approach:{} for approach in list_approaches}
            for approach in list_approaches:
                list_qrels = self.get_list_files(qrels_path_downsamples+approach+'/')
                qrels_names = {int(x.split(os.sep)[-1]):x for x in list_qrels}
                qrels_ids = sorted([int(x.split(os.sep)[-1]) for x in list_qrels])
                for qrel_id, qrels_path in sorted(qrels_names.items()):
                    res = self.run_trec_eval(qrels_path, run, metric)
                    if qrel_id not in dict_run_results[run_name][approach]:
                        dict_run_results[run_name][approach][qrel_id] = res['all']
                    # print(run_name,approach, qrel_id, res, dict_run_results[run_name][approach][qrel_id])
                    #     print(qrel_id)

            """ Computing CAM and MM """
            if 'CAM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['CAM'] = {}
            if 'MM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['MM'] = {}
            for qrels_id in sorted(qrels_names):

                #Computing CAM
                if qrels_id not in dict_run_results[run_name]['CAM']:
                    dict_run_results[run_name]['CAM'][qrels_id] =  0.0

                for app in ['rel','use','pop','cred']:
                    dict_run_results[run_name]['CAM'][qrels_id] += (0.25 * dict_run_results[run_name][app][qrels_id])

                #Computing MM
                if qrels_id not in dict_run_results[run_name]['MM']:
                    dict_run_results[run_name]['MM'][qrels_id] =  0.0
                if all(dict_run_results[run_name][app][qrels_id] > 0 for app in ['rel','use','pop','cred']):
                    sum = 0
                    for app in ['rel', 'use', 'pop', 'cred']:
                        sum += (0.25/dict_run_results[run_name][app][qrels_id])
                    dict_run_results[run_name]['MM'][qrels_id] = (1/sum)

        self.results_to_csv(dict_run_results, results_folder, qrels_ids)

        return dict_run_results

    def evaluate_downsampling_decision(self,runs_folder, qrels_path_downsamples, metric='ndcg', results_folder = ''):
        runs = self.get_list_files(runs_folder)
        dict_run_results = {}
        qrels_names = {}
        qrels_ids = []
        for run in runs:
            run_name = run.split(os.sep)[-1]
            if run_name not in dict_run_results:
                dict_run_results[run_name] = {}
            list_approaches = os.listdir(qrels_path_downsamples)
            dict_run_results[run_name] = {approach:{} for approach in list_approaches}
            for approach in list_approaches:
                list_qrels = self.get_list_files(qrels_path_downsamples+approach+'/')
                qrels_names = {int(x.split(os.sep)[-1]):x for x in list_qrels}
                qrels_ids = sorted([int(x.split(os.sep)[-1]) for x in list_qrels])
                for qrel_id, qrels_path in sorted(qrels_names.items()):
                    res = self.run_trec_eval(qrels_path, run, metric)
                    if qrel_id not in dict_run_results[run_name][approach]:
                        dict_run_results[run_name][approach][qrel_id] = res['all']
                    # print(run_name,approach, qrel_id, res, dict_run_results[run_name][approach][qrel_id])
                    #     print(qrel_id)

            """ Computing CAM and MM """
            if 'CAM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['CAM'] = {}
            if 'MM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['MM'] = {}
            for qrels_id in sorted(qrels_names):

                #Computing CAM
                if qrels_id not in dict_run_results[run_name]['CAM']:
                    dict_run_results[run_name]['CAM'][qrels_id] =  0.0

                for app in ['rel','cred','correctness']:
                    dict_run_results[run_name]['CAM'][qrels_id] += ((1/3) * dict_run_results[run_name][app][qrels_id])

                #Computing MM
                if qrels_id not in dict_run_results[run_name]['MM']:
                    dict_run_results[run_name]['MM'][qrels_id] =  0.0
                if all(dict_run_results[run_name][app][qrels_id] > 0 for app in ['rel','cred','correctness']):
                    sum = 0
                    for app in ['rel', 'cred', 'correctness']:
                        sum += ((1/3)/dict_run_results[run_name][app][qrels_id])
                    dict_run_results[run_name]['MM'][qrels_id] = (1/sum)

        self.results_to_csv(dict_run_results, results_folder, qrels_ids)

        return dict_run_results

    def get_set_of_topics(self, runs):
        Q = {}
        with open(runs[0]) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content:
            qid = line.split()[0]
            if qid not in Q:
                Q[int(qid)] = ''
        return [str(q) for q in sorted(list(Q.keys()))]

    def evaluate_downsampling_topicwize(self,runs_folder, qrels_path_downsamples, metric='ndcg', results_folder = ''):
        runs = self.get_list_files(runs_folder)
        set_Q = self.get_set_of_topics(runs)
        print(set_Q)
        dict_run_results = {}
        qrels_names = {}
        qrels_ids = []
        for run in runs:
            run_name = run.split(os.sep)[-1]
            if run_name not in dict_run_results:
                dict_run_results[run_name] = {}
            list_approaches = os.listdir(qrels_path_downsamples)
            dict_run_results[run_name] = {approach:{} for approach in list_approaches}
            for approach in list_approaches:
                list_qrels = self.get_list_files(qrels_path_downsamples+approach+'/')
                qrels_names = {int(x.split(os.sep)[-1]):x for x in list_qrels}
                qrels_ids = sorted([int(x.split(os.sep)[-1]) for x in list_qrels])
                for qrel_id, qrels_path in sorted(qrels_names.items()):
                    res = self.run_trec_eval(qrels_path, run, metric, queries=True)
                    if qrel_id not in dict_run_results[run_name][approach]:
                        dict_run_results[run_name][approach][qrel_id] = []
                    for qid in set_Q:
                        # print(qid)
                        if qid in res:
                            dict_run_results[run_name][approach][qrel_id] += [res[qid]]
                        else:
                            print("else", qid)
                            dict_run_results[run_name][approach][qrel_id] += [0.00]
                    # print(run_name,approach, qrel_id, res, dict_run_results[run_name][approach][qrel_id])
                    #     print(qrel_id)

            """ Computing CAM and MM """
            if 'CAM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['CAM'] = {}
            if 'MM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['MM'] = {}
            for qrels_id in sorted(qrels_names):

                #Computing CAM

                if qrels_id not in dict_run_results[run_name]['CAM']:
                    dict_run_results[run_name]['CAM'][qrels_id] =  []
                for qid in range(len(set_Q)):
                    dict_run_results[run_name]['CAM'][qrels_id] += [np.sum([0.25 * dict_run_results[run_name][app][qrels_id][qid] for app in ['rel','use','pop','cred']])]

                #Computing MM
                if qrels_id not in dict_run_results[run_name]['MM']:
                    dict_run_results[run_name]['MM'][qrels_id] = []
                for qid in range(len(set_Q)):
                    if all(dict_run_results[run_name][app][qrels_id][qid] > 0 for app in ['rel','use','pop','cred']):
                        sum = 0
                        for app in ['rel', 'use', 'pop', 'cred']:
                            sum += (0.25/dict_run_results[run_name][app][qrels_id][qid])
                        dict_run_results[run_name]['MM'][qrels_id]+= [1/sum]
                    else:
                        dict_run_results[run_name]['MM'][qrels_id] += [0.0]

        self.results_to_csv_topicwise(dict_run_results, results_folder, qrels_ids, set_Q)

    def evaluate_downsampling_topicwize_decision(self,runs_folder, qrels_path_downsamples, metric='ndcg', results_folder = ''):
        runs = self.get_list_files(runs_folder)
        set_Q = self.get_set_of_topics(runs)
        print(set_Q)
        dict_run_results = {}
        qrels_names = {}
        qrels_ids = []
        for run in runs:
            run_name = run.split(os.sep)[-1]
            if run_name not in dict_run_results:
                dict_run_results[run_name] = {}
            list_approaches = os.listdir(qrels_path_downsamples)
            dict_run_results[run_name] = {approach:{} for approach in list_approaches}
            for approach in list_approaches:
                list_qrels = self.get_list_files(qrels_path_downsamples+approach+'/')
                qrels_names = {int(x.split(os.sep)[-1]):x for x in list_qrels}
                qrels_ids = sorted([int(x.split(os.sep)[-1]) for x in list_qrels])
                for qrel_id, qrels_path in sorted(qrels_names.items()):
                    res = self.run_trec_eval(qrels_path, run, metric, queries=True)
                    if qrel_id not in dict_run_results[run_name][approach]:
                        dict_run_results[run_name][approach][qrel_id] = []
                    for qid in set_Q:
                        # print(qid)
                        if qid in res:
                            dict_run_results[run_name][approach][qrel_id] += [res[qid]]
                        else:
                            # print("else", qid)
                            dict_run_results[run_name][approach][qrel_id] += [0.00]
                    # print(run_name,approach, qrel_id, res, dict_run_results[run_name][approach][qrel_id])
                    #     print(qrel_id)

            """ Computing CAM and MM """
            if 'CAM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['CAM'] = {}
            if 'MM' not in dict_run_results[run_name]:
                dict_run_results[run_name]['MM'] = {}
            for qrels_id in sorted(qrels_names):

                #Computing CAM

                if qrels_id not in dict_run_results[run_name]['CAM']:
                    dict_run_results[run_name]['CAM'][qrels_id] =  []
                for qid in range(len(set_Q)):
                    dict_run_results[run_name]['CAM'][qrels_id] += [np.sum([(1/3) * dict_run_results[run_name][app][qrels_id][qid] for app in ['rel','cred','correctness']])]

                #Computing MM
                if qrels_id not in dict_run_results[run_name]['MM']:
                    dict_run_results[run_name]['MM'][qrels_id] = []
                for qid in range(len(set_Q)):
                    if all(dict_run_results[run_name][app][qrels_id][qid] > 0 for app in ['rel','cred','correctness']):
                        sum = 0
                        for app in ['rel','cred','correctness']:
                            sum += ((1/3)/dict_run_results[run_name][app][qrels_id][qid])
                        dict_run_results[run_name]['MM'][qrels_id]+= [1/sum]
                    else:
                        dict_run_results[run_name]['MM'][qrels_id] += [0.0]

        self.results_to_csv_topicwise(dict_run_results, results_folder, qrels_ids, set_Q)

    def results_to_csv(self, dict_run_results, results_folder, qrels_ids):
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        else:
            shutil.rmtree(results_folder)
            os.mkdir(results_folder)
        approaches = list(dict_run_results[list(dict_run_results.keys())[0]].keys())
        for run in dict_run_results:
            run_result_csv = open(results_folder+run, 'w')
            print('qid', ','.join(approaches), sep=',', file=run_result_csv)
            for qrel_id in qrels_ids:
                print(qrel_id, ','.join([str(round(dict_run_results[run][approach][qrel_id], 4)) for approach in approaches]), sep=',', file=run_result_csv)

    def results_to_csv_topicwise(self, dict_run_results, results_folder, qrels_ids, set_Q):
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        else:
            shutil.rmtree(results_folder)
            os.mkdir(results_folder)
        approaches = list(dict_run_results[list(dict_run_results.keys())[0]].keys())
        for qrel_id in qrels_ids:
            os.mkdir(results_folder+str(qrel_id))

            for run in dict_run_results:
                run_result_csv = open(results_folder+str(qrel_id)+'/'+run, 'w')
                print('qid', ','.join(approaches), sep=',', file=run_result_csv)
                for qid in range(len(set_Q)):
                    print(qid+1, ','.join([str(round(dict_run_results[run][approach][qrel_id][qid], 4)) for approach in approaches]), sep=',', file=run_result_csv)
                run_result_csv.close()

    def plot_results_by_run(self, results_folder):
        plt.figure()
        for i, file in enumerate(self.get_list_files(results_folder)):
            df = pd.read_csv(file)
            # for data_column in list(df.keys()[1:]):
            plt.subplot(2,3,i+1)
            for data_column in ['euclidean','manhattan','CAM','MM']:
                plt.plot('qid', data_column, data=df)
            plt.legend(loc=1, prop={'size': 5})
            plt.title(file.split('/')[-1])
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)
        plt.show()

    def plot_results_summary(self, results_folder):
        num_file = len(self.get_list_files(results_folder))
        df_t = pd.read_csv(self.get_list_files(results_folder)[0])
        num_vals = len(df_t['qid'].values)
        dict_vals = {'euclidean':[0]*(num_vals), 'manhattan':[0]*(num_vals), 'CAM':[0]*(num_vals), 'MM':[0]*(num_vals)}
        print(dict_vals)

        for i, file in enumerate(self.get_list_files(results_folder)):
            df = pd.read_csv(file)
            for data_column in ['euclidean', 'manhattan', 'CAM', 'MM']:
                dict_vals[data_column] = np.add(np.array(df[data_column].values)/num_file, dict_vals[data_column])
            # print(df['euclidean'].values)
            # [math.df['euclidean'][0] - val for val in df['euclidean'].values[1:]]:

            print(df['euclidean'].values[1:])
        for data_column in ['euclidean', 'manhattan', 'CAM', 'MM']:
            plt.plot(df['qid'], dict_vals[data_column], label=data_column)

        plt.xlabel('Percentage removed from assessements')
        plt.ylabel('Average measure score')
        plt.legend()
        plt.show()


    def read_results_from_topicwise_folder(self, results_folder, qrels_ids):
        dict_results = {}
        for qrels_id in qrels_ids:
            if qrels_id not in dict_results:
                dict_results[qrels_id] = {}
            results_files = self.get_list_files(results_folder+qrels_id+'/')
            for model in results_files:
                model_id = model.split('/')[-1]
                with open(model) as f:
                    content = f.readlines()
                content = [x.rstrip() for x in content]
                metrics = content[0].split(',')[1:]
                for line in content[1:]:
                    qid = int(line.split(',')[0])
                    if qid not in dict_results[qrels_id]:
                        dict_results[qrels_id][qid] = {}
                    for metric, score in zip(metrics, line.split(',')[1:]):
                        if metric not in dict_results[qrels_id][qid]:
                            dict_results[qrels_id][qid][metric] = [{model_id:score}]
                        else:
                            dict_results[qrels_id][qid][metric] += [{model_id:score}]
        return dict_results


    def calc_corr_by_topic(self, results_folder = ''):
        self.dict_of_models_order_by_query = {}
        qrels_ids = os.listdir(results_folder)
        dict_results = self.read_results_from_topicwise_folder(results_folder, qrels_ids)
        set_Q = list(dict_results['0'].keys())
        model_ids = [ model.split('/')[-1] for model in self.get_list_files(results_folder+qrels_ids[0]+'/')]

        # print(dict_results['0'][1]['manhattan'])
        # print(dict_results['15'][1]['manhattan'])
        # exit()
        # #
        for metric in ['manhattan','euclidean','chebyshev','MM','CAM',]:
            if metric not in self.dict_of_models_order_by_query:
                self.dict_of_models_order_by_query[metric] = {}
            for qrel_id in qrels_ids:
                if qrel_id not in self.dict_of_models_order_by_query[metric]:
                    self.dict_of_models_order_by_query[metric][qrel_id] = {}
                for query in set_Q:
                    if query not in self.dict_of_models_order_by_query[metric][qrel_id]:
                        self.dict_of_models_order_by_query[metric][qrel_id][query] = []
                    models_values = dict((key,d[key]) for d in dict_results[qrel_id][query][metric] for key in d)
                    order = sorted(models_values, key=models_values.get)
                    self.dict_of_models_order_by_query[metric][qrel_id][query] = order
        # print(self.dict_of_models_order_by_query[metric][qrel_id][query])

        self.dict_correlations_by_topic = {}
        for metric in ['manhattan', 'euclidean','chebyshev', 'MM', 'CAM']:
            if metric not in self.dict_correlations_by_topic :
                self.dict_correlations_by_topic[metric] = {}
                for qrel_id in qrels_ids:
                    if qrel_id not in self.dict_correlations_by_topic[metric]:
                        self.dict_correlations_by_topic[metric][qrel_id] = 0.0
                    correlations = []
                    for query in set_Q:
                        tau, p_value = stats.kendalltau(self.dict_of_models_order_by_query[metric]['0'][query],
                                                        self.dict_of_models_order_by_query[metric][qrel_id][query])
                        correlations += [tau]
                    self.dict_correlations_by_topic[metric][qrel_id] = np.mean(correlations)

        # pprint.pprint(self.dict_correlations_by_topic)
        return self.dict_correlations_by_topic
        # self.plot_correlation(self.dict_correlations_by_topic)

    def calc_rmse_by_topic(self, results_folder = ''):
        self.dict_of_models_order_by_query = {}
        qrels_ids = os.listdir(results_folder)
        dict_results = self.read_results_from_topicwise_folder(results_folder, qrels_ids)
        set_Q = list(dict_results['0'].keys())
        model_ids = [model.split('/')[-1] for model in self.get_list_files(results_folder + qrels_ids[0] + '/')]

        # print(dict_results['0'][1]['manhattan'])
        # print(dict_results['15'][1]['manhattan'])
        # exit()
        # #
        for metric in ['manhattan', 'euclidean','chebyshev', 'MM', 'CAM', ]:
            if metric not in self.dict_of_models_order_by_query:
                self.dict_of_models_order_by_query[metric] = {}
            for qrel_id in qrels_ids:
                if qrel_id not in self.dict_of_models_order_by_query[metric]:
                    self.dict_of_models_order_by_query[metric][qrel_id] = {}
                for query in set_Q:
                    if query not in self.dict_of_models_order_by_query[metric][qrel_id]:
                        self.dict_of_models_order_by_query[metric][qrel_id][query] = {}
                    models_values = dict((key, d[key]) for d in dict_results[qrel_id][query][metric] for key in d)
                    self.dict_of_models_order_by_query[metric][qrel_id][query] = models_values
        # print(self.dict_of_models_order_by_query[metric][qrel_id][query])

        self.dict_rmse_by_model = {}
        for metric in ['manhattan', 'euclidean','chebyshev', 'MM', 'CAM']:
            if metric not in self.dict_rmse_by_model:
                self.dict_rmse_by_model[metric] = {}
                for qrel_id in qrels_ids:
                    if qrel_id not in self.dict_rmse_by_model[metric]:
                        self.dict_rmse_by_model[metric][qrel_id] = {}
                    SRE = []
                    for model in model_ids:
                        if model not in self.dict_rmse_by_model[metric][qrel_id]:
                            self.dict_rmse_by_model[metric][qrel_id][model] = 0.0
                        for query in set_Q:
                            SRE += [math.pow((float(self.dict_of_models_order_by_query[metric]['0'][query][model]) - float(self.dict_of_models_order_by_query[metric][qrel_id][query][model])),2)]
                            # print(metric, qrel_id, model, query,self.dict_of_models_order_by_query[metric]['0'][query][model], self.dict_of_models_order_by_query[metric][qrel_id][query][model])
                        self.dict_rmse_by_model[metric][qrel_id][model] = math.sqrt(np.mean(SRE))

        dict_summary_results = {}
        for metric in ['manhattan', 'euclidean', 'chebyshev','MM', 'CAM']:
            if metric not in dict_summary_results:
                dict_summary_results[metric] = {}
            for qrel_id in qrels_ids:
                if qrel_id not in dict_summary_results[metric]:
                    dict_summary_results[metric][qrel_id] = 0.0
                arr = [rmse for run,rmse in self.dict_rmse_by_model[metric][qrel_id].items()]
                dict_summary_results[metric][qrel_id] = np.mean(arr)

        # pprint.pprint(dict_summary_results)
        return dict_summary_results
        # if you want to plot for a single case, remove the return above and uncomment the line below
        # self.plot_RMSE(dict_summary_results)

    def plot_correlation(self, correlations):
        i=1
        matplotlib.rcParams.update({'font.size': 12})
        plt.rc('axes', labelsize=16)
        for metric, values in correlations.items():
            plt.plot(values.keys(), values.values(),label=metric, marker = i)
            i += 1
        plt.xlabel('% docs removed from the pool')
        plt.ylabel('Kendall’s τ')
        plt.legend()
        plt.savefig('data/dataset_statistics/correlation_downsampling_decision.jpeg')
        plt.clf()

    def plot_RMSE(self, correlations):
        i = 1
        matplotlib.rcParams.update({'font.size': 12})
        plt.rc('axes', labelsize=16)
        plt.ylim(0, 0.25)
        for metric, values in correlations.items():
            plt.plot(values.keys(), values.values(), label=metric, marker=i)
            i += 1


        plt.xlabel('% docs removed from the pool.')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig('data/dataset_statistics/rmse_downsampling_2015.jpeg')
        plt.clf()

    # def plot_mean_and_CI(self, mean, lb, ub, color_mean=None, color_shading=None):
    def plot_mean_and_CI(self, dict_to_plot, plot_path, x_labels, corr=None):
        # plot the shaded range of the confidence intervals
        colors = ['b','g','r','c','m','y','k']
        matplotlib.rcParams.update({'font.size': 12})
        plt.rc('axes', labelsize=16)
        if corr is None:
            plt.ylim(0, 0.25)
        for i, key in enumerate(dict_to_plot):
            plt.fill_between(range(len(dict_to_plot[key]['mean'])), dict_to_plot[key]['ub'], dict_to_plot[key]['lb'],
                             color=colors[i], alpha=.5)
            # plot the mean on top
            plt.plot(x_labels, dict_to_plot[key]['mean'], colors[i], label=key, marker=i)
        plt.xlabel('% docs removed from the pool.')
        if corr is not None:
            plt.ylabel('Kendall’s τ')
        else:
            plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def generate_downsampling(self, qrels_path, folder_new, step=1, num_aspects=4):
        if not os.path.exists(folder_new):
            os.mkdir(folder_new)
        else:
            shutil.rmtree(folder_new)
            os.mkdir(folder_new)
        self.pre_process(qrels_path, step=step, folder_new=folder_new)
        if num_aspects != 3:
            self.compute_partial_order(folder_new, num_aspects)
        else:
            self.compute_partial_order_decision(folder_new, num_aspects)
