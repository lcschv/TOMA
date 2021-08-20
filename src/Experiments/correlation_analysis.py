import pprint
import os
import pandas as pd
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import permutations
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from pathlib import Path
from collections import Counter

class CorrAnalysis(object):
    def __init__(self, results_folder=None, file_out=None, trec='task'):
        self.self = self

        self.dict_results = {}
        self.dict_results_average = {}
        self.trec = trec

        if results_folder is not None:
            self.results_folder = results_folder
        else:
            self.results_folder = 'data/results/WebTrack2013/'

        if file_out is None:
            self.file_out = 'correlation_analysis_notspecified.txt'
        else:
            self.file_out = file_out


        files = self.get_list_files(self.results_folder)
        self.models = [model.split(os.sep)[-1].replace('.csv','') for model in files]
        # print(self.models)
        # exit()
        self.path_models = files
        # print("Models:", self.path_models)
        self.run()

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def read_run(self, run):
        with open(run) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]


        self.metrics = content[0].split(',')[1:]
        # print()
        # print("Metrics:", self.metrics)
        # exit()
        dict_run = {}
        for line in content[1:]:
            parts = line.split(',')
            qid = parts[0]
            if qid not in dict_run:
                dict_run[qid] = {metric:score for metric,score in zip(self.metrics,parts[1:])}
        return dict_run

    def get_average_results(self, run):
        data = pd.read_csv(run, encoding="iso-8859-1", sep=",")
        dict_average = {}
        df = pd.DataFrame(data=data)
        dict_average = {colum:df[colum].mean() for colum in df.columns[1:]}
        return dict_average

    def preprocess_data(self):
        for model, path in zip(self.models, self.path_models):
            if model not in self.dict_results:
                self.dict_results[model] = {}
                self.dict_results_average[model] = {}

                self.dict_results[model].update(self.read_run(path))
                self.dict_results_average[model].update(self.get_average_results(path))

                self.queries = list(self.dict_results[model].keys())

    def correlation_average_evaluation_score(self):
        self.dict_of_models_order = {}
        for metric in self.metrics:

            for app in list(self.dict_results_average.keys()):
                if metric not in self.dict_of_models_order:
                    self.dict_of_models_order[metric] = {}
                if app not in self.dict_of_models_order[metric]:
                    self.dict_of_models_order[metric][app] = {}
                self.dict_of_models_order[metric][app] = self.dict_results_average[app][metric]
        # pprint.pprint(self.dict_results_average)
        # print("Dict_of_models")
        # pprint.pprint(self.dict_of_models_order)
        # exit()
        self.dict_correlations_average = {}
        # for approach in self.approaches:
        #     if approach not in self.self.dict_correlations_average:
        #         self.self.dict_correlations_average[approach] = {}
        # print(list(permutations(self.metrics,2)))
        for comb in permutations(self.metrics,2):
            if comb[0] not in self.self.dict_correlations_average:
                self.self.dict_correlations_average[comb[0]] = {comb[1]:0}
            if comb[0] not in self.self.dict_correlations_average[comb[0]]:
                self.self.dict_correlations_average[comb[0]] = {comb[0]:1.0}

            a = sorted(self.dict_of_models_order[comb[0]], key=self.dict_of_models_order[comb[0]].get)
            # print(self.dict_of_models_order[comb[0]])
            b = sorted(self.dict_of_models_order[comb[1]], key=self.dict_of_models_order[comb[1]].get)
            tau, p_value = stats.kendalltau(a, b)
            self.dict_correlations_average[comb[0]][comb[1]] = tau
        # print(self.dict_correlations_average.keys())
        # pprint.pprint(self.dict_correlations_average)
        ### print results in latex format ###
        # exit()
        # uncomment this if you want to print the results
        self.print_average_corr(self.dict_correlations_average)
        # self.print_average_corr_new_template(self.dict_correlations_average)

    def print_average_corr(self, dict_correlations_average , file_out=None):

        if self.file_out is not None:
            file_out = open(self.file_out, 'w')
        else:
            file_out = open(r"data/dataset_statistics/average_corr_analysis.txt", 'w')
        #
        print("\\begin{table}[]", file=file_out)
        print("\small", file=file_out)
        #
        print("\\begin{table}[]")
        print("\small")
        #
        print(
            "\caption{Correlation between evaluation measures average scores over 221 queries for 3 different retrieval models: BM25, LM\_dir, LM\_jel.}",
            file=file_out)
        print("\label{tab:corr_analysis}", file=file_out)
        print("\\begin{tabular}"+"{"+"l"*(len(self.metrics)+1)+"}", file=file_out)
        print("\\hline", file=file_out)
        print(" &"+ '&'.join(["\\textbf{"+met+"}" for met in self.metrics]), "\\\\ \\hline", file=file_out)
        # print("\\hline", file=file_out)
        #
        #
        # #### JUST PRINTING ####
        print(
            "\caption{Correlation between evaluation measures average scores over 221 queries for 3 different retrieval models: BM25, LM\_dir, LM\_jel.}")
        print("\label{tab:corr_analysis}")
        print("\\begin{tabular}" + "{" + "l" * (len(self.metrics) + 1) + "}", "\\\\ \\toprule")
        print(" &" + '&'.join(["\\textbf{" + met + "}" for met in self.metrics]))


        for app in dict_correlations_average:
            # print(app)
            # print("\multicolumn{1}{l}{\\textbf{"+app.replace('_','\_').replace('.txt','')+"}} &  &  &  &  & \\\ \hline")
            # print("\multicolumn{1}{l}{\\textbf{"+app.replace('_','\_').replace('.txt','')+r"}} &  &  &  &  & \\", file=file_out)
        #     # for metric in self.metrics:
            print(r'\textbf{',app, "}&",' & '.join([str(round(dict_correlations_average[app][x],2)) for x in self.metrics]),"\\\\")
            print(r'\textbf{',app, "}&", ' & '.join([str(round(dict_correlations_average[app][x],2)) for x in self.metrics]),"\\\\", file=file_out)
        print(r'\bottomrule')
        print(r'\bottomrule', file=file_out)
            # print('\hline', file=file_out)
        #

        print("\end{tabular}", file=file_out)
        print("\end{table}", file=file_out)
        print("\end{tabular}")
        print("\end{table}")

    def print_average_corr_new_template(self, dict_correlations_average, file_out=None):
        # print("YEAH")
        # pprint.pprint(dict_correlations_average)
        # exit()
        print('\\textbf{TOMA cheb}&',round(dict_correlations_average['chebyshev']['euclidean'],2), '&', round(dict_correlations_average['chebyshev']['manhattan'],2),'&',
              round(dict_correlations_average['chebyshev']['CAM'],2),'&',round(dict_correlations_average['chebyshev']['MM'],2), '\\\\')

        print('\\textbf{TOMA eucl}&', '\cellcolor{gray}', '&',
              round(dict_correlations_average['euclidean']['manhattan'],2), '&',
              round(dict_correlations_average['euclidean']['CAM'],2), '&', round(dict_correlations_average['euclidean']['MM'],2), '\\\\')

        print('\\textbf{TOMA manh}&', '\cellcolor{gray}', '&',
              '\cellcolor{gray}', '&',
              round(dict_correlations_average['manhattan']['CAM'],2), '&', round(dict_correlations_average['manhattan']['MM'],2), '\\\\')

        print('\\textbf{CAM}&', '\cellcolor{gray}', '&',
              '\cellcolor{gray}', '&',
              '\cellcolor{gray}', '&', round(dict_correlations_average['CAM']['MM'],2), '\\\\')

        print("SINGLE ASPECTS!!!")

        if self.trec == 'task':
            print('\\textbf{TOMA cheb}&',round(dict_correlations_average['chebyshev']['rel'],2), '&', round(dict_correlations_average['chebyshev']['use'],2),'&',
                  round(dict_correlations_average['chebyshev']['pop'],2),'&',round(dict_correlations_average['chebyshev']['cred'],2), '\\\\')

            print('\\textbf{TOMA eucl}&',round(dict_correlations_average['euclidean']['rel'],2), '&', round(dict_correlations_average['euclidean']['use'],2),'&',
                  round(dict_correlations_average['euclidean']['pop'],2),'&',round(dict_correlations_average['euclidean']['cred'],2), '\\\\')

            print('\\textbf{TOMA manh}&',round(dict_correlations_average['manhattan']['rel'],2), '&', round(dict_correlations_average['manhattan']['use'],2),'&',
                  round(dict_correlations_average['manhattan']['pop'],2),'&',round(dict_correlations_average['manhattan']['cred'],2), '\\\\')

            print('\\textbf{CAM}&',round(dict_correlations_average['CAM']['rel'],2), '&', round(dict_correlations_average['CAM']['use'],2),'&',
                  round(dict_correlations_average['CAM']['pop'],2),'&',round(dict_correlations_average['CAM']['cred'],2), '\\\\')

            print('\\textbf{MM}&', round(dict_correlations_average['MM']['rel'], 2), '&',
                  round(dict_correlations_average['MM']['use'], 2), '&',
                  round(dict_correlations_average['MM']['pop'], 2), '&',
                  round(dict_correlations_average['MM']['cred'], 2), '\\\\')

            print('\midrule')
            print('\\textbf{Use}&', round(dict_correlations_average['use']['rel'],2), "& \cellcolor{gray} & \cellcolor{gray} & \cellcolor{gray}\\\\" )
            print('\\textbf{Pop}&', round(dict_correlations_average['pop']['rel'],2),'&' ,round(dict_correlations_average['pop']['use'],2), "& \cellcolor{gray} & \cellcolor{gray}\\\\" )
            print('\\textbf{Spa}&', round(dict_correlations_average['cred']['rel'],2),'&' ,round(dict_correlations_average['cred']['use'],2),'&', round(dict_correlations_average['cred']['pop'],2),"& \cellcolor{gray}\\\\" )

        elif self.trec == 'decision':
            print('\\textbf{TOMA cheb}&', round(dict_correlations_average['chebyshev']['rel'], 2), '&',
                  round(dict_correlations_average['chebyshev']['cred'], 2), '&',
                  round(dict_correlations_average['chebyshev']['correc'], 2), '\\\\')

            print('\\textbf{TOMA eucl}&', round(dict_correlations_average['euclidean']['rel'], 2), '&',
                  round(dict_correlations_average['euclidean']['cred'], 2), '&',
                  round(dict_correlations_average['euclidean']['correc'], 2), '\\\\')

            print('\\textbf{TOMA manh}&', round(dict_correlations_average['manhattan']['rel'], 2), '&',
                  round(dict_correlations_average['manhattan']['cred'], 2), '&',
                  round(dict_correlations_average['manhattan']['correc'], 2), '\\\\')

            print('\\textbf{CAM}&', round(dict_correlations_average['CAM']['rel'], 2), '&',
                  round(dict_correlations_average['CAM']['cred'], 2), '&',
                  round(dict_correlations_average['CAM']['correc'], 2), '\\\\')

            print('\\textbf{MM}&', round(dict_correlations_average['MM']['rel'], 2), '&',
                  round(dict_correlations_average['MM']['cred'], 2), '&',
                  round(dict_correlations_average['MM']['correc'], 2), '\\\\')

            print('\midrule')
            print('\\textbf{Cred}&', round(dict_correlations_average['cred']['rel'],2),
                  "& \cellcolor{gray} & \cellcolor{gray}\\\\")
            print('\\textbf{Corr}&', round(dict_correlations_average['correc']['rel'],2), '&',
                  round(dict_correlations_average['correc']['cred'],2), "& \cellcolor{gray}\\\\")

        elif self.trec == 'web':
            print('\\textbf{TOMA cheb}&', round(dict_correlations_average['chebyshev']['rel'], 2), '&',
                  round(dict_correlations_average['chebyshev']['cred'], 2), '&',
                  round(dict_correlations_average['chebyshev']['correc'], 2), '\\\\')

            print('\\textbf{TOMA eucl}&', round(dict_correlations_average['euclidean']['rel'], 2), '&',
                  round(dict_correlations_average['euclidean']['cred'], 2), '&',
                  round(dict_correlations_average['euclidean']['correc'], 2), '\\\\')

            print('\\textbf{TOMA manh}&', round(dict_correlations_average['manhattan']['rel'], 2), '&',
                  round(dict_correlations_average['manhattan']['cred'], 2), '&',
                  round(dict_correlations_average['manhattan']['correc'], 2), '\\\\')

            print('\\textbf{CAM}&', round(dict_correlations_average['CAM']['rel'], 2), '&',
                  round(dict_correlations_average['CAM']['cred'], 2), '&',
                  round(dict_correlations_average['CAM']['correc'], 2), '\\\\')

            print('\\textbf{MM}&', round(dict_correlations_average['MM']['rel'], 2), '&',
                  round(dict_correlations_average['MM']['cred'], 2), '&',
                  round(dict_correlations_average['MM']['correc'], 2), '\\\\')

            print('\midrule')
            print('\\textbf{Pop}&', round(dict_correlations_average['cred']['rel'],2),
                  "& \cellcolor{gray} & \cellcolor{gray}\\\\")
            print('\\textbf{Spa}&', round(dict_correlations_average['correc']['rel'],2), '&',
                  round(dict_correlations_average['correc']['cred'],2), "& \cellcolor{gray}\\\\")

    def print_average_corr_weighting(self, dict_correlations_average , file_out=None):

        if self.file_out is not None:
            file_out = open(self.file_out, 'w')
        else:
            file_out = open(r"data/dataset_statistics/average_corr_analysis.txt", 'w')
        #
        # print("\\begin{table}[]", file=file_out)
        # print("\small", file=file_out)
        # #
        # print("\\begin{table}[]")
        # print("\small")
        # #
        # print(
        #     "\caption{Correlation between evaluation measures average scores over 221 queries for 3 different retrieval models: BM25, LM\_dir, LM\_jel.}",
        #     file=file_out)
        # print("\label{tab:corr_analysis}", file=file_out)
        # print("\\begin{tabular}"+"{"+"l"*(len(self.metrics)+1)+"}", file=file_out)
        # print("\\hline", file=file_out)
        # print(" &"+ '&'.join(["\\textbf{"+met+"}" for met in self.metrics]), "\\\\ \\hline", file=file_out)
        # # print("\\hline", file=file_out)
        # #
        # #
        # # #### JUST PRINTING ####
        # print(
        #     "\caption{Correlation between evaluation measures average scores over 221 queries for 3 different retrieval models: BM25, LM\_dir, LM\_jel.}")
        # print("\label{tab:corr_analysis}")
        # print("\\begin{tabular}" + "{" + "l" * (len(self.metrics) + 1) + "}", "\\\\ \\toprule")
        print("metrics", '\t'.join([met.replace('double_aspect-','D').replace('equispaced_samerange_aspect-','E').replace('_','') for met in self.metrics]),sep='\t')


        for app in dict_correlations_average:
            # print(app)
            # print("\multicolumn{1}{l}{\\textbf{"+app.replace('_','\_').replace('.txt','')+"}} &  &  &  &  & \\\ \hline")
            # print("\multicolumn{1}{l}{\\textbf{"+app.replace('_','\_').replace('.txt','')+r"}} &  &  &  &  & \\", file=file_out)
        #     # for metric in self.metrics:
            print(app.replace('double_aspect-','D').replace('equispaced_samerange_aspect-','E').replace('_',''), "\t".join([str(round(dict_correlations_average[app][x],2)) for x in self.metrics]), sep='\t')
            # print(app, '\t'.join([str(round(dict_correlations_average[app][x],2)) for x in self.metrics]), file=file_out)
        # print(r'\bottomrule')
        # print(r'\bottomrule', file=file_out)
            # print('\hline', file=file_out)
        #

        # print("\end{tabular}", file=file_out)
        # print("\end{table}", file=file_out)
        # print("\end{tabular}")
        # print("\end{table}")

    def print_table_sigir_misinfo(self, dict_correlations_average, file_out=None):
        # print("YEAH")
        # pprint.pprint(dict_correlations_average)
        # exit()
        print('EUCL - CAM', round(dict_correlations_average['euclidean']['CAM'],2))
        print('EUCL - MM', round(dict_correlations_average['euclidean']['MM'],2))
        print('MANH - CAM', round(dict_correlations_average['manhattan']['CAM'],2))
        print('MANH - MM', round(dict_correlations_average['manhattan']['MM'],2))
        print('CHEB - CAM', round(dict_correlations_average['chebyshev']['CAM'],2))
        print('CHEB - MM', round(dict_correlations_average['chebyshev']['MM'],2))
        print('EUCL - MANH', round(dict_correlations_average['euclidean']['manhattan'],2))
        print('EUCL - CHEB', round(dict_correlations_average['euclidean']['chebyshev'],2))
        print('MANH - CHEB', round(dict_correlations_average['manhattan']['chebyshev'],2))
        print('CAM - MM', round(dict_correlations_average['CAM']['MM'],2))

        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('Credibility - Correctness', round(dict_correlations_average['cred']['correc'],2) )
        print('Relevance - Credibility', round(dict_correlations_average['rel']['cred'],2))
        print('Relevance - Correctness', round(dict_correlations_average['rel']['correc'],2))

       

    def calc_corr_by_topic(self, file_out=None):
        self.dict_of_models_order_by_query = {}

        # pprint.pprint(self.dict_results)
        # exit()
        for metric in self.metrics:
            if metric not in self.dict_of_models_order_by_query:
                self.dict_of_models_order_by_query[metric] = {}
            for query in self.queries:

                models_values = {model: self.dict_results[model][query][metric] for model in self.models}
                order = sorted(models_values, key=models_values.get)
                # exit()
                # print(models_values)
                # print(order)

                if query not in self.dict_of_models_order_by_query[metric]:
                    self.dict_of_models_order_by_query[metric][query] = []
                self.dict_of_models_order_by_query[metric][query] = order
        self.dict_correlations_by_topic = {}
        for comb in permutations(self.metrics, 2):
            if comb[0] not in self.self.dict_correlations_by_topic:
                self.dict_correlations_by_topic[comb[0]] = {}
            if comb[1] not in self.self.dict_correlations_by_topic[comb[0]]:
                self.dict_correlations_by_topic[comb[0]][comb[1]] = {}
            if comb[0] not in self.self.dict_correlations_by_topic[comb[0]]:
                self.dict_correlations_by_topic[comb[0]][comb[0]] = 1

            correlations = []
            for query in self.queries:
                tau, p_value = stats.kendalltau(self.dict_of_models_order_by_query[comb[0]][query], self.dict_of_models_order_by_query[comb[1]][query])
                correlations += [tau]

            ##### OBS. IF you want to get the correlation topicwise, just need to add the correlations of each topic to the dictionary ####
            self.dict_correlations_by_topic[comb[0]][comb[1]] = np.mean(correlations)

        #uncomment this if you want to print the results
        self.print_average_corr_weighting(self.dict_correlations_by_topic)
        # self.print_average_corr_new_template(self.dict_correlations_by_topic)
        # self.print_table_sigir_misinfo(self.dict_correlations_by_topic)
        # self.print_average_corr(self.dict_correlations_by_topic, file_out=r"data/dataset_statistics/corr_by_topic2014.txt")

    def calc_corr_by_topic_setsize(self, set_size ,file_out=None):
        self.dict_of_models_order_by_query = {}

        for approach in self.approaches:
            if approach not in self.dict_of_models_order_by_query:
                self.dict_of_models_order_by_query[approach] = {}
            for metric in self.metrics:
                if metric not in self.dict_of_models_order_by_query[approach]:
                    self.dict_of_models_order_by_query[approach][metric] = {}
                for query in self.queries:
                    models_values = {model: self.dict_results[model][approach][query][metric] for model in self.models}
                    # order = sorted(models_values, key=models_values.get)

                    if query not in self.dict_of_models_order_by_query[approach][metric]:
                        self.dict_of_models_order_by_query[approach][metric][query] = {}
                    self.dict_of_models_order_by_query[approach][metric][query] = models_values
        # range_size = [i for i in range(1,226)]
        range_size = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50, 100, 200]
        self.dict_correlations_by_topic = {}
        for set_size in range_size:
            print(set_size)
            if set_size not in self.dict_correlations_by_topic:
                self.dict_correlations_by_topic[set_size] = {}
            for approach in self.approaches:
                if approach not in self.dict_correlations_by_topic[set_size]:
                    self.dict_correlations_by_topic[set_size][approach] = {}
                for comb in permutations(self.metrics, 2):
                    if comb[0] not in self.self.dict_correlations_by_topic[set_size][approach]:
                        self.dict_correlations_by_topic[set_size][approach][comb[0]] = {}
                    if comb[1] not in self.self.dict_correlations_by_topic[set_size][approach][comb[0]]:
                        self.dict_correlations_by_topic[set_size][approach][comb[0]][comb[1]] = {}
                    if comb[0] not in self.self.dict_correlations_by_topic[set_size][approach][comb[0]]:
                        self.dict_correlations_by_topic[set_size][approach][comb[0]][comb[0]] = 1

                    correlations = []

                    for i in range(200):
                        tmp_set = set(random.sample(self.queries, set_size))
                        # print(tmp_set)
                        temp_dict_one = {comb[0]: {model:0.0 for model in self.models}}
                        temp_dict_two = {comb[1]: {model:0.0 for model in self.models}}
                        for query in tmp_set:
                            for model in  self.models:
                                temp_dict_one[comb[0]][model] += float(self.dict_of_models_order_by_query[approach][comb[0]][query][model])
                                temp_dict_two[comb[1]][model] += float(self.dict_of_models_order_by_query[approach][comb[1]][query][model])
                                # print(temp_dict_one, temp_dict_two)
                        order = sorted(temp_dict_one[comb[0]], key=temp_dict_one[comb[0]].get)
                        order_two = sorted(temp_dict_two[comb[1]], key=temp_dict_two[comb[1]].get)
                        # print(order, order_two)
                        tau, p_value = stats.kendalltau(order, order_two)
                        correlations += [tau]

                ##### OBS. IF you want to get the correlation topicwise, just need to add the correlations of each topic to the dictionary ####
                    self.dict_correlations_by_topic[set_size][approach][comb[0]][comb[1]] = np.mean(correlations)

        self.plot_corr_by_topic_setsize(range_size)

    def plot_corr_by_topic_setsize(self, range_size):


        import matplotlib.pyplot as plt
        import numpy as np
        # fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(6, 6))
        #
        # fig.text(25, 0.04, 'common X', ha='center')
        # fig.text(0.05, 0.5, 'common Y', va='center', rotation='vertical')

        for i, approach in enumerate(self.approaches,1):
            plt.subplot(3, 2, i)
            # plt.legend([""])
            print(self.metrics)
            # arr = self.metrics.remove('Kendals variation')
            # print(arr)
            for comb in combinations(self.metrics,2):
                # plt.ylim(-1, 1)
                plt.plot(range_size[:-3], [self.dict_correlations_by_topic[set_size][approach][comb[0]][comb[1]] for set_size in range_size[:-3]], label=comb)
            plt.legend(loc=4, prop={'size': 4})
        # plt.xlabel("Hue")

        plt.savefig('foo.png')
        plt.show()

        self.dict_correlations_by_topic
        pass

    def plot_corr_vals_by_topic(self):
        low_corr = {}
        dict_corrs_freq = {}
        group_metrics = ['euclidean','skyline','CAM','MM']
        group_metrics = ['skyline','CAM']
        dict_corr_values = {}
        pairs_ = list(combinations(group_metrics, 2))
        pairs_ = [('skyline','CAM'), ('skyline','MM'), ('skyline','euclidean'), ('euclidean','CAM'), ('euclidean','MM')]
        for comb in pairs_:
            for query in self.queries:
                tau, p_value = stats.kendalltau(self.dict_of_models_order_by_query[comb[0]][query],
                                                self.dict_of_models_order_by_query[comb[1]][query])
                if tau not in dict_corr_values:
                    dict_corr_values[tau] = ''
                # if tau <= 0.0:
                if comb[0] not in low_corr:
                    low_corr[comb[0]] = {}
                    dict_corrs_freq[comb[0]] = {}
                if comb[1] not in low_corr[comb[0]]:
                    low_corr[comb[0]][comb[1]] = {}
                    dict_corrs_freq[comb[0]][comb[1]] = {'x':[],'y':[]}
                if query not in low_corr[comb[0]][comb[1]]:
                    low_corr[comb[0]][comb[1]][query] = tau
        # pprint.pprint(low_corr)
        # exit()
        for comb in pairs_:
            dict_freq = dict(Counter(low_corr[comb[0]][comb[1]].values()))
            for corr_val in sorted(dict_corr_values):
                if corr_val in dict_freq:
                    dict_corrs_freq[comb[0]][comb[1]]['x'] += [corr_val]
                    dict_corrs_freq[comb[0]][comb[1]]['y'] += [dict_freq[corr_val]/len(low_corr[comb[0]][comb[1]])]
                else:
                    dict_corrs_freq[comb[0]][comb[1]]['x'] += [corr_val]
                    dict_corrs_freq[comb[0]][comb[1]]['y'] += [0]

            # print(comb[0], comb[1], Counter(low_corr[comb[0]][comb[1]].values()))
        # pprint.pprint(dict_corrs_freq)
        line_styles = ['.:', '.-.', '*-', '+--', 'x:', 'o-.']
        markers = ['o', 'x', '*', '>', '<', '1','2','3']
        for i, comb in enumerate(pairs_):
            cum_sum = np.cumsum(dict_corrs_freq[comb[0]][comb[1]]['y'])
            plt.plot(dict_corrs_freq[comb[0]][comb[1]]['x'], cum_sum, label=str(comb), marker=markers[i])
            # plt.plot(dict_corrs_freq[comb[0]][comb[1]]['x'], dict_corrs_freq[comb[0]][comb[1]]['y'], label=str(comb), marker=markers[i])
        plt.legend()  # To draw legend
        plt.xlabel("Kendall's correlation value")
        plt.ylabel('Accumulated topic frequency')
        # plt.show()
        plt.savefig('data/dataset_statistics/accumulated_corr_2016.png')
        plt.clf()

        # print(low_corr['skyline']['euclidean'])

    def takeSecond(self, elem):
        return elem[1]

    def get_least_correlated_topics(self):
        low_corr = {}
        dict_corrs_freq = {}
        group_metrics = ['euclidean', 'skyline', 'CAM', 'MM']
        group_metrics = ['skyline', 'CAM']
        dict_corr_values = {}
        pairs_ = list(combinations(group_metrics, 2))
        pairs_ = [('skyline', 'CAM'), ('skyline', 'MM'), ('euclidean', 'CAM'),
                  ('euclidean', 'MM')]
        for comb in pairs_:
            for query in self.queries:
                tau, p_value = stats.kendalltau(self.dict_of_models_order_by_query[comb[0]][query],
                                                self.dict_of_models_order_by_query[comb[1]][query])
                if tau not in dict_corr_values:
                    dict_corr_values[tau] = ''
                # if tau <= 0.0:
                if comb[0] not in low_corr:
                    low_corr[comb[0]] = {}
                    dict_corrs_freq[comb[0]] = {}
                if comb[1] not in low_corr[comb[0]]:
                    low_corr[comb[0]][comb[1]] = {}
                    dict_corrs_freq[comb[0]][comb[1]] = {'x': [], 'y': []}
                if query not in low_corr[comb[0]][comb[1]]:
                    low_corr[comb[0]][comb[1]][query] = tau
        # pprint.pprint(low_corr)
        # exit()
        for comb in pairs_:
            sort_list = sorted(low_corr[comb[0]][comb[1]],key=low_corr[comb[0]][comb[1]].get)
            # print(r'\textbf{'+str(comb).replace("'",'')+'}&','&'.join(['$t_{'+str(val)+'}$'+'('+str(round(low_corr[comb[0]][comb[1]][val],2))+')' for val in sort_list[:3]]),r'\\')
            for query in sort_list[:1]:
                print(comb, "Topic:", query)
                # model2= "a"
                # model1 = model2
                # while model1 == model2:
                #     val = random.randrange(len(self.dict_of_models_order_by_query[comb[0]][query]))
                model1 = self.dict_of_models_order_by_query[comb[0]][query][-1]
                model2 = self.dict_of_models_order_by_query[comb[1]][query][-1]

                print('Run1',model1.split('/')[-1],":",self.dict_results[model1][str(query)][comb[0]],'\tRun2' ,model2.split('/')[-1],":",self.dict_results[model2][str(query)][comb[1]])
                self.get_list_rel_docs_in_both_ranks(model1, model2, query)

    def get_list_rel_docs_in_both_ranks(self, run1, run2, topic):
        with open('data/processed_data/qrels/qrels_discrete_aspects.txt') as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        dict_rel_docs = {}
        topic = str(topic)
        for line in content:
            parts = line.split()
            if parts[0] == topic:
                for weight in parts[3:]:
                    if int(weight) > 0:
                        if parts[2] not in dict_rel_docs:
                            dict_rel_docs[parts[2]] = parts[3:]

        dict_res = {'run1':{},"run2":{}}
        if '2015' in run1:
            base_path = "data/runs_track_newids/2015/"
        else:
            base_path = 'data/runs_track_newids/2016/'
        with open (base_path+run1.split('/')[-1]) as f:
            content = f.readlines()
            for line in content:
                parts = line.split()
                if parts[0] == topic:
                    "36 Q0 clueweb12-0102wb-96-21901 15 101.811622 udelRun1C"
                    if parts[2] in dict_rel_docs:
                        if parts[2] not in dict_res['run1']:
                            if int(parts[3]) < 20:
                                dict_res['run1'][parts[2]] = {'pos':parts[3], 'weights':dict_rel_docs[parts[2]]}

        with open (base_path+run2.split('/')[-1]) as f:
            content = f.readlines()
            for line in content:
                parts = line.split()
                if parts[0] == topic:
                    if parts[2] in dict_rel_docs:
                        if parts[2] not in dict_res['run2']:
                            # if int(parts[3]) < 30:
                            dict_res['run2'][parts[2]] = {'pos':parts[3], 'weights':dict_rel_docs[parts[2]]}

        pprint.pprint(dict_res)

    def run(self):
        self.preprocess_data()
        # exit()
        # self.correlation_average_evaluation_score()
        self.calc_corr_by_topic()
        # pprint.pprint(self.dict_of_models_order_by_query)
        ###CALL THIS TO PLOT CORR_BY TOPIC###
        # self.plot_corr_vals_by_topic()

        # self.get_least_correlated_topics()
        # self.get_list_rel_docs_in_both_ranks('','',30)
        # self.plot_corr_by_topic()

if __name__ == '__main__':
    corr_analysis = CorrAnalysis()
