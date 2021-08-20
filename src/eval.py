import pprint
from os import listdir
from os.path import isfile, join
from pathlib import Path
import subprocess
import os
from src.params import Params
from collections import OrderedDict
from itertools import islice


class Eval(Params):

    def __init__(self):
        super(Eval, self).__init__()
        self.eval_parameters = self.get_order_parameters(qrels_config="config/eval.json")

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def take(self, n, iterable):
        "Return first n items of the iterable as a list"
        return list(islice(iterable, n))

    def get_top_runs(self):
        models = self.eval_parameters['top_runs_experiment']['models']
        number_top_runs = self.eval_parameters['top_runs_experiment']['number_top_runs']
        metric = self.eval_parameters['top_runs_experiment']['metric']

        dict_results_run = {model: {} for model in models}
        for model in models:
            tmp = []
            runs = self.get_list_files(self.eval_parameters['top_runs_experiment']['runs_folder'] + model+"/")
            for file in runs:
                if file not in dict_results_run[model]:
                    dict_results_run[model][file] = 0

                # print('../trec_eval-master/trec_eval', '-m', metric, self.eval_parameters['qrels_file'], file)

                out1 = subprocess.check_output(
                    ['../trec_eval-master/trec_eval', '-m', metric, self.eval_parameters['top_runs_experiment']['qrels_file'], file])
                val = str(out1.rstrip().split()[2]).replace('b\'','').replace('\'','')
                dict_results_run[model][file] = float(val)


        fileout = open('data/top_runs/info.txt',"w")
        for model, runs in dict_results_run.items():
            Path('data/top_runs/'+model).mkdir(parents=True, exist_ok=True)
            print("BestRuns:", model, file=fileout)
            sorted_runs = OrderedDict(sorted(runs.items(), key=lambda t: t[1],reverse=True))
            n_items = self.take(number_top_runs, sorted_runs.items())

            for run, score in n_items:
                os.system('cp '+run +" "+"data/top_runs/"+model)
                print(run,score, file=fileout)

        fileout.close()

    def get_average_scores(self):
        """Compute the average scores of the measures given in 'average_experiment' in the eval.json file."""
        models = self.eval_parameters['average_experiment']['models']
        metrics = self.eval_parameters['average_experiment']['metrics_list']
        metrics_keys = list(metrics.keys())
        print(r"\begin{table}[]")
        print("\centering")
        print(r"\tiny")
        print("\caption{Average results over 85 topics. Each row represents a different run (top 10 runs of each model). Each column represents a different assessments aggregation.}")
        print("\label{tab:average_results}")
        print(r"\begin{tabular}{@{}"+''.join(['l']*(len(metrics_keys)+1))+"@{}}")
        print("runid", '&'.join(metric.replace('_','\_')for metric in metrics_keys),sep='&')
        print(r"\\ \midrule")
        for model in models:
            runs = self.get_list_files(self.eval_parameters['average_experiment']['runs_folder'] + model+"/")
            for file in runs:
                val = []
                for metric_id in  metrics_keys:
                    out1 = subprocess.check_output(
                        ['../trec_eval-master/trec_eval', '-m', metrics[metric_id]['metric'],
                         metrics[metric_id]['qrels'], file])
                    val += [str(out1.rstrip().split()[2]).replace('b\'', '').replace('\'', '')]
                print(file.replace(self.eval_parameters['average_experiment']['runs_folder'],'').replace('_','\_'),'&', '&'.join(val),r"\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    def run_trec_eval(self, qrels, run, metric, queries=False):
        dict_results = {}
        if queries:
            out1 = subprocess.check_output(
                ['../trec_eval-master/trec_eval', "-q",'-m', metric,
                 qrels, run])
            content = out1.split(b'\n')
            # content = [x.rstrip() for x in content]
            for line in content[:-2]:
                val = [str(x).replace('b\'', '').replace('\'','') for x in line.rstrip().split()]
                # val = str(line.rstrip().split()).replace('b\'', '')
                if len(val) == 0:
                    continue
                qid = val[1]
                score = val[2]
                if qid not in dict_results:
                    dict_results[qid] = float(score)
        else:
            out1 = subprocess.check_output(
                ['../trec_eval-master/trec_eval', '-m', metric,
                 qrels, run])
            val = str(out1.rstrip().split()[2]).replace('b\'', '').replace('\'', '')
            dict_results['all'] = float(val)
        return dict_results
