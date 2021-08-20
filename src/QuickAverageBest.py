from os import listdir
import pandas as pd
import pprint
import numpy as np



for measure in ['AP','NDCG']:
    for track in ['adhoc','recall']:
        # folder = "../data/results/TaskTrack2015/"
        # folder = "../data/results_cutoff100_ap/Decision/"
        # folder = "../data/results_cutoff100_ap/TaskTrack2016/"
        if measure == 'AP':
            folder = "data/results/cutoff_5/Misinfo-"+str(track)+'_ap'+"/"
        else:
            folder = "data/results/cutoff_5/Misinfo-"+str(track)+"/"
        # instantiation = "AP"
        instantiation = measure
        # runs_folder = "../data/decisionRuns/"
        runs_folder = "data/misinfo-runs_extracted/"+track+'/'
        # runs_folder = "../data/runs_track_newids/2016/"
        qrels_file = "data/qrels_misinfo/qrels/misinfo-qrels"
        # qrels_file = "../data/qrels_decision/qrels_decision_3aspects.txt"
        # qrels_file = "../data/processed_data/qrels/qrels_2016_4aspects.txt"
        # results_folder = "../data/top_100_per_track_permetric_perquery_cutoff/Decision2019/"
        # results_folder = "data/results/top_5_per_track_perquery_cutoff/Misinfo-"+track+'/'
        # results_folder = "../data/top_100_per_track_permetric_perquery_cutoff/Task2016/"
        # trec= "TASK2016"
        trec= "MISINFO2020"+track
        """GET METRIC SCORES PER FILE"""
        # folder = "../data/results/TaskTrack2015/"
        # folder = "../data/results_ap_cufoff/WebTrack2014/"
        # runs_folder = "../data/webtrack2014Runs_extracted/"
        # runs_folder = "../data/runs_track_newids/2015/"
        # qrels_file = "../data/qrels_webtrack2014/qrels_discrete_aspects.txt"
        # qrels_file = "../data/processed_data/qrels/qrels_2015_4aspects.txt"
        results_folder = "data/results/top_5_per_track_cutoff/Misinfo-"+track+'/'
        # results_folder = "../data/top_5_per_track_perquery_cutoff/Web2014/"
        # results_folder = "../data/top_5_per_track_perquery_cutoff/Task2015/"


        files = listdir(folder)
        dict_results = {}
        header = []
        dict_qid = {}
        for file in files:
            with open(folder+file) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if file not in dict_results:
                dict_results[file] = {}
            header = content[0].split(',')
            for line in content[1:]:
                parts = line.split(',')
                if parts[0] not in dict_qid:
                    dict_qid[int(parts[0])] = ''
                if parts[0] not in dict_results[file]:
                    dict_results[file][parts[0]] = {k: float(v) for k,v in zip(header[1:], parts[1:])}
        dict_qid = sorted(list(dict_qid.keys()))
        # pprint.pprint(dict_results)
        """GET BEST RUN PER METRIC"""
        dict_average_per_file = {}
        for metric in header[1:]:
            if metric not in dict_average_per_file:
                dict_average_per_file[metric] = {}
            for file in files:
                if file not in dict_average_per_file[metric]:
                    dict_average_per_file[metric][file] = []
                arr = []
                for qid in dict_qid:
                    arr += [dict_results[file][str(qid)][metric]]
                dict_average_per_file[metric][file] = np.average(arr)
        dict_best_run_per_metric = {}
        for metric, runs in dict_average_per_file.items():
            if metric not in dict_best_run_per_metric:
                dict_best_run_per_metric[metric] = ('run_id',-1)
            for runid, average_score in runs.items():
                if average_score > dict_best_run_per_metric[metric][1]:
                    dict_best_run_per_metric[metric] = (runid,average_score)
        # pprint.pprint(dict_best_run_per_metric)

        """GET TOP 5 RANK PER TOPIC OF THE BEST RUNS"""
        dict_top_ranks5_per_topic = {}
        for metric, run_score in dict_best_run_per_metric.items():
            if metric not in dict_top_ranks5_per_topic:
                dict_top_ranks5_per_topic[metric] = {}
            for qid in dict_qid:
                if str(qid) not in dict_top_ranks5_per_topic[metric]:
                    dict_top_ranks5_per_topic[metric][str(qid)] = []
            with open(runs_folder+run_score[0].replace('.csv','')) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            for line in content:
                if trec == "TASK2015" or trec == "TASK2016" or trec == "WEB2010":
                    parts = line.split()
                else:
                    parts = line.split('\t')
                if parts[0] in dict_top_ranks5_per_topic[metric]:
                    if len(dict_top_ranks5_per_topic[metric][parts[0]]) < 5:
                        dict_top_ranks5_per_topic[metric][parts[0]] += [parts[2]]

        # pprint.pprint(dict_top_ranks5_per_topic)
        qrels_dict = {}
        with open(qrels_file) as f:
            content = f.readlines()
            for line in content:
                parts = line.split()
                if parts[0] not in qrels_dict:
                    qrels_dict[parts[0]] = {}
                if parts[2] not in qrels_dict[parts[0]]:
                    qrels_dict[parts[0]][parts[2]] = parts[3:]


        metrics_to_print = {'CAM':'CAM','MM':'MM','euclidean':'EUCL','manhattan':'MANH','chebyshev':'CHEB'}
        for metric, top5 in dict_top_ranks5_per_topic.items():
            if metric in metrics_to_print:
                metric_file_name = metrics_to_print[metric]
            else:
                continue
            file_out = open(results_folder+trec+'-'+metric_file_name+'-'+instantiation+'_cutoff.txt', 'w')
            for qid in top5:
                for doc_pos, document in enumerate(dict_top_ranks5_per_topic[metric][qid]):
                    if document in qrels_dict[qid]:
                        print(qid, doc_pos+1,'#'.join([str(x) for x in qrels_dict[qid][document]]), sep="#", file=file_out)
                    else:
                        print(qid, doc_pos+1,"NA", "NA", "NA",sep="#",file=file_out)
            file_out.close()
