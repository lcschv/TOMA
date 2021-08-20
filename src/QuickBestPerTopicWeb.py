from os import listdir
import pandas as pd
import pprint
import numpy as np


for measure in ['AP','NDCG']:
    for track in ['2009','2010', '2011', '2012', '2013', '2014']:
        # folder = "../data/results/TaskTrack2015/"
        # folder = "../data/results_cutoff100_ap/Decision/"
        # folder = "../data/results_cutoff100_ap/TaskTrack2016/"
        if measure == 'AP':
            folder = "data/results_weighting_cutoff_5_ap/WebTrack"+str(track)+"/"
        else:
            folder = "data/results_weighting_cutoff_5/WebTrack"+str(track)+"/"
        # instantiation = "AP"
        instantiation = measure
        # runs_folder = "../data/decisionRuns/"
        runs_folder = "data/webtrack"+track+"Runs_extracted/"
        # runs_folder = "../data/runs_track_newids/2016/"
        qrels_file = "data/qrels_webtrack"+track+"/qrels_discrete_aspects.txt"
        # qrels_file = "../data/qrels_decision/qrels_decision_3aspects.txt"
        # qrels_file = "../data/processed_data/qrels/qrels_2016_4aspects.txt"
        # results_folder = "../data/top_100_per_track_permetric_perquery_cutoff/Decision2019/"
        # results_folder = "../data/top_100_per_track_permetric_perquery_cutoff/Task2016/"
        # trec= "TASK2016"
        trec= "WEB"+track
        """GET METRIC SCORES PER FILE"""
        # folder = "../data/results/TaskTrack2015/"
        # folder = "../data/results_ap_cufoff/WebTrack2014/"
        # runs_folder = "../data/webtrack2014Runs_extracted/"
        # runs_folder = "../data/runs_track_newids/2015/"
        # qrels_file = "../data/qrels_webtrack2014/qrels_discrete_aspects.txt"
        # qrels_file = "../data/processed_data/qrels/qrels_2015_4aspects.txt"
        results_folder = "data/results_weighting/top_5_per_track_cutoff/WebTrack"+track+'/'
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
        """GET BEST RUN PER TOPIC"""
        dict_best_per_topic = {}
        for metric in header[1:]:
            if metric not in dict_best_per_topic:
                dict_best_per_topic[metric] = {}
            for qid in dict_qid:
                if qid not in dict_best_per_topic[metric]:
                    dict_best_per_topic[metric][qid] = ('None', -1)
                for run in files:
                    if dict_best_per_topic[metric][qid][1] < dict_results[run][str(qid)][metric]:
                        dict_best_per_topic[metric][qid] = (run, dict_results[run][str(qid)][metric])

        # pprint.pprint(dict_best_per_topic)

        """GET TOP 5 RANK PER TOPIC OF THE PER TOPIC BEST RUNS"""
        dict_top_ranks5_per_topic = {}
        for metric, run_score in dict_best_per_topic.items():
            if metric not in dict_top_ranks5_per_topic:
                dict_top_ranks5_per_topic[metric] = {}
            for qid in dict_qid:
                if str(qid) not in dict_top_ranks5_per_topic[metric]:
                    dict_top_ranks5_per_topic[metric][str(qid)] = []
                with open(runs_folder+run_score[qid][0].replace('.csv','')) as f:
                    content = f.readlines()
                content = [x.rstrip() for x in content]
                for line in content:
                    try:
                        if trec == "TASK2015" or trec == "TASK2016" or trec == "WEB2010":
                            parts = line.split()
                        else:
                            parts = line.split('\t')
                        if parts[0] in dict_top_ranks5_per_topic[metric]:
                            if len(dict_top_ranks5_per_topic[metric][parts[0]]) < 5:
                                dict_top_ranks5_per_topic[metric][parts[0]] += [parts[2]]
                    except:
                        print(runs_folder+run_score[qid][0].replace('.csv',''), line)
        # print(dict_best_per_topic['CAM'][1])
        qrels_dict = {}
        with open(qrels_file) as f:
            content = f.readlines()
            for line in content:
                parts = line.split()
                if parts[0] not in qrels_dict:
                    qrels_dict[parts[0]] = {}
                if parts[2] not in qrels_dict[parts[0]]:
                    qrels_dict[parts[0]][parts[2]] = parts[3:]

        metrics_to_print = {'CAM': 'CAM', 'MM': 'MM', 'euclidean': 'EUCL', 'manhattan': 'MANH', 'chebyshev': 'CHEB',
                            'skyline': 'Skyline', 'double_aspect-1_chebyshev': 'D1-CHEB',
                            'double_aspect-2_chebyshev': 'D2-CHEB',
                            'double_aspect-3_chebyshev': 'D3-CHEB', 'double_aspect-1_manhattan': 'D1-MANH',
                            'double_aspect-2_manhattan': 'D2-MANH',
                            'double_aspect-3_manhattan': 'D3-MANH',
                            'equispaced_samerange_aspect-1_manhattan': 'E1-MANH',
                            'equispaced_samerange_aspect-2_manhattan': 'E2-MANH',
                            'equispaced_samerange_aspect-3_manhattan': 'E3-MANH',
                            'equispaced_samerange_aspect-1_chebyshev': 'E1-CHEB',
                            'equispaced_samerange_aspect-2_chebyshev': 'E2-CHEB',
                            'equispaced_samerange_aspect-3_chebyshev': 'E3-CHEB',
                            'equispaced_samerange_aspect-1_euclidean': 'E1-EUCL',
                            'equispaced_samerange_aspect-2_euclidean': 'E2-EUCL',
                            'equispaced_samerange_aspect-3_euclidean': 'E3-EUCL',
                            'double_aspect-1_euclidean': 'D1-EUCL', 'double_aspect-2_euclidean': 'D2-EUCL',
                            'double_aspect-3_euclidean': 'D3-EUCL', 'urbp': 'uRBP'
                            }
        for metric, top5 in dict_top_ranks5_per_topic.items():
            if metric in metrics_to_print:
                metric_file_name = metrics_to_print[metric]
            else:
                continue
            file_out = open(results_folder+trec+'-'+metric_file_name+'-'+instantiation+'_perquery_cutoff.txt', 'w')
            for qid in top5:
                for doc_pos, document in enumerate(dict_top_ranks5_per_topic[metric][qid]):
                    if document in qrels_dict[qid]:
                        print(qid, doc_pos+1,'#'.join([str(x) for x in qrels_dict[qid][document]]), sep="#", file=file_out)
                    else:
                        print(qid, doc_pos+1,"NA", "NA", "NA",sep="#",file=file_out)
            file_out.close()
