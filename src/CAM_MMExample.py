from os import listdir
import pandas as pd
import pprint
import numpy as np

"""GET METRIC SCORES PER FILE"""
# folder = "../data/results/TaskTrack2015/"
# folder = "../data/results_cufoff/TaskTrack2016/"
folder = "data/ExampleCAMMM/"
# instantiation = "AP"
# # runs_folder = "../data/decisionRuns/"
# runs_folder = "../data/webtrack2013Runs_extracted/"
# # runs_folder = "../data/runs_track_newids/2016/"
# qrels_file = "../data/qrels_webtrack2013/qrels_discrete_aspects.txt"
# # qrels_file = "../data/qrels_decision/qrels_decision_3aspects.txt"
# # qrels_file = "../data/processed_data/qrels/qrels_2016_4aspects.txt"
# # results_folder = "../data/top_5_per_track_cutoff/Decision2019/"
# results_folder = "../data/top_5_per_track_perquery_cutoff/Web2013/"
# # results_folder = "../data/top_5_per_track_cutoff/Task2016/"
# trec= "WEB2012"


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
pprint.pprint(dict_results)
"""GET BEST RUN PER METRIC"""
dict_average_per_file = {}
for metric in ["chebyshev","euclidean","manhattan","rel","cred","correc","CAM","MM"]:
    if metric not in dict_average_per_file:
        dict_average_per_file[metric] = {}
    for file in files:
        if file not in dict_average_per_file[metric]:
            dict_average_per_file[metric][file] = []
        arr = []
        for qid in dict_qid:
            arr += [dict_results[file][str(qid)][metric]]
        dict_average_per_file[metric][file] = np.average(arr)
pprint.pprint(dict_average_per_file)

first = ['rel', 'cred', 'correc']
sec = ['CAM','MM']
third = ['chebyshev', 'manhattan', 'euclidean']

print("Run_id | AP(relevance) AP(credibility) AP(correctness) | CAM(AP) MM(AP) | Toma_chebyshev(AP) TOMA_manhattan(AP) TOMA_euclidean(AP)")


print("UWaterMDS_BM25(Baseline)", ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_AP.csv'],4)) for x in first]),
      ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_AP.csv'],4)) for x in sec]),
      ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_AP.csv'],4)) for x in third]),sep='|')
print("UWatMDSBM25_HC3(Advanced)", ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_AP.csv'],4)) for x in first]),
      ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_AP.csv'],4)) for x in sec]),
      ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_AP.csv'],4)) for x in third]),sep='|')



print("Run_id | nDCG(relevance) nDCG(credibility) nDCG(correctness) | CAM(nDCG) MM(nDCG) | Toma_chebyshev(nDCG) TOMA_manhattan(nDCG) TOMA_euclidean(nDCG)")


print("UWaterMDS_BM25(Baseline)", ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_NDCG.csv'],4)) for x in first]),
      ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_NDCG.csv'],4)) for x in sec]),
      ','.join([str(round(dict_average_per_file[x]['UWaterMDS_BM25_NDCG.csv'],4)) for x in third]), sep='|')

print("UWatMDSBM25_HC3(Advanced)", ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_NDCG.csv'],4)) for x in first]),
      ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_NDCG.csv'],4)) for x in sec]),
      ','.join([str(round(dict_average_per_file[x]['UWatMDSBM25_HC3_NDCG.csv'],4)) for x in third]),sep='|')

