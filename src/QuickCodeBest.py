from os import listdir
import pandas as pd
import pprint
import numpy as np

# folder = "../data/results_ap/WebTrack2012/"
# files = listdir(folder)
# dict_results = {}
# header = []
# for file in files:
#     with open(folder+file) as f:
#         content = f.readlines()
#     content = [x.rstrip() for x in content]
#     if file not in dict_results:
#         dict_results[file] = {}
#     header = content[0].split(',')
#     for line in content[1:]:
#         parts = line.split(',')
#         if parts[0] not in dict_results[file]:
#             dict_results[file][parts[0]] = {k: float(v) for k,v in zip(header[1:], parts[1:])}
# dict_average = {}
# for file, qids in dict_results.items():
#     for qid, scores in qids.items():
#         if qid not in dict_average:
#             dict_average[qid] = {}
#         for metric, score in scores.items():
#             if metric not in dict_average[qid]:
#                 dict_average[qid][metric] = []
#             dict_average[qid][metric] += [score]
# file_out = open('../data/results_best_all_average/Web2012/map.txt', 'w')
# print(','.join(header), file=file_out)
# for qid in sorted([int(x) for x in list(dict_average.keys())]):
#     print(str(qid), ','.join([str(round(np.average(dict_average[str(qid)][metric]),4)) for metric in header[1:]]),sep=',', file=file_out)
# pprint.pprint(dict_average)

# #
dict_final = {}
folder = "../data/results_best_all_average/Web2009/"
files = listdir(folder)
for file in files:
    metric = file.replace('.txt', '')
    with open(folder+file) as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    if metric not in dict_final:
        dict_final[metric] = {}
    header = content[0].split(',')
    for line in content[1:]:
        parts = line.split(',')
        if parts[0] not in dict_final[metric]:
            dict_final[metric][parts[0]] = {k: float(v) for k, v in zip(header[1:], parts[1:])}
# pprint.pprint(dict_final)
single = ['rel', 'cred', 'correc']
# single = ['rel', 'use', 'pop', 'spam']
# single = ['rel', 'pop', 'spam']
# print(dict_final)
file_out = open('../data/results_best_all_average/WEB2009-ALL.txt','w')
for qid in sorted([int(x) for x in list(dict_final['map'].keys())]):
    print(qid, '#'.join([str(dict_final[metric[0]][str(qid)][metric[1]]) for metric in [('ndcg','CAM'),('ndcg','MM'),('ndcg','euclidean'),('ndcg','manhattan'),('ndcg','chebyshev'),
                   ('map','CAM'),('map','MM'),('map','euclidean'),('map','manhattan'),('map','chebyshev'),
                   ('ndcg','rel'),('ndcg','cred'),('ndcg','correc'),
                   ('map','rel'),('map','cred'),('map','correc')]]),
                    sep='#',file=file_out)
file_out.close()

# for metric in ['ndcg', 'map']:
#     for qid in sorted()



