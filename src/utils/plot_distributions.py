from src.params import Params
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd



class Distributions(Params):
    def __init__(self):
        super(Distributions, self).__init__()
        self.parameters = self.get_order_parameters('config/plot_distributions.json')

        # 1st to run
        self.dict_qrels = self.read_qrels()

        # 2nd to run
        self.dict_frequency_per_query, self.number_of_runs = self.get_frequency_rank_pos_per_query()

    def read_qrels(self):
        dict_qrels = {}
        content = self.get_file_content(self.parameters['qrels_file'])
        for line in content:
            parts = line.split()
            qid = int(parts[0])
            docid = parts[2]
            if qid not in dict_qrels:
                dict_qrels[qid] = {}
            if docid not in dict_qrels[qid]:
                dict_qrels[qid][docid] = {}
            dict_qrels[qid][docid] = {aspect:int(weight) for aspect, weight in enumerate(parts[3:])}
        return dict_qrels


    def get_frequency_rank_pos_per_query(self):
        dict_frequency_per_query = {}
        files = self.get_list_files(self.parameters['runs_folder'])
        number_runs = len(files)
        for run in files:
            content = self.get_file_content(run)
            for line in content:
                parts = line.split()
                "1  Q0 clueweb12-1600wb-85-02070 2 46.2239 merged_task_track"
                qid= int(parts[0])
                _ = parts[1]
                docid = parts[2]
                pos = int(parts[3])

                # For each aspect creates an array size of the ranking
                if qid not in dict_frequency_per_query:
                    dict_frequency_per_query[qid] = {aspect:[0]*self.parameters['rank_size'] for aspect in range(self.parameters['number_aspects'])}

                # Check if docid was assessed as 'relevant' for each aspect
                if qid in self.dict_qrels:
                    if docid in self.dict_qrels[qid]:
                        for aspect in self.dict_qrels[qid][docid]:
                            if self.dict_qrels[qid][docid][aspect] > 0:
                                dict_frequency_per_query[qid][aspect][pos-1] += 1
        return dict_frequency_per_query, number_runs


    def plot_num_relevant_per_query(self):
        dict_count_rel = {}
        for query, docs in self.dict_qrels.items():
            if query not in dict_count_rel:
                dict_count_rel[query] = {k:0 for k in range(self.parameters['number_aspects'])}
            for doc, assessments in docs.items():
                for k,v in assessments.items():
                    if v>0:
                        dict_count_rel[query][k] += 1
        x = list(dict_count_rel.keys())

        # pprint(dict_count_rel)
        # exit()
        frec_aspect_by_query = {}
        for aspect in range(self.parameters['number_aspects']):
            if aspect not in frec_aspect_by_query:
                frec_aspect_by_query[aspect] = []
            for query in x:
                frec_aspect_by_query[aspect] += [dict_count_rel[query][aspect]]
        aspects = ['relevant', 'usefull', 'popular', 'credible']
        color = ['b','g','r','c']

        fig = plt.figure()
        for i, (k, v) in enumerate(frec_aspect_by_query.items()):
            # plt.plot(np.array(self.dict_frequency_per_query[41][0][:10])/self.number_of_runs, label='Rel')
            plt.subplot(2,2,i+1)
            plt.plot(x, sorted(v,reverse=True), linestyle="", marker='o',markerfacecolor='k',color=color[i], label=aspects[i], markersize=3, linewidth=1)
            # plt.xlabel('Queries')
            plt.ylim(top=100)
            plt.legend()  # To draw legend
            plt.grid(True)
        # plt.ylabel('Doc assessed as '+aspects[i])
        # fig.text(50,100,'Common Y', va='center',rotation='vertical')

        # plt.show()
        plt.savefig('data/dataset_statistics/num_rel_by_query.png')


    def plot_query_distributions(self):
        for query, dict_aspects in self.dict_frequency_per_query.items():
            # print(self.dict_frequency_per_query.keys())
            # line_styles = ['.-', '.--', '.:', '.-.', '*-', '+--', 'x:', 'o-.']
            markers = ['o','x','*','>']
            aspects = ['rel', 'use', 'pop', 'cred']
            for i, (k, v) in enumerate(dict_aspects.items()):
                # plt.plot(np.array(self.dict_frequency_per_query[41][0][:10])/self.number_of_runs, label='Rel')
                plt.plot(range(1,self.parameters['max_rank_pos']+1), np.array(v[:self.parameters['max_rank_pos']])/self.number_of_runs,linestyle="-",marker=markers[i], label=aspects[i])
                plt.xticks(range(1,20+1))
                plt.grid(True)
                # plt.xlim(left=1)
        # plt.hist(dict_aspects[0], bins='auto')
            plt.legend()  # To draw legend
            # plt.show()
            plt.savefig(self.parameters['plots_path']+str(query)+'.png')
            plt.clf()


    def get_frequency_by_rank_pos(self):
        dict_frequency_per_rank = {}
        # files = self.get_list_files(self.parameters['runs_folder'])
        files = self.get_list_files("data/quick_solution/")
        number_runs = len(files)

        for run in files:
            content = self.get_file_content(run)
            for line in content:
                parts = line.split()
                "1  Q0 clueweb12-1600wb-85-02070 2 46.2239 merged_task_track"
                qid= int(parts[0])
                _ = parts[1]
                docid = parts[2]
                pos = int(parts[3])

                # For each aspect creates an array size of the ranking
                if pos not in dict_frequency_per_rank:
                    dict_frequency_per_rank[pos] = {aspect:0 for aspect in range(self.parameters['number_aspects'])}
                for aspect in range(self.parameters['number_aspects']):
                    if qid in self.dict_qrels and docid in self.dict_qrels[qid] and self.dict_qrels[qid][docid][aspect] > 0:
                        dict_frequency_per_rank[pos][aspect] += 1
        labels = ['Relevance', 'Usefulness', 'Popularity', 'Credibility']
        df = pd.DataFrame.from_dict(dict_frequency_per_rank, orient='index')
        sns.heatmap(df[:50], cmap='RdYlGn_r', xticklabels=labels, robust=True)
        plt.savefig('data/dataset_statistics/heatmap_rankpos_topruns.png')
        plt.clf()
        # plt.show()
        # dict_frequenc_by_aspect = {}
        # for aspect in range(self.parameters['number_aspects']):
        #     for i in range(1,len(dict_frequency_per_rank)+1):
        #         if aspect not in dict_frequenc_by_aspect:
        #             dict_frequenc_by_aspect[aspect] = []
        #         dict_frequenc_by_aspect[aspect] += [(i,dict_frequency_per_rank[i][aspect])]
        #     dict_frequenc_by_aspect[aspect] = sorted(dict_frequenc_by_aspect[aspect], key=lambda tup: tup[1], reverse=True)


        markers = ['o', 'x', '*', '>']
        aspects = ['rel', 'use', 'pop', 'cred']

        # for i, (k, v) in enumerate(dict_frequenc_by_aspect.items()):
        #     # plt.plot(np.array(self.dict_frequency_per_query[41][0][:10])/self.number_of_runs, label='Rel')
        #     plt.plot(range(1, self.parameters['max_rank_pos'] + 1),
        #              np.array(v[:self.parameters['max_rank_pos']]) / self.number_of_runs, linestyle="-",
        #              marker=markers[i], label=aspects[i])
        #     plt.xticks(range(1, 20 + 1))
        #     plt.grid(True)
        # # return dict_frequency_per_rank, number_runs

    def plot_cumulative_dist_rank_pos(self):
        dict_frequency_per_rank = {}
        # files = self.get_list_files(self.parameters['runs_folder'])
        files = self.get_list_files('data/runs_track_newids/2016/')
        # files = self.get_list_files(self.runs)
        number_runs = len(files)

        for run in files:
            content = self.get_file_content(run)
            for line in content:
                parts = line.split()
                "1  Q0 clueweb12-1600wb-85-02070 2 46.2239 merged_task_track"
                qid = int(parts[0])
                _ = parts[1]
                docid = parts[2]
                pos = int(parts[3])

                # For each aspect creates an array size of the ranking
                if pos not in dict_frequency_per_rank:
                    dict_frequency_per_rank[pos] = {aspect: 0 for aspect in range(self.parameters['number_aspects'])}
                for aspect in range(self.parameters['number_aspects']):
                    if qid in self.dict_qrels and docid in self.dict_qrels[qid] and self.dict_qrels[qid][docid][aspect] > 0:
                        dict_frequency_per_rank[pos][aspect] += 1

        dict_cumulated_per_rank = {aspect:[] for aspect in range(self.parameters['number_aspects'])}
        for rank_pos, vals in dict_frequency_per_rank.items():
            for k,v in vals.items():
                dict_cumulated_per_rank[k].append(v)
        for aspect in range(self.parameters['number_aspects']):
            dict_cumulated_per_rank[aspect] = np.cumsum(dict_cumulated_per_rank[aspect]/np.sum(dict_cumulated_per_rank[aspect]))


        markers = [':', '-', '-.', '--']
        aspects = ['relevance', 'usefulness', 'popularity', 'credibility']
        for i, (k, v) in enumerate(dict_cumulated_per_rank.items()):
            plt.plot(range(1, self.parameters['rank_size']+1),
                     np.array(v), linestyle=markers[i],
                     label=aspects[i])
            plt.grid(True)
        plt.xlabel('Rank Position')
        plt.ylabel('Frequency of relevant documents')
        plt.legend()  # To draw legend
        # plt.show()
        plt.savefig('data/dataset_statistics/'+ '2016_cumulated_frequency' + '.png')
        plt.show()
        plt.clf()