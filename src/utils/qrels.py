import pprint
from src.params import Params

class Qrels(Params):
    def __init__(self):
        super(Qrels, self).__init__()
        self.qrels_parameters = self.get_order_parameters("config/qrelswebtrack2009.json")
        self.pagerank_values = []
        self.spam_score_values = []
        self.dict_qrels = self.read_qrels()
        self.pagerank_cut_levels = self.get_discriminative_pagerankvalues()
        pprint.pprint(self.pagerank_cut_levels)
        self.spam_cut_levels = self.qrels_parameters['spam_cut_levels']
        self.dict_discriminated_qrels = self.get_discriminate_dict_qrels()
        self.write_qrels_to_file(self.dict_discriminated_qrels, self.qrels_parameters['new_qrels_path'])

    def get_file_content(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        return content

    def read_qrels(self):
        dict_qrels = {}
        content = self.get_file_content(self.qrels_parameters['qrels_file'])
        for line in content:
            parts = line.split()
            qid = parts[0]
            docid = parts[2]
            if qid not in dict_qrels:
                dict_qrels[qid] = {}
            if docid not in dict_qrels[qid]:
                dict_qrels[qid][docid] = {}
            dict_qrels[qid][docid] = {aspect:weight for aspect, weight in enumerate(parts[3:])}
            self.pagerank_values += [float(parts[4])]
        return dict_qrels

    def get_discriminative_pagerankvalues(self):
        percentages = self.qrels_parameters['pagerank_distribution']
        self.pagerank_values = sorted(self.pagerank_values)
        previous_percentage = 0
        num_pagerank = len(self.pagerank_values)
        continuos_values = {}
        previous_value = 0
        for aspect, ratio in percentages.items():
            new_ratio = ratio+previous_percentage
            if new_ratio != 1:
                pos = int((new_ratio*num_pagerank)) - num_pagerank

                continuos_values[aspect] = [previous_value, self.pagerank_values[pos]]
                previous_value = self.pagerank_values[pos]

            else:
                continuos_values[aspect] = [previous_value, self.pagerank_values[-1]]
                previous_value = self.pagerank_values[-1]
            previous_percentage += ratio
        return continuos_values

    def discriminate_value(self, value, dict_labels):
        for label, range in dict_labels.items():
            if value >= range[0] and value <= range[1]:
                return label

    def get_discriminate_dict_qrels(self):
        dict_qrels = {}
        content = self.get_file_content(self.qrels_parameters['qrels_file'])
        for line in content:
            parts = line.split()
            qid = parts[0]
            docid = parts[2]
            rel= int(parts[3])
            if rel < 0:
                rel = 0
            # usefulness = int(parts[4])
            pagerank = float(parts[4])
            spam_score = int(parts[5])
            if qid not in dict_qrels:
                dict_qrels[qid] = {}
            if docid not in dict_qrels[qid]:
                dict_qrels[qid][docid] = {}

            pr_label = self.discriminate_value(pagerank, self.pagerank_cut_levels)
            cred_label = self.discriminate_value(spam_score, self.spam_cut_levels)
            aspects = [rel, pr_label, cred_label]
            dict_qrels[qid][docid] = {aspect: weight for aspect, weight in enumerate(aspects)}
            # print(qid, docid, rel, usefulness, pr_label, cred_label)
        # pprint.pprint(dict_qrels)
        return dict_qrels

    def write_qrels_to_file(self, dict, filename):
        fileout = open(filename, "w")
        for qid, docs in dict.items():
            for doc, aspects in docs.items():
                print(qid, "Q0", doc, ' '.join([str(val) for k,val in aspects.items()]), file=fileout)
        fileout.close()
