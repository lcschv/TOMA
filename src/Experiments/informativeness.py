from src.eval import Eval
import pprint
import numpy as np
from scipy.optimize import minimize

import math

class Informativeness(Eval):
    """
        Implementation of the The Maximum Entropy Method for Analyzing
        Retrieval Measures by Javed A. Aslam, Emine Yilmaz, Virgiliu Pavlu
    """
    def __init__(self):
        super(Informativeness, self).__init__()
        self.informativeness_parameters = self.get_order_parameters("config/informativeness.json")
        self.run_informativeness_test()


    def eval_all_runs(self):
        #Dict containing each key as RunID and values the score of each measure in the order of self.Q (sorted)
        run_results = {}
        for run in self.runs[:5]:
            res = self.run_trec_eval(self.informativeness_parameters['qrels_file'], run,
                                            self.informativeness_parameters['metric'], queries=True)
            res_num_rel = self.run_trec_eval(self.informativeness_parameters['qrels_file'], run,
                                     'num_rel', queries=True)

            res_num_R_ret = self.run_trec_eval(self.informativeness_parameters['qrels_file'], run,
                                         'num_rel_ret', queries=True)


            for qid, value in res.items():
                if run not in run_results:
                    run_results[run] = {}
                if qid not in run_results[run]:
                    run_results[run][qid] = {"score": 0, "num_rel": 0, "num_rel_ret": 0}
                if qid in res:
                    run_results[run][qid]['score'] = res[qid]
                if qid in res_num_R_ret:
                    run_results[run][qid]['num_rel_ret'] = res_num_R_ret[qid]
                if qid in res_num_rel:
                    run_results[run][qid]['num_rel'] = res_num_rel[qid]

        return run_results

    def entropy(self, pi):
        assert pi > 0, "pi should be bigger than 0"
        assert pi < 1, "pi should be smaller than 1"
        return (-pi*math.log(pi)) - ((1-pi)*math.log(1-pi))

    def objective(self, x):
        H = 0
        for pi in x:
            H += self.entropy(pi)
        return -H

    def constraint1(self, x):
        """ Contraint using the evaluation measure expected value"""
        out_sum = 0
        for i, pi in enumerate(x):
            inner_sum = 0
            for j in range(i-1):
                inner_sum += x[j]
            out_sum += ((pi/(i+1)) * (1+inner_sum))
        return ((1/float(self.num_rel)) * out_sum) - self.score

    def constraint2(self, x):
        """ Constraint given the information of Number of Relevant Retrieved"""
        return np.sum(x) - self.R_ret


    def run_informativeness_test(self):
        self.runs = self.get_list_files(self.informativeness_parameters['runs_folder'])
        run_results = self.eval_all_runs()
        con1 = {'type': 'eq', 'fun': self.constraint1}
        con2 = {'type': 'eq', 'fun': self.constraint2}
        cons = ([con1, con2])
        b = (0.0000001, 0.9999999)

        bounds = ([b]*self.informativeness_parameters['rank_size'])
        for run_id, rankings in run_results.items():
            for qid, values in rankings.items():
                self.num_rel = values['num_rel']
                self.score = values['score']
                self.R_ret = values['num_rel_ret']
                if self.R_ret < 1:
                    continue
                x0 = np.array([0.99]*self.informativeness_parameters['rank_size'])
                print("Optimizing:",run_id, qid, values)
                sol = minimize(self.objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1,
                                                                                                             'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08})
                print(sol)





