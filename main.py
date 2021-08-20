from src.utils.pre_process import PreProcess
from src.utils.prepare_runs_track import RunsTrack
from src.utils.pagerank import PageRank
from src.utils.spam_score import SpamScore
from src.search import Search
from src.partial_order_relation.partial_order import *
from src.statistics import *
from src.utils.qrels import Qrels
from src.eval import Eval
from src.Experiments.discriminative_power import DiscriminativePower
from src.Experiments.informativeness import Informativeness
from src.utils.plot_distributions import Distributions
from src.measurescores import Results
from src.Experiments.correlation_analysis import CorrAnalysis
from src.Experiments.consistency import Consistency
from src.Experiments.intuitiveness import Intuitiveness
from src.Experiments.unanimity import Unanimity
from src.Experiments.pool_downsampling import PoolDownsampling
from scipy.stats import t
from numpy import average, std
from math import sqrt
from trectools import TrecQrel, TrecRun, TrecEval

def main():
    """Pre-processing and generating qrels (adding spam score and pagerank)"""
    # pre_process = PreProcess()
    # pre_process.transform_queries_qrels_new_ids()
    # pagerank = PageRank("data/pagerank.scoreOrder", "data/qrels_webtrack2009/webtrec2009.txt")
    # pagerank = PageRank("data/ClueWeb09-En-PRranked.txt", "data/qrels_webtrack2009/webtrec2009.txt")
    # pagerank.add_pagerank_qrels()
    # spamscore = SpamScore("data/processed_data/qrels_webtrack2014/qrels_with_pr_normalized.txt", "data/waterloo-spam-cw12-decoded/")
    # spamscore = SpamScore("data/processed_data/qrels_webtrack2009/qrels_with_pr_normalized.txt", "data/clueweb09spam.Fusion", collection=True)
    # spamscore.add_spam_score_to_qrels()
    # runs_track = RunsTrack()
    # runs_track.generate_runs_new_ids()

    """Discretizing the values of the qrels"""  
    # qrels = Qrels()

    """Computing the Partial order relation among documents and generating new qrels."""
    # distance_order = DistanceOrder(is_config=True)
    # skyline_order = SkylineOrder(is_config=True)
    # exit()
    """Computing Dataset Statistics"""
    # dataset_statistics = DatasetStatistics()
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2009/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2009/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2010/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2010/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2011/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2011/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2012/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2012/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2013/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2013/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack2014/qrels_discrete_aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/qrels_webtrack2014/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_decision/qrels_decision_3aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/decision/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/processed_data/qrels/qrels_2015_4aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/task2015/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/processed_data/qrels/qrels_2016_4aspects.txt', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/task2016/qrels_urbp.txt')
    qrels = Qrels_urbp(qrels='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_misinfo/qrels/misinfo-qrels', qrels_out='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_weighting_distanceorder/misinfo2020/qrels_urbp.txt')
    

    exit()
    """Generating queries and searching in Indri"""
    # search = Search()
    # search.prepare_indri_queries()
    # search.search_griparameters()

    """Evaluating the runs"""
    # eval = Eval()
    # val = eval.run_trec_eval(qrels="data/processed_data/qrels/qrels_only_rel.txt", run="data/runs/bm25/0.01_0.6.txt", metric='ndcg', queries=True)
    # eval.run_trec_eval(qrels="data/processed_data/qrels/qrels_only_rel.txt", run="data/runs/bm25/0.01_0.6.txt", metric='ndcg')
    #getting the top runs
    # eval.get_top_runs()
    # eval.get_average_scores()

    """Generating results file with evaluation scores"""
    # results = Results()
    # results.generate_results()

    """Running Experiments"""
    ####  Discriminative Power #####
    # discriminative_power = DiscriminativePower()

    ##### Informativeness #####
    # Informativeness()

    """Correlation Analysis"""
    # corr_analysis = CorrAnalysis()

    """Intuitiveness Experiment"""
    # intuitiveness = Intuitiveness()
    # intuitiveness.experiment_intuitiveness()

    """Unanimity Experiment"""
    # intuitiveness = Unanimity()
    # intuitiveness.experiment_unanimity()

    """Pool Downsampling"""
    # pooldownsampling = PoolDownsampling()
    #
    # #### Task Track 2015 #####
    # """Multiple random samples"""
    # # for i in range(1000):
    # #     # Task Track 2015 and Task Track 2016 ##
    # #     # pooldownsampling.generate_downsampling(qrels_path='data/processed_data/qrels/qrels_2015_4aspects.txt',
    # #     #                                    folder_new='data/multiple_downsampling_qrels_2015/'+str(i)+'/', step=15)
    # #     # pooldownsampling.evaluate_downsampling_topicwize(runs_folder='data/runs_track_newids/2015/',
    # #     #                                                  qrels_path_downsamples='data/multiple_downsampling_qrels_2015/'+str(i)+'/',
    # #     #                                                  results_folder='data/multiple_results_downsampling_topicwise_2015/'+str(i)+'/',
    # #     #                                                  metric='ndcg')
    # #
    # #     # Decision Track ##
    # #     pooldownsampling.generate_downsampling(qrels_path='data/qrels_decision/qrels_decision_3aspects.txt', folder_new='data/multiple_downsampling_qrels_decision/'+str(i)+'/', step=15, num_aspects=3)
    # #     pooldownsampling.evaluate_downsampling_topicwize_decision(runs_folder='data/decisionRuns/',
    # #                                            qrels_path_downsamples='data/multiple_downsampling_qrels_decision/'+str(i)+'/',
    # #                                            results_folder='data/multiple_results_downsampling_topicwise_decision/'+str(i)+'/', metric='ndcg')
    # # exit()
    # dict_results = {}
    # dict_to_plot = {}
    # corr = True
    # for i in range(1000):
    #     if corr is not None:
    #         res = pooldownsampling.calc_corr_by_topic(results_folder='data/multiple_results_downsampling_topicwise_2016/'+str(i)+'/')
    #     else:
    #         res = pooldownsampling.calc_rmse_by_topic(
    #             results_folder='data/multiple_results_downsampling_topicwise_2015/' + str(i) + '/')
    #
    #     for key, percentages in res.items():
    #         if key not in dict_results:
    #             dict_results[key] = {}
    #         for percentage, value in percentages.items():
    #             if percentage not in dict_results[key]:
    #                 dict_results[key][percentage] = []
    #             dict_results[key][percentage] += [value]
    # # pprint.pprint(dict_results)
    # # exit()
    # for key, percentages in dict_results.items():
    #     if key not in dict_to_plot:
    #         dict_to_plot[key] = {'mean':[],
    #                              'lb':[],
    #                              'ub':[]}
    #     # print(sorted(percentages))
    #     for percentage, values in sorted(percentages.items()):
    #         if percentage == '0' and corr is None:
    #             dict_to_plot[key]['mean'] += [0]
    #             dict_to_plot[key]['lb'] += [0]
    #             dict_to_plot[key]['ub'] += [0]
    #             continue
    #         elif percentage == '0' and corr:
    #             dict_to_plot[key]['mean'] += [1]
    #             dict_to_plot[key]['lb'] += [1]
    #             dict_to_plot[key]['ub'] += [1]
    #             continue
    #         stddev = std(values, ddof=1)
    #         t_bounds = t.interval(0.95, len(values) - 1)
    #         ci = [average(values) + critval * stddev / sqrt(len(values)) for critval in t_bounds]
    #         dict_to_plot[key]['mean'] += [average(values)]
    #         dict_to_plot[key]['lb'] += [ci[0]]
    #         dict_to_plot[key]['ub'] += [ci[1]]
    # # pprint.pprint(dict_to_plot)
    # # exit()
    # # pooldownsampling.plot_mean_and_CI(dict_to_plot, plot_path='rmse_downsampling_2016.jpeg', x_labels=sorted(list(dict_results[list(dict_results.keys())[0]].keys())))
    # pooldownsampling.plot_mean_and_CI(dict_to_plot, plot_path='correlation_downsampling_2016.jpeg', x_labels=sorted(list(dict_results[list(dict_results.keys())[0]].keys())), corr=corr)

    # print("After ..")
    # pprint.pprint(dict_results)

    # pooldownsampling.generate_downsampling(qrels_path='data/processed_data/qrels/qrels_2015_4aspects.txt', folder_new='data/downsample_qrels_2015/',step=15)
    # pooldownsampling.evaluate_downsampling(runs_folder='data/runs_track_newids/2015/',qrels_path_downsamples='data/downsample_qrels_2015/',results_folder='data/results_downsampling_2015/' ,metric='ndcg')
    # pooldownsampling.evaluate_downsampling_topicwize(runs_folder='data/runs_track_newids/2015/',qrels_path_downsamples='data/downsample_qrels_2015/',results_folder='data/results_downsampling_topicwise_2015/' ,metric='ndcg')
    # pooldownsampling.evaluate_downsampling_topicwize(runs_folder='data/runs_track_newids/2015/',qrels_path_downsamples='data/downsample_qrels_2015/',results_folder='data/results_downsampling_topicwise_2015/' ,metric='ndcg')
    # pooldownsampling.plot_results_summary("data/results_downsampling_2015/")

    #### Task Track 2016 #####
    # pooldownsampling.generate_downsampling(qrels_path='data/processed_data/qrels/qrels_2016_4aspects.txt', folder_new='data/downsample_qrels_2016/',step=15)
    # pooldownsampling.generate_downsampling(qrels_path='data/processed_data/qrels/qrels_2016_4aspects.txt', folder_new='data/downsample_qrels_2016/',step=15)
    # pooldownsampling.evaluate_downsampling(runs_folder='data/runs_track_newids/2016/',
    #                                        qrels_path_downsamples='data/downsample_qrels_2016/',
    #                                        results_folder='data/results_downsampling_2016/', metric='ndcg')
    # pooldownsampling.evaluate_downsampling_topicwize(runs_folder='data/runs_track_newids/2016/',
    #                                        qrels_path_downsamples='data/downsample_qrels_2016/',
    #                                        results_folder='data/results_downsampling_topicwise_2016/', metric='ndcg')
    # pooldownsampling.plot_results_summary("data/results_downsampling_2016/")
    # pooldownsampling.calc_corr_by_topic(results_folder='data/results_downsampling_topicwise_2016/')
    # pooldownsampling.calc_rmse_by_topic(results_folder='data/results_downsampling_topicwise_2016/')
    # pooldownsampling.calc_rmse_by_topic(results_folder='data/results_downsampling_topicwise_2015/')

    #### Decision TRACK #####
    # pooldownsampling.generate_downsampling(qrels_path='data/qrels_decision/qrels_decision_3aspects.txt', folder_new='data/downsample_qrels_decision/',step=15, num_aspects=3)
    # pooldownsampling.evaluate_downsampling_decision(runs_folder='data/decisionRuns/',
    #                                        qrels_path_downsamples='data/downsample_qrels_decision/',
    #                                        results_folder='data/results_downsampling_decision/', metric='ndcg')
    # pooldownsampling.evaluate_downsampling_topicwize_decision(runs_folder='data/decisionRuns/',
    #                                        qrels_path_downsamples='data/downsample_qrels_decision/',
    #                                        results_folder='data/results_downsampling_decision_topicwise/', metric='ndcg')
    # pooldownsampling.plot_results_summary("data/results_downsampling_decision/")
    #
    # pooldownsampling.calc_corr_by_topic(results_folder='data/results_downsampling_decision_topicwise/')
    # pooldownsampling.calc_rmse_by_topic(results_folder='data/results_downsampling_decision_topicwise/')


    ### Ploting distributions ###
    # distributions = Distributions()
    # distributions.get_frequency_by_rank_pos()
    # distributions.plot_num_relevant_per_query()

    ## COMPUTING LIMITATIONS ##

    # for year in [2012]:
    #     # dt.plot_histogram_qrels()
    #     dt.generate_individualIdealRankings(qrels_path='data/qrels_webtrack'+str(year)+'/qrels_discrete_aspects.txt',outpath='data/run_test_limitations_webtrack'+str(year)+'/', num_aspects=3)
    #     dt.generate_idealRuns_Single(qrels_path='data/qrels_webtrack'+str(year)+'/qrels_webtrack'+str(year)+'_manhattan.txt',
    #                                  out_path='data/run_test_limitations_webtrack'+str(year)+'/manhattan.txt', aspects=[0,1,2])
    #     dt.generate_idealRuns_Single(qrels_path='data/qrels_webtrack'+str(year)+'/qrels_webtrack'+str(year)+'_euclidean.txt',
    #                                  out_path='data/run_test_limitations_webtrack'+str(year)+'/euclidean.txt', aspects=[0, 1, 2])
    #     dt.generate_idealRuns_Single(qrels_path='data/qrels_webtrack'+str(year)+'/qrels_webtrack'+str(year)+'_manhattan.txt',
    #                                  out_path='data/run_test_limitations_webtrack'+str(year)+'/chebyshev.txt', aspects=[0, 1, 2])
    # dt = DatasetStatistics()
    #
    # dt.generate_individualIdealRankings(qrels_path="data/qrels_decision/qrels_decision_3aspects.txt",
    #                                     outpath='data/runs_test_limitations_decision/', num_aspects=3)
    # dt.generate_idealRuns_Single(
    #     qrels_path="data/qrels_misinfo/qrels/qrels_misinfo_chebyshev.txt",
    #     out_path='data/run_test_limitations_misinformation/individual/chebyshev.txt', aspects=[0, 1, 2])
    # dt.generate_idealRuns_Single(
    #     qrels_path="data/qrels_misinfo/qrels/qrels_misinfo_manhattan.txt",
    #     out_path='data/run_test_limitations_misinformation/individual/manhattan.txt', aspects=[0, 1, 2])
    # dt.generate_idealRuns_Single(
    #     qrels_path="data/qrels_misinfo/qrels/qrels_misinfo_euclidean.txt",
    #     out_path='data/run_test_limitations_misinformation/individual/euclidean.txt', aspects=[0, 1, 2])
    # for year in ['2009','2010','2011','2012','2013','2014']:
    #     for metric in ['ndcg','map']:
    #         dt = DatasetStatistics()
    #         dict_parameters = {'qrels_arr': [
    #                 "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack"+str(year)+"/qrels_only_relevance.txt",
    #                 "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack"+str(year)+"/qrels_only_pagerank.txt",
    #                 "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_webtrack"+str(year)+"/qrels_only_spam.txt",
    #                 ],
    #                'runs_folder': '/science/image/cluster-homedirs/krn788/LLMS_multieval/data/run_test_limitations_webtrack'+year+'/individual/',
    #                 'task':'Webtrack'+year,
    #                 'metric': metric,
    #                # 'topic_range': [36, 85],
    #                # 'single_runs': {'chebyshev': ("data/processed_data/qrels_chebyshev_iteration_qrels_only_query.txt",
    #                #                               "data/runs_test_limitations_decision/chebyshev.txt"),
    #                #                 'euclidean': ("data/processed_data/qrels_euclidean_iteration_qrels_only_query.txt",
    #                #                               "data/runs_test_limitations_decision/euclidean.txt"),
    #                #                 'manhattan': ("data/processed_data/qrels_manhattan_iteration_qrels_only_query.txt",
    #                #                               "data/runs_test_limitations_decision/manhattan.txt")}
    #         }
    #
    #         dt.plot_scores_by_topic(dict_parameters)
    # exit()
    for metric in ['ndcg', 'map']:
        dt = DatasetStatistics()
        dict_parameters = {'qrels_arr': [
            "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_decision/qrels_decision_rel.txt",
            "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_decision/qrels_decision_cred.txt",
            "/science/image/cluster-homedirs/krn788/LLMS_multieval/data/qrels_decision/qrels_decision_correctness.txt",
        ],
            'runs_folder': '/science/image/cluster-homedirs/krn788/LLMS_multieval/data/runs_test_limitations_decision/individual/',
            'task': 'Decision',
            'metric': metric,
        }
        dt.plot_scores_by_topic(dict_parameters)


    # dt.create_box_plot_format_csv("data/results/WebTrack2009/", "data/results_to_boxplot/WebTrack2009/")


def end_to_end_process():
    """Generating results file with evaluation scores"""
    # for track in [2009, 2010, 2011, 2012, 2013, 2014]:
    parameters_results = {
        'qrels_files': [
            # ('chebyshev', "data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query_2016.txt"),
            # ('euclidean', 'data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query_2016.txt'),
            # ('manhattan', 'data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query_2016.txt'),

            # ('chebyshev', "data/qrels_decision/qrels_decision_chebyshev.txt"),
            # ('euclidean', 'data/qrels_decision/qrels_decision_euclidean.txt'),
            # ('manhattan', 'data/qrels_decision/qrels_decision_manhattan.txt'),

            ('chebyshev', "data/qrels_misinfo/qrels/qrels_misinfo_chebyshev.txt"),
            ('euclidean', 'data/qrels_misinfo/qrels/qrels_misinfo_euclidean.txt'),
            ('manhattan', 'data/qrels_misinfo/qrels/qrels_misinfo_manhattan.txt'),

            # ('chebyshev', "data/qrels_webtrack2009/qrels_webtrack2009_chebyshev.txt"),
            # ('euclidean', 'data/qrels_webtrack2009/qrels_webtrack2009_euclidean.txt'),
            # ('manhattan', 'data/qrels_webtrack2009/qrels_webtrack2009_manhattan.txt'),
            # ('chebyshev', "data/qrels_webtrack"+str(track)+"/qrels_webtrack"+str(track)+"_chebyshev.txt"),
            # ('euclidean', "data/qrels_webtrack"+str(track)+"/qrels_webtrack"+str(track)+"_euclidean.txt"),
            # ('manhattan', "data/qrels_webtrack"+str(track)+"/qrels_webtrack"+str(track)+"_manhattan.txt"),
            # ('rel', 'data/processed_data/qrels/qrels_only_rel_2016.txt'),
            # ('use', 'data/processed_data/qrels/qrels_only_usefulness_2016.txt'),
            # ('pop', 'data/processed_data/qrels/qrels_only_popularity_2016.txt'),
            # ('spam', 'data/processed_data/qrels/qrels_only_credibility_2016.txt')

            # ('rel', 'data/qrels_decision/qrels_decision_rel.txt'),
            # ('cred', 'data/qrels_decision/qrels_decision_cred.txt'),
            # ('correc', 'data/qrels_decision/qrels_decision_correctness.txt'),

            ('rel', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_first'),
            ('cred', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_second'),
            ('correc', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_third'),

            #
            # ('rel', "data/qrels_webtrack"+str(track)+"/qrels_only_relevance.txt"),
            # ('cred', "data/qrels_webtrack"+str(track)+"/qrels_only_pagerank.txt"),
            # ('correc', "data/qrels_webtrack"+str(track)+"/qrels_only_spam.txt"),

        ],
        'runs_folder': [
            # ('Decision', 'data/decisionRuns/'),
            # ('Misinfo-adhoc', 'data/misinfo-runs_extracted/adhoc/'),
            ('Misinfo-recall', 'data/misinfo-runs_extracted/adhoc/'),
            # ('WebTrack2014', 'data/webtrack2014Runs_extracted/'),
            # ("WebTrack"+str(track), "data/webtrack"+str(track)+"Runs_extracted/"),
            # ('TaskTrack2016', 'data/runs_track_newids/2016/'),
            # ('TaskTrack2016', 'data/runs_track_newids/2016/'),
        ],
        # 'results_path': 'data/results_cutoff100/WebTrack'+str(track)+"/",
        # 'results_path': 'data/results/Misinfo-recall/',
        'results_path': 'data/results/cutoff_100/Misinfo-adhoc_ap/',
        # 'results_path': 'data/results/Misinfo-adhoc_ap/',
        # 'results_path': 'data/results/Misinfo-adhoc_ap/',
        # 'results_path': 'data/results_cutoff100_ap/TaskTrack2016/',
        'aspects': ['rel', 'cred', 'correc'],
        # 'aspects': ['rel', 'use', 'pop', 'spam'],
        # 'metrics': 'map',
        # 'metrics': 'map_cut.5',
        'metrics': 'map_cut.100',
        # 'metrics': 'ndcg_cut.5',
        }
    # results = Results(multiple_parameters=parameters_results)
    # results.generate_results()
    # exit()

    # adhoc-map-100



    """ Correlation Analysis """
    # corr_analysis = CorrAnalysis(results_folder='data/results_ap/WebTrack2009/', trec='web')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Decision/', trec='decision')
    # corr_analysis = CorrAnalysis(results_folder='data/results_ap/TaskTrack2016/')
    # print('Misinfo-adhoc_ap')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-adhoc_ap/', trec='decision')
    # print('Misinfo-adhoc')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-adhoc/', trec='decision')
    # print('Misinfo-recall')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-recall/', trec='decision')
    # print('Misinfo-recall_ap')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-recall_ap/', trec='decision')
    # exit()

    """Consistency"""
    # consistency_analysis = Consistency(results_folder='data/results/Misinfo-adhoc/', trec='decision', b = 10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2009/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2010/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2011/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2012/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2013/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2014/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/TaskTrack2015/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/TaskTrack2016/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/Decision/', trec='decision', b=10000)
    #
    # consistency_analysis = Consistency(results_folder='data/results_ap/Misinfo-adhoc_ap/', trec='decision', b = 10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2009/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2010/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2011/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2012/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2013/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2014/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/TaskTrack2015/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/TaskTrack2016/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/Decision/', trec='decision', b=10000)


    """ Discriminative Power """
    multiple_parameters = {
        'qrels_files': [
            ('chebyshev', "data/qrels_misinfo/qrels/qrels_misinfo_chebyshev.txt"),
            ('euclidean', 'data/qrels_misinfo/qrels/qrels_misinfo_euclidean.txt'),
            ('manhattan', 'data/qrels_misinfo/qrels/qrels_misinfo_manhattan.txt'),
        ],
        'runs_folder': [
            ('Misinfo-recall', 'data/misinfo-runs_extracted/adhoc/'),
        ],
        'metrics': ['map'],
        'B': [10000],
        'file_out': 'data/dataset_statistics/discriminative_power_Misinfo-adhoc_10000_map.txt'
    }
    multiple_parameters_baseline = {
        'qrels_files': [
            ('rel', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_first'),
            ('cred', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_second'),
            ('correc', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_third'),
        ],
        'runs_folder': [
            ('Misinfo-recall', 'data/misinfo-runs_extracted/adhoc/'),
        ],
        'metrics': 'map',
        'B': 10000,
    }

    # discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters, multiple_parameters_baseline=multiple_parameters_baseline)


def end_to_end_process_weighting():
    """Generating results file with evaluation scores"""
    # for track in [2009, 2010, 2011, 2012, 2013, 2014]:

    track = 'decision'
    parameters_results = {
        'qrels_files': [
            ('chebyshev', "data/qrels_decision/qrels_decision_chebyshev.txt"),
            ('euclidean', 'data/qrels_decision/qrels_decision_euclidean.txt'),
            ('manhattan', 'data/qrels_decision/qrels_decision_manhattan.txt'),
            ('skyline', 'data/qrels_weighting_distanceorder/'+track+'/skyline.txt'),
            ('urbp', 'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
            ('rel', 'data/qrels_decision/qrels_decision_rel.txt'),
            ('cred', 'data/qrels_decision/qrels_decision_cred.txt'),
            ('correc', 'data/qrels_decision/qrels_decision_correctness.txt'),
            ('double_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_chebyshev.txt'),
            ('double_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_euclidean.txt'),
            ('double_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_manhattan.txt'),
            ('double_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_chebyshev.txt'),
            ('double_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_euclidean.txt'),
            ('double_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_manhattan.txt'),
            ('equispaced_samerange_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_chebyshev.txt'),
            ('equispaced_samerange_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_euclidean.txt'),
            ('equispaced_samerange_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_manhattan.txt'),
            ('equispaced_samerange_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_chebyshev.txt'),
            ('equispaced_samerange_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_euclidean.txt'),
            ('equispaced_samerange_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_manhattan.txt'),

        ],
        'runs_folder': [
            ('Decision', 'data/decisionRuns/'),
        ],

        # 'results_path': 'data/results_weighting/Decision/',
        # 'results_path': 'data/results_weighting_cutoff_5_ap/Decision/',
        'results_path': 'data/results_weighting_cutoff100/Decision/',

        'aspects': ['rel', 'cred', 'correc'],
        # 'aspects': ['rel', 'use', 'pop', 'spam'],
        # 'metrics': 'map',
        # 'metrics': 'map_cut.5',
        # 'metrics': 'map_cut.100',
        # 'metrics': 'ndcg_cut.1000',
        'metrics': 'ndcg_cut.100',
        }
    # exit()
    results = Results(multiple_parameters=parameters_results)
    results.generate_results()

    track = 'misinfo2020'
    parameters_results = {
        'qrels_files': [

            ('chebyshev', "data/qrels_misinfo/qrels/qrels_misinfo_chebyshev.txt"),
            ('euclidean', 'data/qrels_misinfo/qrels/qrels_misinfo_euclidean.txt'),
            ('manhattan', 'data/qrels_misinfo/qrels/qrels_misinfo_manhattan.txt'),
            ('skyline', 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
            ('urbp', 'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),

            ('rel', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_first'),
            ('cred', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_second'),
            ('correc', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_third'),

            ('double_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_chebyshev.txt'),
            ('double_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_euclidean.txt'),
            ('double_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_manhattan.txt'),
            ('double_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_chebyshev.txt'),
            ('double_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_euclidean.txt'),
            ('double_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_manhattan.txt'),
            ('equispaced_samerange_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_chebyshev.txt'),
            ('equispaced_samerange_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_euclidean.txt'),
            ('equispaced_samerange_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_manhattan.txt'),
            ('equispaced_samerange_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_chebyshev.txt'),
            ('equispaced_samerange_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_euclidean.txt'),
            ('equispaced_samerange_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_manhattan.txt'),

        ],
        'runs_folder': [
            ('Misinfo-adhoc', 'data/misinfo-runs_extracted/adhoc/'),
        ],

        # 'results_path': 'data/results_weighting/Misinfo-adhoc/',
        'results_path': 'data/results_weighting_cutoff100/Misinfo-adhoc/',

        'aspects': ['rel', 'cred', 'correc'],
        # 'aspects': ['rel', 'use', 'pop', 'spam'],
        # 'metrics': 'map',
        # 'metrics': 'map_cut.5',
        # 'metrics': 'ndcg_cut.1000',
        # 'metrics': 'map_cut.100',
        # 'metrics': 'ndcg_cut.5',
        'metrics': 'ndcg_cut.100',
        }
    results = Results(multiple_parameters=parameters_results)
    results.generate_results()

    for year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        track = 'qrels_webtrack'+year
        year = year
        parameters_results = {
            'qrels_files': [

                ('chebyshev', "data/qrels_webtrack"+year+"/qrels_webtrack"+year+"_chebyshev.txt"),
                ('euclidean', 'data/qrels_webtrack'+year+'/qrels_webtrack'+year+'_euclidean.txt'),
                ('manhattan', 'data/qrels_webtrack'+year+'/qrels_webtrack'+year+'_manhattan.txt'),
                ('skyline',
                 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
                ('urbp',
                 'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),

                ('rel', "data/qrels_webtrack"+str(year)+"/qrels_only_relevance.txt"),
                ('cred', "data/qrels_webtrack"+str(year)+"/qrels_only_pagerank.txt"),
                ('correc', "data/qrels_webtrack"+str(year)+"/qrels_only_spam.txt"),

                ('double_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
                ('double_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
                ('double_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
                ('double_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
                ('double_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
                ('double_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
                ('equispaced_samerange_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
                ('equispaced_samerange_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
                ('equispaced_samerange_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
                ('equispaced_samerange_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
                ('equispaced_samerange_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
                ('equispaced_samerange_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),

            ],
            'runs_folder': [
                ("WebTrack"+str(year), "data/webtrack"+str(year)+"Runs_extracted/"),
            ],

            # 'results_path': 'data/results_weighting/WebTrack'+str(year)+"/",
            'results_path': 'data/results_weighting_cutoff100/WebTrack'+str(year)+"/",

            'aspects': ['rel', 'cred', 'correc'],
            # 'aspects': ['rel', 'use', 'pop', 'spam'],
            # 'metrics': 'map',
            # 'metrics': 'map_cut.5',
            # 'metrics': 'map_cut.100',
            # 'metrics': 'ndcg_cut.1000',
            'metrics': 'ndcg_cut.100',
        }
        results = Results(multiple_parameters=parameters_results)
        results.generate_results()
    #
    for year in ['2015']:
        if year == '2015':
            track = 'task2015'
            year = ''
        else:
            track = 'task2016'

        parameters_results = {
            'qrels_files': [

                ('chebyshev', 'data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query'+year+'.txt'),
                ('euclidean', 'data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query'+year+'.txt'),
                ('manhattan', 'data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query'+year+'.txt'),
                ('skyline',
                 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
                ('urbp',
                 'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
                ('rel', 'data/processed_data/qrels/qrels_only_rel'+year+'.txt'),
                ('use', 'data/processed_data/qrels/qrels_only_usefulness'+year+'.txt'),
                ('pop', 'data/processed_data/qrels/qrels_only_popularity'+year+'.txt'),
                ('spam', 'data/processed_data/qrels/qrels_only_credibility'+year+'.txt'),

                ('double_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
                ('double_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
                ('double_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
                ('double_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
                ('double_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
                ('double_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
                ('double_aspect-3_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_chebyshev.txt'),
                ('double_aspect-3_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_euclidean.txt'),
                ('double_aspect-3_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_manhattan.txt'),
                ('equispaced_samerange_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
                ('equispaced_samerange_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
                ('equispaced_samerange_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
                ('equispaced_samerange_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
                ('equispaced_samerange_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
                ('equispaced_samerange_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),
                ('equispaced_samerange_aspect-3_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_chebyshev.txt'),
                ('equispaced_samerange_aspect-3_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_euclidean.txt'),
                ('equispaced_samerange_aspect-3_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_manhattan.txt'),

            ],
            'runs_folder': [
                ('TaskTrack2015', 'data/runs_track_newids/2015/'),
                # ('TaskTrack'+year, 'data/runs_track_newids/'+year+'/'),
            ],

            # 'results_path': 'data/results_weighting/TaskTrack2015/',
            'results_path': 'data/results_weighting_cutoff100/TaskTrack2015/',
            # 'results_path': 'data/results_weighting/TaskTrack'+year+'/',

            # 'aspects': ['rel', 'cred', 'correc'],
            'aspects': ['rel', 'use', 'pop', 'spam'],
            # 'metrics': 'map',
            # 'metrics': 'map_cut.5',
            # 'metrics': 'map_cut.100',
            # 'metrics': 'ndcg_cut.1000',
            'metrics': 'ndcg_cut.100',
        }


        results = Results(multiple_parameters=parameters_results)
        results.generate_results()

    for year in ['2016']:
        if year == '2015':
            track = 'task2015'
            year = ''
        else:
            track = 'task2016'

        parameters_results = {
            'qrels_files': [

                ('chebyshev',
                 'data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query_' + year + '.txt'),
                ('euclidean',
                 'data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query_' + year + '.txt'),
                ('manhattan',
                 'data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query_' + year + '.txt'),
                ('skyline',
                 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
                ('urbp',
                 'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
                ('rel', 'data/processed_data/qrels/qrels_only_rel_' + year + '.txt'),
                ('use', 'data/processed_data/qrels/qrels_only_usefulness_' + year + '.txt'),
                ('pop', 'data/processed_data/qrels/qrels_only_popularity_' + year + '.txt'),
                ('spam', 'data/processed_data/qrels/qrels_only_credibility_' + year + '.txt'),

                ('double_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
                ('double_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
                ('double_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
                ('double_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
                ('double_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
                ('double_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
                ('double_aspect-3_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_chebyshev.txt'),
                ('double_aspect-3_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_euclidean.txt'),
                ('double_aspect-3_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_manhattan.txt'),
                ('equispaced_samerange_aspect-1_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
                ('equispaced_samerange_aspect-1_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
                ('equispaced_samerange_aspect-1_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
                ('equispaced_samerange_aspect-2_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
                ('equispaced_samerange_aspect-2_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
                ('equispaced_samerange_aspect-2_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),
                ('equispaced_samerange_aspect-3_chebyshev',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_chebyshev.txt'),
                ('equispaced_samerange_aspect-3_euclidean',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_euclidean.txt'),
                ('equispaced_samerange_aspect-3_manhattan',
                 'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_manhattan.txt'),

            ],
            'runs_folder': [
                ('TaskTrack2016', 'data/runs_track_newids/2016/'),
                # ('TaskTrack'+year, 'data/runs_track_newids/'+year+'/'),
            ],

            # 'results_path': 'data/results_weighting/TaskTrack2016/',
            'results_path': 'data/results_weighting_cutoff100/TaskTrack2016/',
            # 'results_path': 'data/results_weighting/TaskTrack'+year+'/',

            # 'aspects': ['rel', 'cred', 'correc'],
            'aspects': ['rel', 'use', 'pop', 'spam'],
            # 'metrics': 'map',
            # 'metrics': 'map_cut.5',
            # 'metrics': 'ndcg_cut.1000',
            # 'metrics': 'map_cut.100',
            # 'metrics': 'ndcg_cut.5',
            'metrics': 'ndcg_cut.100',
        }
        results = Results(multiple_parameters=parameters_results)
        results.generate_results()
    exit()

    """ Correlation Analysis """
    # corr_analysis = CorrAnalysis(results_folder='data/results_ap/WebTrack2009/', trec='web')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Decision/', trec='decision')
    # corr_analysis = CorrAnalysis(results_folder='data/results_ap/TaskTrack2016/')
    # print('Misinfo-adhoc_ap')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-adhoc_ap/', trec='decision')
    # print('Misinfo-adhoc')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-adhoc/', trec='decision')
    # print('Misinfo-recall')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-recall/', trec='decision')
    # print('Misinfo-recall_ap')
    # corr_analysis = CorrAnalysis(results_folder='data/results/Misinfo-recall_ap/', trec='decision')
    # exit()

    """Consistency"""
    # consistency_analysis = Consistency(results_folder='data/results/Misinfo-adhoc/', trec='decision', b = 10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2009/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2010/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2011/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2012/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2013/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/WebTrack2014/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/TaskTrack2015/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/TaskTrack2016/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results/Decision/', trec='decision', b=10000)
    #
    # consistency_analysis = Consistency(results_folder='data/results_ap/Misinfo-adhoc_ap/', trec='decision', b = 10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2009/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2010/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2011/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2012/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2013/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/WebTrack2014/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/TaskTrack2015/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/TaskTrack2016/', trec='decision', b=10000)
    # consistency_analysis = Consistency(results_folder='data/results_ap/Decision/', trec='decision', b=10000)

    # """ Discriminative Power """

def discriminative_weighting():
    """ Discriminative Power """
    # for year in ['2016']:
    #     if year == '2015':
    #         track = 'task2015'
    #         year = ''
    #     else:
    #         track = 'task2016'
    #
    #     multiple_parameters = {
    #         'qrels_files': [
    #
    #             # ('chebyshev', 'data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query_'+year+'.txt'),
    #             # ('euclidean', 'data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query_'+year+'.txt'),
    #             # ('manhattan', 'data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query_'+year+'.txt'),
    #             # ('skyline',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
    #             ('urbp',
    #              'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
    #
    #             # ('double_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
    #             # ('double_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
    #             # ('double_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
    #             # ('double_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
    #             # ('double_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
    #             # ('double_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
    #             # ('double_aspect-3_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_chebyshev.txt'),
    #             # ('double_aspect-3_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_euclidean.txt'),
    #             # ('double_aspect-3_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-3_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-3_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-3_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_manhattan.txt'),
    #
    #
    #         ],
    #         'runs_folder': [
    #             # ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #             ('TaskTrack2016', 'data/runs_track_newids/2016/'),
    #         ],
    #         'metrics': ['ndcg'],
    #         'B': [10000],
    #         # 'file_out': "data/dataset_statistics/TaskTrack2015_10000_ndcg_weighting.txt",
    #         # 'file_out': "data/dataset_statistics/TaskTrack2016_10000_ndcg_weighting.txt",
    #         'file_out': "data/dataset_statistics/TaskTrack2016_10000_ndcg_weighting_URBP.txt",
    #     }
    #     multiple_parameters_baseline = {
    #         'qrels_files': [
    #             ('rel', 'data/processed_data/qrels/qrels_only_rel_'+year+'.txt'),
    #             ('use', 'data/processed_data/qrels/qrels_only_usefulness_'+year+'.txt'),
    #             ('pop', 'data/processed_data/qrels/qrels_only_popularity_'+year+'.txt'),
    #             ('spam', 'data/processed_data/qrels/qrels_only_credibility_'+year+'.txt'),
    #         ],
    #         'runs_folder': [
    #             # ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #             ('TaskTrack2016', 'data/runs_track_newids/2016/'),
    #         ],
    #         'metrics': 'ndcg',
    #         'B': 10000,
    #     }
    #     discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters,
    #                                                multiple_parameters_baseline=multiple_parameters_baseline)
    # exit()
    # for year in ['2015']:
    #     if year == '2015':
    #         track = 'task2015'
    #         year = ''
    #     else:
    #         track = 'task2016'
    #
    #     multiple_parameters = {
    #         'qrels_files': [
    #
    #             # ('chebyshev', 'data/processed_data/mapped_qrels/qrels_chebyshev_iteration_qrels_only_query'+year+'.txt'),
    #             # ('euclidean', 'data/processed_data/mapped_qrels/qrels_euclidean_iteration_qrels_only_query'+year+'.txt'),
    #             # ('manhattan', 'data/processed_data/mapped_qrels/qrels_manhattan_iteration_qrels_only_query'+year+'.txt'),
    #             # ('skyline',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
    #             ('urbp',
    #              'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
    #
    #             # ('double_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
    #             # ('double_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
    #             # ('double_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
    #             # ('double_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
    #             # ('double_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
    #             # ('double_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
    #             # ('double_aspect-3_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_chebyshev.txt'),
    #             # ('double_aspect-3_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_euclidean.txt'),
    #             # ('double_aspect-3_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-3_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-3_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-3_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-3_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-3_manhattan.txt'),
    #
    #
    #         ],
    #         'runs_folder': [
    #             ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #             # ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #         ],
    #         'metrics': ['ndcg'],
    #         'B': [10000],
    #         # 'file_out': "data/dataset_statistics/TaskTrack2015_10000_ndcg_weighting.txt",
    #         'file_out': "data/dataset_statistics/TaskTrack2015_10000_ndcg_weighting_URBP.txt",
    #         # 'file_out': 'data/dataset_statistics/TaskTrack2016_10000_ndcg_weighting.txt",
    #     }
    #     multiple_parameters_baseline = {
    #         'qrels_files': [
    #             ('rel', 'data/processed_data/qrels/qrels_only_rel'+year+'.txt'),
    #             ('use', 'data/processed_data/qrels/qrels_only_usefulness'+year+'.txt'),
    #             ('pop', 'data/processed_data/qrels/qrels_only_popularity'+year+'.txt'),
    #             ('spam', 'data/processed_data/qrels/qrels_only_credibility'+year+'.txt'),
    #         ],
    #         'runs_folder': [
    #             ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #             # ('TaskTrack2015', 'data/runs_track_newids/2015/'),
    #         ],
    #         'metrics': 'ndcg',
    #         'B': 10000,
    #     }
    #     discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters,
    #                                                multiple_parameters_baseline=multiple_parameters_baseline)
    #
    # exit()

    # for year in ['2009', '2010', '2011', '2012', '2013', '2014']:
    # # for year in ['2013']:
    #     track = 'qrels_webtrack'+year
    #     year = year
    #
    #     multiple_parameters = {
    #         'qrels_files': [
    #
    #             # ('chebyshev', "data/qrels_webtrack"+year+"/qrels_webtrack"+year+"_chebyshev.txt"),
    #             # ('euclidean', 'data/qrels_webtrack'+year+'/qrels_webtrack'+year+'_euclidean.txt'),
    #             # ('manhattan', 'data/qrels_webtrack'+year+'/qrels_webtrack'+year+'_manhattan.txt'),
    #             # ('skyline',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
    #             ('urbp',
    #              'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
    #
    #             # ('double_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_chebyshev.txt'),
    #             # ('double_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_euclidean.txt'),
    #             # ('double_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-1_manhattan.txt'),
    #             # ('double_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_chebyshev.txt'),
    #             # ('double_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_euclidean.txt'),
    #             # ('double_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/double_aspect-2_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-1_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-1_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-1_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-1_manhattan.txt'),
    #             # ('equispaced_samerange_aspect-2_chebyshev',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_chebyshev.txt'),
    #             # ('equispaced_samerange_aspect-2_euclidean',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_euclidean.txt'),
    #             # ('equispaced_samerange_aspect-2_manhattan',
    #             #  'data/qrels_weighting_distanceorder/' + track + '/equispaced_samerange_aspect-2_manhattan.txt'),
    #
    #
    #         ],
    #         'runs_folder': [
    #             ("WebTrack"+str(year), "data/webtrack"+str(year)+"Runs_extracted/"),
    #         ],
    #         'metrics': ['ndcg_cut.1000'],
    #         'B': [10000],
    #         'file_out': 'data/dataset_statistics/WebTrack' + str(year) + "_10000_ndcg_weighting.txt",
    #     }
    #     multiple_parameters_baseline = {
    #         'qrels_files': [
    #             ('rel', "data/qrels_webtrack"+str(year)+"/qrels_only_relevance.txt"),
    #             ('cred', "data/qrels_webtrack"+str(year)+"/qrels_only_pagerank.txt"),
    #             ('correc', "data/qrels_webtrack"+str(year)+"/qrels_only_spam.txt"),
    #         ],
    #         'runs_folder': [
    #             ("WebTrack"+str(year), "data/webtrack"+str(year)+"Runs_extracted/"),
    #         ],
    #         'metrics': 'ndcg_cut.1000',
    #         'B': 10000,
    #     }
    #     discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters,
    #                                                multiple_parameters_baseline=multiple_parameters_baseline)
    # exit()

    # track = 'misinfo2020'
    # multiple_parameters = {
    #     'qrels_files': [
    #
    #         # ('chebyshev', "data/qrels_misinfo/qrels/qrels_misinfo_chebyshev.txt"),
    #         # ('euclidean', 'data/qrels_misinfo/qrels/qrels_misinfo_euclidean.txt'),
    #         # ('manhattan', 'data/qrels_misinfo/qrels/qrels_misinfo_manhattan.txt'),
    #         # ('skyline', 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
    #         ('urbp',
    #          'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),
    #         #
    #         # ('double_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_chebyshev.txt'),
    #         # ('double_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_euclidean.txt'),
    #         # ('double_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_manhattan.txt'),
    #         # ('double_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_chebyshev.txt'),
    #         # ('double_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_euclidean.txt'),
    #         # ('double_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_manhattan.txt'),
    #         # ('equispaced_samerange_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_chebyshev.txt'),
    #         # ('equispaced_samerange_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_euclidean.txt'),
    #         # ('equispaced_samerange_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_manhattan.txt'),
    #         # ('equispaced_samerange_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_chebyshev.txt'),
    #         # ('equispaced_samerange_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_euclidean.txt'),
    #         # ('equispaced_samerange_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_manhattan.txt'),
    #
    #     ],
    #     'runs_folder': [
    #         ('Misinfo-recall', 'data/misinfo-runs_extracted/adhoc/'),
    #     ],
    #     'metrics': ['ndcg'],
    #     'B': [10000],
    #     # 'file_out': 'data/dataset_statistics/discriminative_power_Misinfo-adhoc_10000_ndcg_weighting.txt'
    #     'file_out': 'data/dataset_statistics/discriminative_power_Misinfo-adhoc_10000_ndcg_weighting_URBP.txt'
    # }
    # multiple_parameters_baseline = {
    #     'qrels_files': [
    #         ('rel', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_first'),
    #         ('cred', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_second'),
    #         ('correc', 'data/qrels_misinfo/qrels/misinfo-2020-qrels_third'),
    #     ],
    #     'runs_folder': [
    #         ('Misinfo-recall', 'data/misinfo-runs_extracted/adhoc/'),
    #     ],
    #     'metrics': 'ndcg',
    #     'B': 10000,
    # }
    # discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters,
    #                                            multiple_parameters_baseline=multiple_parameters_baseline)

    track = 'decision'
    """ Discriminative Power """
    multiple_parameters = {
        'qrels_files': [

            # ('chebyshev', "data/qrels_decision/qrels_decision_chebyshev.txt"),
            # ('euclidean', 'data/qrels_decision/qrels_decision_euclidean.txt'),
            # ('manhattan', 'data/qrels_decision/qrels_decision_manhattan.txt'),
            # ('skyline', 'data/qrels_weighting_distanceorder/' + track + '/skyline.txt'),
            ('urbp',
             'data/qrels_weighting_distanceorder/' + track + '/qrels_urbp.txt'),

            # ('double_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_chebyshev.txt'),
            # ('double_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_euclidean.txt'),
            # ('double_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-1_manhattan.txt'),
            # ('double_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_chebyshev.txt'),
            # ('double_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_euclidean.txt'),
            # ('double_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/double_aspect-2_manhattan.txt'),
            # ('equispaced_samerange_aspect-1_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_chebyshev.txt'),
            # ('equispaced_samerange_aspect-1_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_euclidean.txt'),
            # ('equispaced_samerange_aspect-1_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-1_manhattan.txt'),
            # ('equispaced_samerange_aspect-2_chebyshev', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_chebyshev.txt'),
            # ('equispaced_samerange_aspect-2_euclidean', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_euclidean.txt'),
            # ('equispaced_samerange_aspect-2_manhattan', 'data/qrels_weighting_distanceorder/'+track+'/equispaced_samerange_aspect-2_manhattan.txt'),


        ],
        'runs_folder': [
            ('Decision2019', 'data/decisionRuns/'),
        ],
        'metrics': ['ndcg'],
        'B': [10000],
        'file_out': 'data/dataset_statistics/discriminative_power_Decision2019_10000_ndcg_weighting_URBP.txt'
        # 'file_out': 'data/dataset_statistics/discriminative_power_Decision2019_10000_ndcg_weighting.txt'
    }
    multiple_parameters_baseline = {
        'qrels_files': [
            ('rel', 'data/qrels_decision/qrels_decision_rel.txt'),
            ('cred', 'data/qrels_decision/qrels_decision_cred.txt'),
            ('correc', 'data/qrels_decision/qrels_decision_correctness.txt'),
        ],
        'runs_folder': [
            ('Decision2019', 'data/decisionRuns/'),
        ],
        'metrics': 'ndcg',
        'B': 10000,
    }
    discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters,
                                               multiple_parameters_baseline=multiple_parameters_baseline)






    # discriminative_power = DiscriminativePower(multiple_parameters=multiple_parameters, multiple_parameters_baseline=multiple_parameters_baseline)

def correlation_weighting():
    """ Correlation Analysis """
    print('WebTrack2009')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2009/', trec='web')
    print()
    print('WebTrack2010')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2010/', trec='web')
    print()
    print('WebTrack2011')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2011/', trec='web')
    print()
    print('WebTrack2012')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2012/', trec='web')
    print()
    print('WebTrack2013')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2013/', trec='web')
    print()
    print('WebTrack2014')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/WebTrack2014/', trec='web')
    print()
    print('Decision')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/Decision/', trec='decision')
    print()
    print('Misinfo-adhoc')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/Misinfo-adhoc/', trec='decision')
    print()
    print('TaskTrack2015')
    corr_analysis = CorrAnalysis(results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/TaskTrack2015/')
    print()
    print('TaskTrack2016')
    corr_analysis = CorrAnalysis(
        results_folder='/science/image/cluster-homedirs/krn788/LLMS_multieval/data/results_weighting/TaskTrack2016/')


if __name__ == '__main__':
    # main()
    # end_to_end_process()
    end_to_end_process_weighting()
    # discriminative_weighting()
    # correlation_weighting()