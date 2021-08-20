import pprint
import itertools
from scipy.spatial import distance
from src.params import Params
import numpy as np

class PartialOrder(Params):
    def __init__(self, config_file=None, order_parameters = None, is_config=None):
        super(PartialOrder, self).__init__()
        if is_config is not None:
            self.order_parameters = Params.get_order_parameters(self, config_file)
        else:
            self.order_parameters = order_parameters

        self.qrels_file = self.order_parameters['qrels_file']
        if self.order_parameters['from_qrels']:
            self.labels_grade = self.get_label_grades()
            self.multi_aspect_labels_L = self.calc_L_set()
            self.best_label = self.get_best_label()
        else:
            self.best_label = tuple(self.order_parameters["greatest_label"])
            self.min_label = tuple(self.order_parameters["minimum_label"])
            self.labels_grade = self.get_label_grades(from_qrels=False)
            self.multi_aspect_labels_L = self.calc_L_set()


    def get_file_content(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        return content

    def calc_L_set(self):
        """Compute the Cartesian product of L_a"""
        somelists = [ values for x, values in self.labels_grade.items()]
        return [element for element in itertools.product(*somelists)]

    def get_label_grades(self, from_qrels=True):

        if from_qrels:
            """Get all the labels that appeared for each aspect in the qrels"""
            content = self.get_file_content(self.qrels_file)
            self.number_aspects = len(content[0].split()[3:])
            dict_labels_byaspect = {a:[] for a in range(self.number_aspects)}

            for line in content:
                aspect_weights = line.split()[3:]
                [dict_labels_byaspect[key].append(weight) for key, weight in zip(dict_labels_byaspect, aspect_weights) if weight not in dict_labels_byaspect[key]]
            return dict_labels_byaspect
        else:
            """Get all the labels that appeared for each aspect from the given range in distance_order.json file"""
            dict_labels_byaspect = {a: list(range(self.min_label[a], self.best_label[a]+1)) for a in range(self.order_parameters['number_of_aspects'])}
            return dict_labels_byaspect

    def get_best_label(self):
        dict_best_label_by_aspect = {a: -1 for a in range(self.number_aspects)}
        for pair in self.multi_aspect_labels_L:
            for aspect,val in zip(dict_best_label_by_aspect,pair):
                if int(val) > dict_best_label_by_aspect[aspect]:
                    dict_best_label_by_aspect[aspect] = int(val)
        best_label = (dict_best_label_by_aspect[aspect] for aspect in dict_best_label_by_aspect)
        best_label = tuple(best_label)
        return best_label

    def write_new_qrels(self, file_out, dict_new_weights_by_tuple):
        content = self.get_file_content(self.qrels_file)
        new_qrels_file = open(file_out, "w")
        for line in content:
            parts = line.split()
            tuple_weights = tuple(int(x) for x in parts[3:])
            if tuple_weights in dict_new_weights_by_tuple:
                print(parts[0], parts[1], parts[2], dict_new_weights_by_tuple[tuple_weights]['mapping'], file=new_qrels_file)
            else:
                raise Exception("Something went wrong when assigning the new weights to qrels."
                                "Please make sure that the range of the aspects are correct.")

        new_qrels_file.close()

class DistanceOrder(PartialOrder):

    def __init__(self, is_config=None, configuration_file="config/distance_order.json", order_parameters=None):

        """
            Dict of distance functions. To add a new distance metric, include the function on the dictionary below.
        """
        self.dict_distance_functions = {'euclidean': self.euclidean_distance, 'manhattan': self.manhattan_distance,
                                        'chebyshev': self.chebyshev_distance}

        """Dict of mapping functions. To add a new mapping approach, include the function on the dict below."""
        self.dict_mapping_functions = {'iteration': self.iteration_mapping}

        self.dict_distances_by_tuple = {}

        if is_config is not None:
            super(DistanceOrder, self).__init__(config_file=configuration_file, is_config=True)
        else:
            super(DistanceOrder, self).__init__(order_parameters=order_parameters)
            self.distance_order_parameters = order_parameters
        try:
            """Pick from the distance_order.json file the distance function to be used."""
            self.distance_function = self.dict_distance_functions[self.distance_order_parameters['distance_metric']]
        except KeyError:
            raise ValueError("The distance metric {} is a invalid distance measure! \n "
                             "Please use one of the following: {}".format(self.distance_order_parameters['distance_metric'], list(self.dict_distance_functions.keys())))

        try:
            """Pick from the distance_order.json file the mapping function to be used."""
            self.mapping_function = self.dict_mapping_functions[self.distance_order_parameters['mapping']]
        except KeyError:
            raise ValueError("The distance metric {} is a invalid mapping! \n "
                             "Please use one of the following: {}".format(self.distance_order_parameters['mapping'],
                                                                          list(self.dict_mapping_functions.keys())))

        self.set_of_distances = self.compute_distance_grades()
        self.compute_mapping()
        self.write_new_qrels(self.distance_order_parameters['new_qrels_path'], self.dict_distances_by_tuple)

    def compute_mapping(self):
        """ Compute the mapping using the mapping defined in the distance_order.json. """
        dict_of_dist_mapping_to_weight = self.mapping_function(self.set_of_distances)
        for tuple in self.multi_aspect_labels_L:
            dst = self.dict_distances_by_tuple[tuple]['distance']
            if dst in dict_of_dist_mapping_to_weight:
                self.dict_distances_by_tuple[tuple]['mapping'] = dict_of_dist_mapping_to_weight[dst]

    def compute_distance_grades(self):
        """The distance computed here is from each tuple of labels to the best label"""
        set_of_distances = []
        for tuple in self.multi_aspect_labels_L:
            if tuple not in self.dict_distances_by_tuple:
                self.dict_distances_by_tuple[tuple] = {"distance": 0, "mapping": 0}

            # Compute the distance between the tuple and the best label tuple
            dst = self.distance_function(tuple, self.best_label)
            self.dict_distances_by_tuple[tuple]['distance'] = dst
            # print(tuple, dst)
            set_of_distances.append(dst)

        set_of_distances = set(set_of_distances)
        return set_of_distances

    """ Different mapping functions: mapping weights to labels."""
    def iteration_mapping(self, set_of_distances):
        return {dst: i for i, dst in enumerate(sorted(set_of_distances,reverse=True))}

    """Different distance metrics."""
    def manhattan_distance(self, l, l_star):
        return distance.cityblock(l, l_star)

    def euclidean_distance(self, l, l_star):
        return distance.euclidean(l, l_star)

    def chebyshev_distance(self, l, l_star):
        return distance.chebyshev(l, l_star)

class SkylineOrder(PartialOrder):
    def __init__(self, is_config=None, configuration_file="config/skylineorder.json", order_parameters=None):
        if is_config is not None:
            super(SkylineOrder, self).__init__(config_file=configuration_file, is_config=True)
        else:
            super(SkylineOrder, self).__init__(order_parameters=order_parameters)
            self.order_parameters = order_parameters

        self.qrels_file = self.order_parameters['qrels_file']
        """Dict of mapping functions. To add a new mapping approach, include the function on the dict below."""
        self.dict_mapping_functions = {'iteration': self.iteration_mapping}

        try:
            """Pick from the distance_order.json file the mapping function to be used."""
            self.mapping_function = self.dict_mapping_functions[self.order_parameters['mapping']]
        except KeyError:
            raise ValueError("The distance metric {} is a invalid mapping! \n "
                             "Please use one of the following: {}".format(self.order_parameters['mapping'],
                                                                          list(self.dict_mapping_functions.keys())))
        self.dict_levels_by_tuple = {}
        self.compute_skyline_levels()
        self.write_new_qrels(self.order_parameters['new_qrels_path'], self.dict_levels_by_tuple)

    def get_skyline_sets(self, copy_multi_aspect_labels_L):
        """This use the skyline operator idea to compute the skyline set."""
        current_level = [self.best_label]
        skyline_sets = [current_level]
        cond = 0
        while cond == 0:
            new_level = []
            for tupla in current_level:
                for aspect in range(len(tupla)):
                    arr = np.zeros(self.order_parameters['number_of_aspects'],dtype=int)
                    arr[aspect] = -self.order_parameters['aspect_step'][aspect]
                    new_level += [tuple(np.add(tupla, arr))]

            new_level = list(dict.fromkeys(new_level))
            current_level = []
            for val in new_level:
                ignore = 0
                for weight in val:
                    if weight < 0:
                        ignore = 1
                if ignore != 1:
                    current_level += [val]

            if len(current_level) == 0:
                cond = 1
            else:
                skyline_sets += [current_level]
        return skyline_sets


    """ Different mapping functions: mapping weights to labels."""
    def iteration_mapping(self, skyline_sets):
        return {i: set for i, set in enumerate(reversed(skyline_sets))}

    def compute_mapping(self, skyline_sets):
        dict_level_sets = self.mapping_function(skyline_sets)
        for level, set in dict_level_sets.items():
            for tuple in set:
                if tuple not in self.dict_levels_by_tuple:
                    self.dict_levels_by_tuple[tuple] = {"mapping":level}

    def compute_skyline_levels(self):
        copy_multi_aspect_labels_L = self.multi_aspect_labels_L.copy()
        copy_multi_aspect_labels_L.remove(self.best_label)
        copy_multi_aspect_labels_L.remove(self.min_label)
        skyline_sets = self.get_skyline_sets(copy_multi_aspect_labels_L)
        self.compute_mapping(skyline_sets)

class Qrels_urbp(object):
    def __init__(self, qrels, qrels_out):
        self.qrels = qrels
        self.qrels_out = qrels_out
        self.compute_rbp_qrels()

    def compute_rbp_qrels(self):
        with open(self.qrels) as f:
            content = [x.rstrip() for x in f.readlines()]
        file_out = open(self.qrels_out,'w')
        for line in content:
            parts = line.split()
            "1 0 clueweb12-0000wb-88-07607 0 0 0"
            qid = parts[0]
            docid = parts[2]
            assessments = [int(x) for x in parts[3:]]
            score = np.prod(assessments)
            # print(qid, '0', docid, str(score), parts[3:])
            print(qid, '0', docid, str(score), file=file_out)
        # file_out.close()




if __name__ == '__main__':
    distance_order = DistanceOrder()
