import json
import pprint

from TransformerModel.util import plot_bar

# use the criteria defined in freqAnalysis
from freqAnalysis import find_most_freq_paths
import criteria


def predict_efficiency_given_paths(predictors, test_set, is_positive, is_negative):
    """
    Evaluate topologies in the test set using the set of predictors.
    Concretely, given the first n paths (predictors), consider the topologies in the test set that contain any of these
    predictors. How many of them are positive / negative?
    The 'first n paths' here can be either good ones or bad ones.

    :return: pos_counts[n]: number of topos that contain any of the first n predictors and are *positive*
        neg_counts[n]: number of topos that contain any of the first n predictors and are *negative*
    """
    size_of_predictors = len(predictors)

    # a helper to decide if any of the first `n` predictors occurs in `paths`
    first_n_occurred = lambda paths, n: any(path in paths for path in predictors[:n])

    pos_counts = []
    neg_counts = []

    for n in range(1, size_of_predictors + 1):
        pos_count = 0
        neg_count = 0

        for datum in test_set.values():
            paths = datum['paths']
            eff = datum['eff']
            vout = datum['vout']

            if first_n_occurred(paths, n):
                if is_positive(eff, vout):
                    pos_count += 1
                elif is_negative(eff, vout):
                    neg_count += 1
                # there are neither pos nor neg cases, ignore them

        pos_counts.append(pos_count)
        neg_counts.append(neg_count)
        #pos_counts.append(1. * pos_count / (pos_count + neg_count))
        #neg_counts.append(1. * neg_count / (pos_count + neg_count))

    return pos_counts, neg_counts


def predict_efficiency_using_first_n(training_data, test_data, is_positive, is_negative):
    """
    See if we can predict performance of topologies using paths.

    Divide into training and test sets.
    Look at the positive / negative cases in the training sets and find the most useful paths for prediction (call freqAnalysis).

    Then test their performance on the test set:
    - For topos containing the *most* useful paths, how many of them are positive / negative ones?
    - For topos containing the *least* useful paths, how many of them are positive / negative ones?

    :param training_data: contains paths, eff, vout info
    :param test_data: same format
    :return: None. plot to files
    """
    size_of_predictors = 20

    path_results = find_most_freq_paths(training_data, is_positive, is_negative)
    paths = [path for path, freq in path_results]

    pos_paths = paths[:size_of_predictors]
    neg_paths = paths[-size_of_predictors:]
    print('pos paths')
    pprint.pprint(pos_paths)
    print('neg paths')
    pprint.pprint(neg_paths)

    pos_counts, neg_counts = predict_efficiency_given_paths(pos_paths, test_data, is_positive, is_negative)
    plot_bar(range(1, size_of_predictors + 1), pos_counts, 'Containing Any of the First N *Positive* Paths', '*Positive* Topology Counts', 'first_n_pos')
    plot_bar(range(1, size_of_predictors + 1), neg_counts, 'Containing Any of the First N *Positive* Paths', '*Negative* Topology Counts', 'first_n_neg')

    pos_counts, neg_counts = predict_efficiency_given_paths(neg_paths, test_data, is_positive, is_negative)
    plot_bar(range(1, size_of_predictors + 1), pos_counts, 'Containing Any of the First N *Negative* Paths', '*Positive* Topology Counts', 'last_n_pos')
    plot_bar(range(1, size_of_predictors + 1), neg_counts, 'Containing Any of the First N *Negative* Paths', '*Negative* Topology Counts', 'last_n_neg')


if __name__ == '__main__':
    training_size = 1000
    test_size = 3000

    data = json.load(open('words.json'))
    names = list(data.keys())

    # slice the dict
    training_set = {name: data[name] for name in names[:training_size]}
    test_set = {name: data[name] for name in names[training_size: training_size + test_size]}

    predict_efficiency_using_first_n(training_set, test_set,
                                     is_positive=criteria.good_eff,
                                     is_negative=criteria.bad_eff)
