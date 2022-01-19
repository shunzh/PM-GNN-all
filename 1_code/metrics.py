import _thread
import copy
import csv
import json
import os

import numpy as np

def top_K_coverage_on_ground_truth(preds, ground_truth, k_pred, k_ground_truth):
    """
    Find the top k_pred topologies predicted by the surrogate model, find how out much of top k_ground_truth topologies
    they can cover.
    :return: the coverage ratio
    """
    preds = np.array(preds)
    ground_truth = np.array(ground_truth)

    top_k_pred_indices = preds.argsort()[-k_pred:]
    top_k_gt_indices = ground_truth.argsort()[-k_ground_truth:]

    shared_indices = list(set(top_k_pred_indices) & set(top_k_gt_indices))

    return 1. * len(shared_indices) / k_ground_truth

def evaluate_top_K(preds, ground_truth, k, threshold=0.6):
    """
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: statistical analysis of the top-k topologies

            {'max': the maximum true reward in the top-k topologies predicted by the surrogate model
             'mean': the mean value of above,
             'std': the standard deviation of above,
             'precision': out of all of top-k topologies, how many of them are above the threshold,
             'recall': out of all of topologies above the threshold, how many of them are in the top-k topologies
            }
    """
    preds = np.array(preds)
    ground_truth = np.array(ground_truth)

    # evaluate top-k
    top_k_indices = preds.argsort()[-k:]

    # the ground truth values of these candidates
    ground_truth_of_top_k = ground_truth[top_k_indices]

    stats = {'max': max(ground_truth_of_top_k),
             'mean': np.mean(ground_truth_of_top_k),
             'std': np.std(ground_truth_of_top_k),
             # out of all of top-k topologies, how many of them are above the threshold
             'precision': len([x for x in ground_truth_of_top_k if x > threshold]) / k,
             # out of all of topologies above the threshold, how many of them are in the top-k topologies
             'recall': len([x for x in ground_truth_of_top_k if x > threshold]) /
                       (len([x for x in ground_truth if x > threshold]) + 1e-3)
             }

    return stats


def evaluate_bottom_K(preds, ground_truth, k, threshold=0.4):
    """
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: statistical analysis of the top-k topologies, in the same format as evaluate_top_K
    """
    preds = np.array(preds)
    ground_truth = np.array(ground_truth)

    # get the ones with the highest surrogate rewards
    bottom_k_indices = preds.argsort()[:k]

    # the ground truth values of these candidates
    ground_truth_of_bottom_k = ground_truth[bottom_k_indices]

    stats = {'max': max(ground_truth_of_bottom_k),
             'mean': np.mean(ground_truth_of_bottom_k),
             'std': np.std(ground_truth_of_bottom_k),
             # out of all of top-k topologies, how many of them are BELOW the threshold
             'precision': len([x for x in ground_truth_of_bottom_k if x < threshold]) / k,
             # out of all of topologies BELOW the threshold, how many of them are in the BOTTOM-k topologies
             'recall': len([x for x in ground_truth_of_bottom_k if x < threshold]) /
                       (len([x for x in ground_truth if x < threshold]) + 1e-3)
             }

    return stats


def compute_statistic(preds, ground_truth, good_topo_threshold=0.6):
    """
    Generate TPR, FPR points under different thresholds for the ROC curve.
    Reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: {threshold: {'TPR': true positive rate, 'FPR': false positive rate}}
    """
    preds, ground_truth = np.array(preds), np.array(ground_truth)

    thresholds = np.arange(0.001, 1., step=0.001)  # (0.1, 0.2, ..., 0.9)
    result = {}

    for thres in thresholds:
        good = np.where(ground_truth >= good_topo_threshold)[0]
        G_count = len(good)
        gt_pos = np.where(ground_truth >= thres)[0]
        P_count = len(gt_pos)
        gt_neg = np.where(ground_truth < thres)[0]
        N_count = len(gt_neg)
        predict_pos = np.where(preds >= thres)[0]
        PP_count = len(predict_pos)
        predict_neg = np.where(preds < thres)[0]
        PN_count = len(predict_neg)

        if len(gt_pos) == 0:
            TPR, TP_max, TP_count, TP_good_count, TP_good_rate, TP_good_whole_rate = 0, 0, 0, 0, 0, 0
        else:
            TP = np.intersect1d(gt_pos, predict_pos)
            TP_count = len(TP)
            TPR = TP_count / P_count
            if TP_count == 0:
                TP_max, TP_good_count, TP_good_rate, TP_good_whole_rate = 0, 0, 0, 0
            else:
                TP_max = max([ground_truth[idx] for idx in TP])
                TP_good = [ground_truth[idx] for idx in TP if ground_truth[idx] >= good_topo_threshold]
                TP_good_count = len(TP_good)
                TP_good_rate = TP_good_count / TP_count
                TP_good_whole_rate = TP_good_count / G_count

        if len(gt_pos) == 0:
            FNR, FN_max, FN_count, FN_good_count, FN_good_rate, FN_good_whole_rate = 0, 0, 0, 0, 0, 0
        else:
            FN = np.intersect1d(gt_pos, predict_neg)
            FN_count = len(FN)
            FNR = len(FN) / len(gt_pos)
            if FN_count == 0:
                FN_max, FN_good_count, FN_good_rate, FN_good_whole_rate = 0, 0, 0, 0
            else:
                FN_max = max([ground_truth[idx] for idx in FN])
                FN_good = [ground_truth[idx] for idx in FN if ground_truth[idx] >= good_topo_threshold]
                FN_good_count = len(FN_good)
                FN_good_rate = FN_good_count / FN_count
                FN_good_whole_rate = FN_good_count / G_count

        if len(gt_neg) == 0:
            FPR = 0
        else:
            FPR = len(np.intersect1d(gt_neg, predict_pos)) / len(gt_neg)

        result[thres] = dict(P_count=P_count, N_count=N_count, PP_count=PP_count, PN_count=PN_count,
                             TP_count=TP_count, TPR=TPR, TP_max=TP_max, TP_good_count=TP_good_count,
                             TP_good_rate=TP_good_rate, TP_good_whole_rate=TP_good_whole_rate,
                             FN_count=FN_count, FNR=FNR, FN_max=FN_max, FN_good_count=FN_good_count,
                             FN_good_rate=FN_good_rate, FN_good_whole_rate=FN_good_whole_rate,
                             FPR=FPR)

    return result


def save_statistic_result_to_files(statisitc_file_name, statisitc_result):
    """

    @param statisitc_file_name:
    @param statisitc_result:
    @return:
    """
    for threshold, statistic in statisitc_result.items():
        print(threshold, ' ', statistic)
    with open(statisitc_file_name + '.json', 'w') as f:
        json.dump(statisitc_result, f)
    f.close()


def write_results_to_csv_file(file_name, results):
    """
    @param file_name:
    @param results: {k:{'max':max, 'mean':mean, 'std':std, 'precision', precision, 'recall', recall}}
    @return:
    """

    output_for_ks = []
    index = 0
    header = ['k']
    for k, values in results.items():
        csv_list = [k]
        csv_list.extend(list(values.values()))
        output_for_ks.append(csv_list)
        header = ['k']
        header.extend([key for key in values])
    with open(file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(output_for_ks)
    f.close()


def save_raw_data_to_files(file_name, raw_data):
    """
    Save the raw data to the file to plot the ground_truth-prediction graph
    @param file_name: file name of the raw data, [model_name] + '-' + str(number of data) +
    '-' + '[ground truth name]_as_gt'
    @param raw_data: raw_data = {'pred_rewards': pred_rewards, 'ground_truth': ground_truth},
    it depends on which you want to treat as ground truth
    @return: None
    """
    with open(file_name + '.json', 'w') as f:
        json.dump(raw_data, f)
    f.close()

    raw_data_csv = []
    for i in range(len(raw_data['pred_rewards'])):
        raw_data_csv.append([raw_data['pred_rewards'][i], raw_data['ground_truth'][i]])
    header = ['pred_rewards', 'ground_truth']
    with open(file_name + '.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(raw_data_csv)
    f.close()


def class_data_and_save_to_files(class_data_file_name, raw_data, class_number=10):
    """
    save the reward class number into the file to plot the box-plot(need to Transpose the result)
    @param class_data_file_name: output class file name:[model] + '-' + str(number of data) +
    '-' + '-class-' + str(class_number)
    @param raw_data: list of the rewards that will be classified
    @param class_number: number of reward class. Every class include the rewards that in
    @return: classified raw data
    """
    class_raw_data = {}
    class_csv_raw_data = []
    header = []
    for i in range(class_number):
        class_raw_data[i * 1 / class_number] = [data for data in raw_data if
                                                i * 1 / class_number <= data < (i + 1) * 1 / class_number]
        class_csv_raw_data.append([data for data in raw_data if
                                   i * 1 / class_number <= data < (i + 1) * 1 / class_number])
        header.append(str(format(i * 1 / class_number, '.3f')) +
                      '-' + str(format((i + 1) * 1 / class_number, '.3f')))
    with open(class_data_file_name + '.json', 'w') as f:
        json.dump(class_raw_data, f)
    f.close()
    with open(class_data_file_name + '.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(class_csv_raw_data)
    f.close()
    return class_raw_data


def save_verification_results(result_folder, model_name, pred_rewards, ground_truth, ground_truth_name, dataset_size,
                              class_number=20):
    '''
    save all the verification results
    @param result_folder:
    @param ground_truth_name:
    @param model_name:
    @param pred_rewards:
    @param ground_truth:
    @param class_number:
    @param dataset_size:
    @return:
    '''
    os.system('mkdir ' + result_folder)
    raw_data = {'pred_rewards': pred_rewards, 'ground_truth': ground_truth}
    raw_data_file_name = model_name + '-' + str(dataset_size) + '-' + ground_truth_name + '_as_gt'
    save_raw_data_to_files(file_name=result_folder + raw_data_file_name, raw_data=raw_data)
    class_data_file_name = raw_data_file_name + '-class-' + str(class_number)
    class_data_and_save_to_files(class_data_file_name=result_folder + 'sim-class-'+ str(class_number),
                                 raw_data=ground_truth, class_number=class_number)
    class_data_and_save_to_files(class_data_file_name=result_folder + class_data_file_name,
                                 raw_data=pred_rewards, class_number=class_number)

    result = compute_statistic(preds=pred_rewards, ground_truth=ground_truth, good_topo_threshold=0.6)
    file_name = model_name + '_' + str(dataset_size) + '_statistic'
    save_statistic_result_to_files(statisitc_file_name=result_folder + file_name, statisitc_result=result)
    write_results_to_csv_file(result_folder + file_name + '.csv', result)


'''
pred_rewards = [gnn_raw_reward.item() for gnn_raw_reward in gnn_rewards]
raw_data = {'pred_rewards': pred_rewards, 'ground_truth': anal_rewards}
raw_data_file_name = model_name + '-' + str(len(isom_topo_dict)) + '-' + 'anal_as_gt'

save_raw_data_to_files(file_name=raw_data_file_name, raw_data=raw_data)
class_data_file_name = raw_data_file_name + '-class-' + str(class_number)
class_data_and_save_to_files(class_data_file_name=class_data_file_name,
                             raw_data=pred_rewards, class_number=class_number)

result = compute_statistic(preds=gnn_rewards, ground_truth=anal_rewards, good_topo_threshold=0.6)
file_name = configs['reward_model'] + '_' + str(len(isom_topo_dict)) + '_statistic'
save_statistic_result_to_files(statisitc_file_name=file_name, statisitc_result=result)
write_results_to_csv_file(file_name + '.csv', result)
statisitc_results[model_name] = result
'''
