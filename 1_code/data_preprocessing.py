import json
import random

from utils import *
import os


def random_dic(dicts):
    '''
    random the dict
    @param dicts:
    @return:
    '''
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts[key]
    return new_dic


def generate_sweep_dataset(dataset, target_vout):
    """

    @param dataset:
    @param target_vout:
    @return:
    """
    sim_sweep_rewards = {}
    sim_sweep_data = {}
    for key_para, topo_info in dataset.items():
        sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                            'output_voltage': topo_info['vout']}, target_vout=target_vout)
        key = key_para.split('$')[0]
        if (key not in sim_sweep_rewards) or (sim_reward > sim_sweep_rewards[key]):
            sim_sweep_rewards[key] = sim_reward
            sim_sweep_data[key] = topo_info
    # assert (len(sim_sweep_data) == int(len(dataset)/5))
    return sim_sweep_data, sim_sweep_rewards


def generate_anal_sweep_dataset(dataset, target_vout):
    """

    @param dataset:
    @param target_vout:
    @return:
    """
    anal_sweep_rewards = {}
    anal_sweep_data = {}
    for key_para, topo_info in dataset.items():
        analytic_reward = calculate_reward(effi={'efficiency': topo_info['eff_analytic'],
                                                 'output_voltage': topo_info['vout_analytic']},
                                           target_vout=target_vout)
        key = key_para.split('$')[0]
        if (key not in anal_sweep_rewards) or (analytic_reward > anal_sweep_rewards[key]):
            anal_sweep_rewards[key] = analytic_reward
            anal_sweep_data[key] = topo_info
    assert (len(anal_sweep_data) == int(len(dataset) / 5))
    return anal_sweep_data, anal_sweep_rewards


def generate_dataset_for_gnn_max_reward_prediction(dataset, target_vout):
    """
    we use the topology with 0.5 duty cycle to represent the topology that has the max reward
    @param dataset:
    @param target_vout:
    @return:
    """
    fixed_para_str = '[0.5, 10, 100]'
    max_reward_sweep_rewards = {}
    max_reward_sweep_data = {}
    for key_para, topo_info in dataset.items():
        sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                            'output_voltage': topo_info['vout']}, target_vout=target_vout)
        key = key_para.split('$')[0]
        max_reward_key_para = key + '$' + fixed_para_str
        if (max_reward_key_para not in max_reward_sweep_rewards) or \
                (sim_reward > max_reward_sweep_rewards[max_reward_key_para]):
            max_reward_sweep_rewards[max_reward_key_para] = sim_reward
        # TODO: actually current dataset does not include all the 5 components, so need add missing data in the dataset
        if key_para == max_reward_key_para:
            max_reward_sweep_data[max_reward_key_para] = topo_info
    assert (len(max_reward_sweep_data) == int(len(dataset) / 5))
    assert (len(max_reward_sweep_rewards) == int(len(dataset) / 5))
    return max_reward_sweep_data, max_reward_sweep_rewards


def generate_single_data_file(key_para, topo_info, single_data_path, ncomp):
    '''
    For gnn prediction. gnn must read from file to get the prediction(Auto). Thus every time we write a single topology
    +para information to ./datasets/single_data_datasets/2_row_dataset'
    @param key_para:
    @param topo_info:
    @param single_data_path:
    @param ncomp:
    @return:
    '''

    single_data = {key_para: topo_info}
    print('rm ' + single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    os.system('rm ' + single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    with open(single_data_path + "/dataset" + "_" + str(ncomp) + ".json", 'w') as f:
        json.dump(single_data, f)
    f.close()





