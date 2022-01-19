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


def generate_no_sweep_dataset(dataset, target_vout):
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

        sim_sweep_rewards[key_para] = sim_reward
        sim_sweep_data[key_para] = topo_info
    return sim_sweep_data, sim_sweep_rewards


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
        if max_reward_key_para not in max_reward_sweep_data:
            max_reward_sweep_data[max_reward_key_para] = topo_info
            max_reward_sweep_data[max_reward_key_para]['duty_cycle'] = 0.5
    # assert (len(max_reward_sweep_data) == int(len(dataset) / 5))
    # assert (len(max_reward_sweep_rewards) == int(len(dataset) / 5))
    return max_reward_sweep_data, max_reward_sweep_rewards


def generate_single_data_file(key_para, topo_info, single_data_path, ncomp):
    """
    For gnn prediction. gnn must read from file to get the prediction(Auto). Thus every time we write a single topology
    +para information to ./datasets/single_data_datasets/2_row_dataset'
    @param key_para:
    @param topo_info:
    @param single_data_path:
    @param ncomp:
    @return:
    """

    single_data = {key_para: topo_info}
    print('rm ' + single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    os.system('rm ' + single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    with open(single_data_path + "/dataset" + "_" + str(ncomp) + ".json", 'w') as f:
        json.dump(single_data, f)
    f.close()


def generate_sim_rewards(sweep, dataset, key_order, target_vout):
    """

    @param key_order:
    @param sweep:
    @param dataset:
    @param target_vout:
    @return:
    """
    sim_rewards_dict = {}
    if sweep:
        sim_sweep_data, sim_rewards_dict = generate_sweep_dataset(dataset=dataset, target_vout=target_vout)
    else:
        for key_para, topo_info in dataset.items():
            sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                                'output_voltage': topo_info['vout']}, target_vout=target_vout)
            sim_rewards_dict[key_para] = sim_reward
    sim_rewards = reordered_rewards(key_order=key_order, rewards_dict=sim_rewards_dict)
    return sim_rewards


def generate_statistic_rewards(sweep, dataset, key_order, target_vout):
    """

    @param key_order:
    @param sweep:
    @param dataset:
    @param target_vout:
    @return:
    """
    # # 5 comp
    vout_range_list = [-500, -200, -150, -100, -50, 0, 50, 100, 150, 200, 500]
    target_vout_list = [-250, -150, -100, -50, 25, 50, 75, 150, 250]

    # # 3 comp
    # vout_range_list = [-500, -200, -150, -100, -50, 0, 50, 100, 150, 200, 500]
    # target_vout_list = [-200, -100, -40, 75, 150, 200]

    # # 4 comp
    # vout_range_list = [-500, -200, -150, -100, -50, 0, 50, 100, 150, 200, 500]
    # target_vout_list = [-200, -100, -40, 75, 150, 200]
    # target_vout_list = [i for i in range(-250, 250, 6)]



    vout_range_result = {}

    results = {}
    for target_vout in target_vout_list:
        results[target_vout] = [-1, 0]

    sim_rewards_dict = {}
    if sweep:
        sim_sweep_data, sim_rewards_dict = generate_sweep_dataset(dataset=dataset, target_vout=target_vout)
    else:
        for key_para, topo_info in dataset.items():
            sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                                'output_voltage': topo_info['vout']}, target_vout=target_vout)
            sim_rewards_dict[key_para] = [sim_reward, topo_info['vout'], topo_info['eff']]
    for key, value in sim_rewards_dict.items():
        for target_vout in target_vout_list:
            sim_reward = calculate_reward(effi={'efficiency': value[2], 'output_voltage': value[1]},
                                          target_vout=target_vout)
            if sim_reward > results[target_vout][0]:
                results[target_vout][0] = sim_reward
            if sim_reward > 0.5:
                results[target_vout][1] += 1
        for i in range(0, len(vout_range_list) - 1):
            if vout_range_list[i] <= value[1] < vout_range_list[i + 1]:
                if vout_range_list[i] not in vout_range_result:
                    vout_range_result[vout_range_list[i]] = 1
                else:
                    vout_range_result[vout_range_list[i]] += 1

    print('-------------------------')
    for k, v in results.items():
        print(k, '\t', v[0], '\t', v[1])

    print('-------------------------')
    for k, v in vout_range_result.items():
        print(k, '\t', v)
    return sim_rewards_dict


def generate_keys_for_consistent_rewards(sweep, dataset):
    """

    @param sweep:
    @param dataset:
    @return:
    """
    dataset_key = {}
    assert isinstance(dataset, dict)
    keys = list(dataset.keys())
    if sweep:
        for key_para in keys:
            key = key_para.split('$')[0]
            dataset_key[key] = 1
        keys = list(dataset_key.keys())
    random.shuffle(keys)
    return keys


def reordered_rewards(rewards_dict, key_order):
    """

    @param rewards_dict:
    @param key_order:
    @return:
    """
    assert len(key_order) == len(rewards_dict)
    rewards = []
    for key in key_order:
        rewards.append(rewards_dict[key])
    return rewards
