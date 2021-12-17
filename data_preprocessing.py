import json
from utils import *
import os


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

        if (key_para not in sim_sweep_rewards) or (sim_reward > sim_sweep_rewards[key_para]):
            sim_sweep_rewards[key_para] = sim_reward
            sim_sweep_data[key_para] = topo_info
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

        if (key_para not in anal_sweep_rewards) or (analytic_reward > anal_sweep_rewards[key_para]):
            anal_sweep_rewards[key_para] = analytic_reward
            anal_sweep_data[key_para] = topo_info
    return anal_sweep_data, anal_sweep_rewards


def generate_single_data_file(key_para, topo_info, single_data_path, ncomp):
    '''

    @param key_para:
    @param topo_info:
    @param single_data_path:
    @param ncomp:
    @return:
    '''

    single_data = {key_para: topo_info}
    print('rm '+single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    os.system('rm '+single_data_path + "/dataset" + "_" + str(ncomp) + ".json")
    with open(single_data_path + "/dataset" + "_" + str(ncomp) + ".json", 'w') as f:
        json.dump(single_data, f)
    f.close()
