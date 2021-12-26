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
        if key_para == max_reward_key_para:
            max_reward_sweep_data[max_reward_key_para] = topo_info
    assert (len(max_reward_sweep_data) == int(len(dataset) / 5))
    assert (len(max_reward_sweep_rewards) == int(len(dataset) / 5))
    return max_reward_sweep_data, max_reward_sweep_rewards


def generate_single_data_file(key_para, topo_info, single_data_path, ncomp):
    '''

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


def dataset_statistic(dataset, target_vout=50):
    """
    note that if run this, may need to disable the assert about length of sweeped data in generate_sweep_dataset
    @param dataset:
    @return:
    """
    count_static = {}
    sim_sweep_data, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=target_vout)
    print(len(sim_sweep_data))
    combination_count_mapping = {}
    for key, topo_info in sim_sweep_data.items():
        if sim_sweep_rewards[key] > 0.5:
            component_count = {'Sa': 0, 'Sb': 0, 'C': 0, 'L': 0}
            for component in topo_info['list_of_node']:
                if type(component) == str:
                    for component_type in component_count:
                        if component_type in component:
                            component_count[component_type] += 1
            print(component_count)
            count_str = str(component_count)
            if count_str not in count_static:
                count_static[count_str] = 1
            else:
                count_static[count_str] += 1
    for k, v in count_static.items():
        print(k, ' ', v)
