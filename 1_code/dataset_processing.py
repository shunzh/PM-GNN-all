"""
This module includes the functions that related with the original dataset processing. Including adding some missing
 data, generate datasets for testing. Note that after generate the dataset, maybe it is still need to be
"""
import copy
import json

from data_preprocessing import generate_sweep_dataset
from SimulatorModel.simulate_with_topology import get_single_sim_result_with_topo_info


def correct_expression_dict(outer_expression_dict):
    '''

    @param outer_expression_dict:
    @return:
    '''
    for k, v in outer_expression_dict.items():
        if v['Expression'] != 'invalid' and len(v['Expression']) == 1:
            for k in v['Expression']:
                key = k
            v['Expression'] = v['Expression'][key]
    return outer_expression_dict


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


def read_no_sweep_sim_result():
    '''
    Read the simulation hash table
    @return:
    '''
    data_json_file = json.load(open('no-sweep-key-sim-result.json'))
    return data_json_file


def save_no_sweep_sim_result(simu_results):
    '''
    Save the newly simulated data to simulation hash table
    @param simu_results:
    @return:
    '''
    with open('no-sweep-key-sim-result.json', 'w') as f:
        json.dump(simu_results, f)
    f.close()


def adding_missing_datapoint_in_dataset(dataset):
    key_sim_effi_ = json.load(open("./no-sweep-key-sim-result.json"))
    parameters = json.load(open("./param.json"))
    candidate_parameters = [[duty_cycle, cap, ind] for duty_cycle in parameters['Duty_Cycle']
                            for cap in parameters['C']
                            for ind in parameters['L']]
    for key_para, topo_info in dataset.items():
        tmp_topo_info = copy.deepcopy(topo_info)
        key = key_para.split('$')[0]
        for para in candidate_parameters:
            if key + '$' + str(para) not in dataset:
                tmp_topo_info["duty_cycle"] = para[0]
                tmp_reward, effi, vout = \
                    get_single_sim_result_with_topo_info(list_of_node=tmp_topo_info['list_of_node'],
                                                         list_of_edge=tmp_topo_info['list_of_edge'],
                                                         netlist=tmp_topo_info['netlist'], para=para,
                                                         key_sim_effi_=key_sim_effi_, target_vout=50)
                tmp_topo_info['eff'], tmp_topo_info['vout'] = effi, vout
                dataset[key + '$' + str(para)] = tmp_topo_info
    return dataset
