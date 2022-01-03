from copy import deepcopy

# from SimulatorModel.gen_topo import *
# from SimulatorModel.UCT_data_collection import *
from utils import *
from AnalyticModel.get_analytic_reward import get_analytics_result, get_one_circuit, get_one_expression
from AnalyticModel.gen_topo_for_analytic import *

simu_root_folder = './SimulatorModel/simulation_files'


def simulate_one_analytics_result(analytics_info):
    """
        input a topology, return the simulation information of different duty cycles
        """
    result_dict = {}
    if analytics_info is None:
        return {'[]': {'efficiency': 0, 'Vout': -500}}
    analytic = analytics_info
    for fn in analytic:
        print(int(fn[6:9]))
        count = 0
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.cki'
            convert_cki_full_path(simu_root_folder + '/' + file_name, param_value, device_name, netlist)
            print(file_name)
            simulate(simu_root_folder + '/' + file_name)
            count = count + 1
    # only one fn in analytic if only simulate one
    for fn in analytic:
        count = 0
        for param in analytic[fn]:
            param_value = [float(value) for value in param[1:-1].split(', ')]
            device_name = analytic[fn][param][0]
            netlist = analytic[fn][param][1]
            file_name = fn + '-' + str(count) + '.simu'
            path = file_name
            vin = param_value[device_name['Vin']]
            freq = param_value[device_name['Frequency']] * 1000000
            rin = param_value[device_name['Rin']]
            rout = param_value[device_name['Rout']]
            print(file_name)
            result = calculate_efficiency(path, vin, freq, rin, rout)
            # print(result)
            param = str(param)
            param_spl = param.split(',')
            para_val = param_spl[0].replace('(', '')
            result_dict[para_val] = result
            count = count + 1
        return result_dict


def get_single_sim_result_with_topo_info(list_of_node, list_of_edge, netlist, para, key_sim_effi_, target_vout=50):
    '''

    @param list_of_node:
    @param list_of_edge:
    @param netlist:
    @param para: [DC, C, L]
    @param key_sim_effi_: simulator hash table
    @param target_vout:
    @return:
    '''
    effi_info = {}
    topk_max_reward, topk_max_para, topk_max_effi, topk_max_vout = -1, [], 0, 500

    parameters = json.load(open("./param.json"))
    fix_paras = {'Duty_Cycle': [para[0]], 'C': [para[1]], 'L': [para[2]]}
    parameters = assign_DC_C_and_L_in_param(param=parameters, fix_paras=fix_paras)

    current_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)

    if key_sim_effi_.__contains__(current_key + '$' + str(para)):
        effi_info_from_hash = key_sim_effi_[current_key + '$' + str(para)]
        print('find pre simulated &&&&&&&&&&&&&&&&&&&&&&')
        effi, vout = effi_info_from_hash[0], effi_info_from_hash[1]
    else:
        key = key_circuit_from_lists(edge_list=list_of_edge, node_list=list_of_node, net_list=net_list)
        circuit_info = get_one_circuit(key=key, net_list=net_list, parameters=parameters)
        expression = get_one_expression(circuit_info=circuit_info)
        results_tmp = get_analytics_result(expression=expression, parameters=parameters)
        current_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        addtional_effi_info = simulate_one_analytics_result(results_tmp)
        assert len(addtional_effi_info) == 1
        for k, effi_result in addtional_effi_info.items():
            effi, vout = effi_result['efficiency'], effi_result['Vout']
        if current_key + '$' + str(para) not in key_sim_effi_:
            key_sim_effi_[current_key + '$' + str(para)] = [effi, vout]
    print(effi, vout)
    tmp_reward = calculate_reward({'efficiency': effi, 'output_voltage': vout}, target_vout)

    return tmp_reward, effi, vout


def find_simu_max_reward_para(para_effi_info, target_vout=50):
    # return the simulation information about the topology in topk that has the highest simulation reward
    max_reward, max_effi_info, max_para = -1, None, None
    if 'result_valid' in para_effi_info and para_effi_info['result_valid'] == False:
        return 0, {'efficiency': 0, 'Vout': -500}, -1
    for para in para_effi_info:
        effi_info = para_effi_info[para]
        effi = {'efficiency': effi_info['efficiency'], 'output_voltage': effi_info['Vout']}
        tmp_reward = calculate_reward(effi, target_vout=target_vout)
        if tmp_reward > max_reward:
            max_reward = tmp_reward
            max_effi_info = deepcopy(effi_info)
            max_para = para
    return max_reward, max_effi_info, max_para
