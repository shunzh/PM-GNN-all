from datetime import datetime

import networkx as nx
from matplotlib import pyplot as plt


def calculate_reward(effi, target_vout, min_vout=None, max_vout=None):
    a = abs(target_vout) / 15
    if effi['efficiency'] > 1 or effi['efficiency'] < 0:
        return 0
    else:
        return effi['efficiency'] * (1.1 ** (-((effi['output_voltage'] - target_vout) / a) ** 2))


def print_calculate_rewards(target_vout, min_vout, max_vout):
    i = min_vout
    while i < max_vout:
        effi = {'efficiency': 1, 'output_voltage': i}
        reward = calculate_reward(effi, target_vout)
        print(i, reward)
        i += 0.1


def assign_DC_C_and_L_in_param(param, fix_paras):
    assert fix_paras['Duty_Cycle'] != []
    assert fix_paras['C'] != []
    assert fix_paras['L'] != []
    param['Duty_Cycle'] = fix_paras['Duty_Cycle']
    param['C'] = fix_paras['C']
    param['L'] = fix_paras['L']
    return param

def topo_info_valid(topo_info):
    list_of_edge = topo_info["list_of_edge"]

    edge_attr = topo_info["edge_attr"]
    node_attr = topo_info["node_attr"]

    if topo_info["vout"] / 100 > 1 or topo_info["vout"] / 100 < 0:
        # print(topo_info)
        return False
    if topo_info["vout"] == -1:
        return False
    if topo_info["eff"] < 0 or topo_info["eff"] > 1:
        # print(topo_info)
        return False
    return True

def visualize(list_of_node, list_of_edge, title=None, figure_folder=None, file_name='default'):
    """
    plot circuit topology
    @param list_of_node:
    @param list_of_edge:
    @param title: title of the plot
    @param figure_folder: where do you want to put the plot
    @return:
    """
    T = nx.Graph()
    T.add_nodes_from(list_of_node)
    T.add_edges_from(list_of_edge)
    if bool(title):
        print('title', title)
        plt.title(" ")
    nx.draw(T, with_labels=True)

    # plt.show()
    dt = datetime.now().strftime(figure_folder + file_name)
    # dt = datetime.datetime.now().strftime(figure_folder + title)
    # print(dt+'-'+str(int(title)))
    plt.savefig(dt + '-' + str(int(title)))
    plt.close()

