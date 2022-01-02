import json
import os

import torch

from utils import *


def metric_get_trans_sweep_prediction_from_topo_info(sim, list_of_node, list_of_edge, candidate_params, target_vout=50):
    """

    @param candidate_params: [candidate params]} that represent a topology and set of candidate parameters
    @param list_of_edge: list of edge
    @param list_of_node: list of node
    @param sim:
    @param target_vout:
    @return:
    """
    max_reward, max_effi, max_vout = -1, -1, -500
    parameters = json.load(open("./param.json"))

    for param in candidate_params:
        fix_paras = {'Duty_Cycle': [param[0]], 'C': [param[1]], 'L': [param[2]]}
        parameters = assign_DC_C_and_L_in_param(param=parameters, fix_paras=fix_paras)
        effi = sim.get_surrogate_eff_with_topo_info(node_list=list_of_node, edge_list=list_of_edge)
        vout = sim.get_surrogate_vout_with_topo_info(node_list=list_of_node, edge_list=list_of_edge)
        reward = calculate_reward(effi={'efficiency': effi, 'output_voltage': vout},
                                  target_vout=target_vout)

        if reward > max_reward:
            max_reward, max_effi, max_vout = reward, effi, vout
    return max_reward, max_effi, max_vout


def metric_get_trans_prediction_from_topo_info(simulator, list_of_node, list_of_edge, param, target_vout=50):
    """
    get the prediction of a topology with list information and parameter
    @param sim:
    @param param: the [duty cycle, C, L] format, need to be add into a full format
    @param target_vout: target output voltage
    @param list_of_node:
    @param list_of_edge:
    @return: reward, efficiency, vout
    """
    parameters = json.load(open("./param.json"))
    fix_paras = {'Duty_Cycle': [param[0]], 'C': [param[1]], 'L': [param[2]]}
    parameters = assign_DC_C_and_L_in_param(param=parameters, fix_paras=fix_paras)

    effi = simulator.get_surrogate_eff_with_topo_info(node_list=list_of_node, edge_list=list_of_edge)
    vout = simulator.get_surrogate_vout_with_topo_info(node_list=list_of_node, edge_list=list_of_edge)
    reward = calculate_reward(effi={'efficiency': effi, 'output_voltage': vout},
                              target_vout=target_vout)
    return reward, effi, vout


def init_transformer_sim(num_component, sim_configs, eff_model_seed, vout_model_seed,
                         epoch=50, patience=50, no_cuda=False):
    """

    @param num_component:
    @param sim_configs:
    @param eff_model_seed:
    @param vout_model_seed:
    @param epoch:
    @param patience:
    @param no_cuda:
    @return: simulator of transformer surrogate model
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')
    from trans_topo_envs.TransformerRewardSim import TransformerRewardSimFactory
    dir = os.path.dirname(__file__)
    factory = TransformerRewardSimFactory(
        eff_model_file=os.path.join(dir, 'transformer_SVGP/save_model/5_comp_eff_'+str(eff_model_seed)+'.pt'),
        vout_model_file=os.path.join(dir, 'transformer_SVGP/save_model/5_comp_vout_'+str(vout_model_seed)+'.pt'),
        vocab_file=os.path.join(dir, 'transformer_SVGP/dataset_5_vocab.json'),
        dev_file=os.path.join(dir, 'transformer_SVGP/dataset_5_dev.json'),
        device=device, eff_model_seed=eff_model_seed, vout_model_seed=vout_model_seed,
        epoch=epoch, patience=patience, sample_ratio=1)
    sim_init = factory.get_sim_init()
    sim = sim_init(sim_configs, num_component)
    return sim


