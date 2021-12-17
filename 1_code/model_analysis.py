import os
import sys

import torch

from get_transformer_reward import *
from ml_utils import initialize_model
import numpy as np
import argparse

from arguments import get_args
from reward_fn import compute_batch_reward
from topo_data_shun import Autopo, split_balance_data
from data_preprocessing import *


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


def get_gnn_single_data_reward(dataset, num_node, model_index, device, gnn_layers,
                               eff_model=None, vout_model=None, eff_vout_model=None, reward_model=None,
                               cls_vout_model=None):
    """
    Find the optimal simulator reward of the topologies with the top-k surrogate rewards.
    """
    n_batch_test = 0
    test_size = len(dataset) * 256
    print("Test bench size:", test_size)

    k_list = [int(test_size * 0.01 + 1), int(test_size * 0.05 + 1), int(test_size * 0.1 + 1), int(test_size * 0.5 + 1)]

    # for data in test_loader:
    data = dataset.data
    # load data in batches and compute their surrogate rewards
    data.to(device)
    L = data.node_attr.shape[0]
    B = int(L / num_node)
    node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
    if model_index == 0:
        edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
    else:
        edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
        edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

    adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])

    sim_eff = data.sim_eff.cpu().detach().numpy()
    sim_vout = data.sim_vout.cpu().detach().numpy()

    n_batch_test = n_batch_test + 1
    if eff_vout_model is not None:
        # using a model that can predict both eff and vout
        out = eff_vout_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
            gnn_layers)).cpu().detach().numpy()
        gnn_eff, gnn_vout = out[:, 0], out[:, 1]

    elif reward_model is not None:
        out = reward_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
            gnn_layers)).cpu().detach().numpy()

        # all_* variables are updated here, instead of end of for loop
        # todo refactor
        # continue
        gnn_reward = out[:, 0]
        return gnn_reward[0]

    elif cls_vout_model is not None:
        eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                               gnn_layers)).cpu().detach().numpy()
        vout = cls_vout_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
            gnn_layers)).cpu().detach().numpy()

        gnn_eff = eff.squeeze(1)
        gnn_vout = vout.squeeze(1)

        tmp_gnn_rewards = []
        for j in range(len(gnn_eff)):
            tmp_gnn_rewards.append(gnn_eff[j] * gnn_vout[j])
        # continue
        return tmp_gnn_rewards[0]

    elif model_index == 0:
        eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
        vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()

        gnn_eff = eff.squeeze(1)
        gnn_vout = vout.squeeze(1)
    else:
        eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                               gnn_layers)).cpu().detach().numpy()
        vout = vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                                 gnn_layers)).cpu().detach().numpy()

        gnn_eff = eff.squeeze(1)
        gnn_vout = vout.squeeze(1)

    gnn_reward = compute_batch_reward(gnn_eff, gnn_vout)
    return gnn_reward[0]




# def analyze_analytic(sweep, ):


def analyze_analytic(sweep, num_component, dataset, target_vout=50):
    analytic_rewards = []
    sim_rewards = []

    if sweep:
        _, anal_sweep_rewards = generate_anal_sweep_dataset(dataset=dataset, target_vout=50)
        analytic_rewards = list(anal_sweep_rewards.values())
        _, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
        sim_rewards = list(sim_sweep_rewards.values())
    else:
        for key_para, topo_info in dataset.items():
            analytic_reward = calculate_reward(effi={'efficiency': topo_info['eff_analytic'],
                                                     'output_voltage': topo_info['vout_analytic']},
                                               target_vout=target_vout)
            analytic_rewards.append(analytic_reward)
            sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                                'output_voltage': topo_info['vout']}, target_vout=target_vout)
            sim_rewards.append(sim_reward)
    print(evaluate_top_K(analytic_rewards, sim_rewards, k=k))
    return


def analyze_transformer(sweep, num_component, eff_model_seed, vout_model_seed, dataset, target_vout=50):
    args_file_name = './TransformerGP/UCFTopo_dev/config'
    sim_configs = {}
    transformer_rewards = []
    sim_rewards = []
    from UCFTopo_dev.utils.util import get_args
    get_args(args_file_name, sim_configs)
    sim = init_transformer_sim(num_component=num_component, sim_configs=sim_configs,
                               eff_model_seed=eff_model_seed, vout_model_seed=vout_model_seed)
    if sweep:
        transformer_sweep_rewards = {}
        sim_sweep_data, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
        sim_rewards = list(sim_sweep_rewards.values())
        for key_para, topo_info in dataset.items():
            transformer_reward, transformer_effi, transformer_vout = \
                get_prediction_from_topo_info(sim=sim,
                                              list_of_node=topo_info['list_of_node'],
                                              list_of_edge=topo_info['list_of_edge'],
                                              param=[0.1, 10, 100])
            if (key_para not in transformer_rewards) or (transformer_reward > transformer_rewards[key_para]):
                transformer_sweep_rewards[key_para] = transformer_reward
        transformer_rewards = list(transformer_sweep_rewards.values())
    else:
        for key_para, topo_info in dataset.items():
            transformer_reward, transformer_effi, transformer_vout = \
                get_prediction_from_topo_info(sim=sim,
                                              list_of_node=topo_info['list_of_node'],
                                              list_of_edge=topo_info['list_of_edge'],
                                              param=[0.1, 10, 100])
            transformer_rewards.append(transformer_reward)
            sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                                'output_voltage': topo_info['vout']}, target_vout=target_vout)
            sim_rewards.append(sim_reward)
    print(evaluate_top_K(transformer_rewards, sim_rewards, k=k))
    return


def clear_files(save_data_folder, raw_data_folder, ncomp):
    '''

    @param save_data_folder:
    @param raw_data_folder:
    @param ncomp:
    @return:
    '''

    os.system('rm '+raw_data_folder + "/dataset" + "_" + str(ncomp) + ".json")
    os.system('rm ' + save_data_folder + '/processed/data.pt')
    os.system('rm ' + save_data_folder + '/processed/pre_filter.pt')
    os.system('rm ' + save_data_folder + '/processed/pre_transform.ptb')




def analyze_gnn(sweep, num_component, args, dataset, target_vout=50):
    '''

    @param sweep:
    @param num_component:
    @param args:
    @param dataset:
    @return:
    '''
    args = get_args()

    batch_size = args.batch_size
    n_epoch = args.n_epoch

    # ======================== Data & Model ==========================#
    nf_size = 4
    ef_size = 3
    nnode = 7
    # single_data_folder = './datasets/single_data_datasets/2_row_dataset'
    # single_data_path = './datasets/single_data_datasets'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model = None, None, None, None, None
    if args.eff_vout_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        # model_state_dict, data_loader = torch.load(args.eff_vout_model)
        model_state_dict, _ = torch.load(args.eff_vout_model)

        model = initialize_model(model_index=args.model_index,
                                 gnn_nodes=args.gnn_nodes,
                                 gnn_layers=args.gnn_layers,
                                 pred_nodes=args.predictor_nodes,
                                 nf_size=nf_size,
                                 ef_size=ef_size,
                                 device=device,
                                 output_size=2)  # need to set output size of the network here
        model.load_state_dict(model_state_dict)


    elif args.reward_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        model_state_dict, data_loader = torch.load(args.reward_model)

        model = initialize_model(model_index=args.model_index,
                                 gnn_nodes=args.gnn_nodes,
                                 gnn_layers=args.gnn_layers,
                                 pred_nodes=args.predictor_nodes,
                                 nf_size=nf_size,
                                 ef_size=ef_size,
                                 device=device,
                                 output_size=1)  # need to set output size of the network here
        model.load_state_dict(model_state_dict)


    elif args.cls_vout_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        cls_vout_model_state_dict, _ = torch.load(args.cls_vout_model)
        cls_vout_model = initialize_model(model_index=args.model_index,
                                          gnn_nodes=args.gnn_nodes,
                                          gnn_layers=args.gnn_layers,
                                          pred_nodes=args.predictor_nodes,
                                          nf_size=nf_size,
                                          ef_size=ef_size,
                                          device=device)
        cls_vout_model.load_state_dict(cls_vout_model_state_dict)

        eff_model_state_dict, data_loader = torch.load(args.eff_model)
        eff_model = initialize_model(model_index=args.model_index,
                                     gnn_nodes=args.gnn_nodes,
                                     gnn_layers=args.gnn_layers,
                                     pred_nodes=args.predictor_nodes,
                                     nf_size=nf_size,
                                     ef_size=ef_size,
                                     device=device)
        eff_model.load_state_dict(eff_model_state_dict)

    else:
        vout_model_state_dict, _ = torch.load(args.vout_model)
        vout_model = initialize_model(model_index=args.model_index,
                                      gnn_nodes=args.gnn_nodes,
                                      gnn_layers=args.gnn_layers,
                                      pred_nodes=args.predictor_nodes,
                                      nf_size=nf_size,
                                      ef_size=ef_size,
                                      device=device)
        vout_model.load_state_dict(vout_model_state_dict)

        eff_model_state_dict, data_loader = torch.load(args.eff_model)
        eff_model = initialize_model(model_index=args.model_index,
                                     gnn_nodes=args.gnn_nodes,
                                     gnn_layers=args.gnn_layers,
                                     pred_nodes=args.predictor_nodes,
                                     nf_size=nf_size,
                                     ef_size=ef_size,
                                     device=device)
        eff_model.load_state_dict(eff_model_state_dict)

    gnn_rewards = []
    sim_rewards = []
    if sweep:
        sim_sweep_data, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
        sim_rewards = list(sim_sweep_rewards.values())
        gnn_sweep_rewards = {}
        for key_para, topo_info in dataset.items():
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)
            generate_single_data_file(key_para=key_para, topo_info=topo_info,
                                      single_data_path=args.single_data_path, ncomp=args.num_component)
            auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
            gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model, vout_model=vout_model,
                                                    eff_vout_model=eff_vout_model, reward_model=reward_model,
                                                    cls_vout_model=cls_vout_model,
                                                    num_node=nnode, model_index=args.model_index,
                                                    gnn_layers=args.gnn_layers, device=device)
            if (key_para not in gnn_sweep_rewards) or (gnn_reward > gnn_sweep_rewards[key_para]):
                gnn_sweep_rewards[key_para] = gnn_reward
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)
        gnn_rewards = list(gnn_sweep_rewards.values())

    else:
        for key_para, topo_info in dataset.items():
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)

            generate_single_data_file(key_para=key_para, topo_info=topo_info,
                                      single_data_path=args.single_data_path, ncomp=args.num_component)
            auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
            gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model, vout_model=vout_model,
                                                    eff_vout_model=eff_vout_model, reward_model=reward_model,
                                                    cls_vout_model=cls_vout_model,
                                                    num_node=nnode, model_index=args.model_index,
                                                    gnn_layers=args.gnn_layers, device=device)
            gnn_rewards.append(gnn_reward)
            sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
                                                'output_voltage': topo_info['vout']}, target_vout=target_vout)
            sim_rewards.append(sim_reward)
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)
    print(evaluate_top_K(gnn_rewards, sim_rewards, k=k))
    return


if __name__ == '__main__':
    # ======================== Arguments ==========================#
    args = get_args()
    dataset = json.load(open('./datasets/dataset_3.json'))

    for k in [100]:
        analyze_gnn(sweep=args.sweep, num_component=args.num_component, args=args, dataset=dataset)
        # test transformer

        analyze_transformer(sweep=args.sweep, num_component=args.num_component,
                            eff_model_seed=6, vout_model_seed=4, dataset=dataset)
        # analytic
        analyze_analytic(sweep=args.sweep, num_component=args.num_component,
                         dataset=dataset)
