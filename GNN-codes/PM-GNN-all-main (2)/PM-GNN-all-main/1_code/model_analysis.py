import torch
from ml_utils import initialize_model
import numpy as np
import argparse

from reward_fn import compute_batch_reward


def compute_roc(preds, ground_truth):
    """
    Generate TPR, FPR points under different thresholds for the ROC curve.
    Reference: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

    :param preds: a list of surrogate model predictions
    :param ground_truth: a list of ground-truth values (e.g. by the simulator)
    :return: {threshold: {'TPR': true positive rate, 'FPR': false positive rate}}
    """
    preds, ground_truth = np.array(preds), np.array(ground_truth)

    thresholds = np.arange(0.1, 1., step=0.1) # (0.1, 0.2, ..., 0.9)
    result = {}

    for thres in thresholds:
        gt_pos = np.where(ground_truth >= thres)[0]
        gt_neg = np.where(ground_truth < thres)[0]

        predict_pos = np.where(preds >= thres)[0]

        if len(gt_pos) == 0: TPR = 0
        else: TPR = len(np.intersect1d(gt_pos, predict_pos)) / len(gt_pos)

        if len(gt_neg) == 0: FPR = 0
        else: FPR = len(np.intersect1d(gt_neg, predict_pos)) / len(gt_neg)

        result[thres] = {'TPR': TPR, 'FPR': FPR}

    return result


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


def analyze_model(test_loader, num_node, model_index, device, gnn_layers,
                  eff_model=None, vout_model=None, eff_vout_model=None, reward_model=None, cls_vout_model=None):
    """
    Find the optimal simulator reward of the topologies with the top-k surrogate rewards.
    """
    n_batch_test = 0

    sim_rewards = []
    gnn_rewards = []

    all_sim_eff = []
    all_sim_vout = []
    all_gnn_eff = []
    all_gnn_vout = []

    test_size=len(test_loader)*256
    print("Test bench size:",test_size)

    k_list = [int(test_size*0.01+1),int(test_size*0.05+1),int(test_size*0.1+1),int(test_size*0.5+1)]

    for data in test_loader:
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
            out = eff_vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()
            gnn_eff, gnn_vout = out[:, 0], out[:, 1]

        elif reward_model is not None:
            out = reward_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()
            all_sim_eff.extend(sim_eff)
            all_sim_vout.extend(sim_vout)
            sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
            gnn_rewards.extend(out[:,0])

            # all_* variables are updated here, instead of end of for loop
            # todo refactor
            continue

        elif cls_vout_model is not None:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()
            vout = cls_vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)
            all_sim_eff.extend(sim_eff)
            all_sim_vout.extend(sim_vout)
            all_gnn_eff.extend(gnn_eff)
            all_gnn_vout.extend(gnn_vout)

            tmp_gnn_rewards=[]
            for j in range(len(gnn_eff)):
                tmp_gnn_rewards.append(gnn_eff[j]*gnn_vout[j])

            sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
            gnn_rewards.extend(tmp_gnn_rewards)
            continue

        elif model_index == 0:
            eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)
        else:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device), gnn_layers)).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)

        all_sim_eff.extend(sim_eff)
        all_sim_vout.extend(sim_vout)
        all_gnn_eff.extend(gnn_eff)
        all_gnn_vout.extend(gnn_vout)

        sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
        gnn_rewards.extend(compute_batch_reward(gnn_eff, gnn_vout))
        #out_list.extend(r)

    for k in k_list:
        print('k', k)
        print('Top-k topology analysis:')
        print(evaluate_top_K(gnn_rewards, sim_rewards, k))
        print('Bottom-k topology analysis:')
        print(evaluate_bottom_K(gnn_rewards, sim_rewards, k))
        #gnn_coverage[k] = top_K_coverage_on_ground_truth(gnn_rewards, sim_rewards, k, k)


if __name__ == '__main__':
    # debug
    #compute_roc(np.array([.1, .5, .7]), np.array([.2, .5, .8]))
    # ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=10, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=20, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=10,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=2, help='number of layer')
    parser.add_argument('-model_index', type=int, default=2, help='model index')

    parser.add_argument('-eff_model', type=str, default=None, help='eff model file name')
    parser.add_argument('-vout_model', type=str, default=None, help='vout model file name')
    parser.add_argument('-eff_vout_model', type=str, default=None, help='file of model that predicts both eff and vout')
    parser.add_argument('-reward_model', type=str, default=None, help='file of model that predicts both eff and vout')
    parser.add_argument('-cls_vout_model', type=str, default=None, help='eff model file name')
 

    args = parser.parse_args()

    batch_size = args.batch_size
    n_epoch = args.n_epoch

    # ======================== Data & Model ==========================#
    nf_size = 4
    ef_size = 3
    nnode = 7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.eff_vout_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        model_state_dict, data_loader = torch.load(args.eff_vout_model)

        model = initialize_model(model_index=args.model_index,
                                 gnn_nodes=args.gnn_nodes,
                                 gnn_layers=args.gnn_layers,
                                 pred_nodes=args.predictor_nodes,
                                 nf_size=nf_size,
                                 ef_size=ef_size,
                                 device=device,
                                 output_size=2) # need to set output size of the network here
        model.load_state_dict(model_state_dict)

        analyze_model(test_loader=data_loader, eff_vout_model=model,
                      num_node=nnode, model_index=args.model_index, gnn_layers=args.gnn_layers, device=device)

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
                                 output_size=1) # need to set output size of the network here
        model.load_state_dict(model_state_dict)

        analyze_model(test_loader=data_loader, reward_model=model,
                      num_node=nnode, model_index=args.model_index, gnn_layers=args.gnn_layers, device=device)

    elif args.cls_vout_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        cls_vout_model_state_dict, data_loader = torch.load(args.cls_vout_model)
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

        analyze_model(test_loader=data_loader, eff_model=eff_model, cls_vout_model=cls_vout_model,
                      num_node=nnode, model_index=args.model_index, gnn_layers=args.gnn_layers, device=device)
    else:
        vout_model_state_dict, data_loader = torch.load(args.vout_model)
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

        analyze_model(test_loader=data_loader, eff_model=eff_model, vout_model=vout_model,
                      num_node=nnode, model_index=args.model_index, gnn_layers=args.gnn_layers, device=device)
