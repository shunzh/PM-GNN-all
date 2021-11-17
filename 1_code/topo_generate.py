import torch
from ml_utils import initialize_model
import numpy as np
import argparse

from reward_fn import compute_batch_reward


def evaluate_top_K(out, ground_truth, k):
    out = np.array(out)
    ground_truth = np.array(ground_truth)

    # get the ones with the highest surrogate rewards
    candidates = out.argsort()[-k:]

    # the ground truth values of these candidates
    candidate_gt = ground_truth[candidates]
    return max(candidate_gt)

def optimize_reward(test_loader, eff_model, vout_model, num_node, model_index, device):
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

    k_list = [1, 10, 20, 30, 50, 100]

    gnn_performs = {k: [] for k in k_list}

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
        if model_index == 0:
            eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
        else:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device))).cpu().detach().numpy()

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
        gnn_perform = evaluate_top_K(gnn_rewards, sim_rewards, k)
        gnn_performs[k].append(gnn_perform)

    return gnn_performs


if __name__ == '__main__':
    # ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=10, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=50, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=10,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=4, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')

    parser.add_argument('-eff_model', type=str, default='reg_eff3.pt', help='eff model file name')
    parser.add_argument('-vout_model', type=str, default='reg_vout3.pt', help='vout model file name')

    args = parser.parse_args()

    batch_size = args.batch_size
    n_epoch = args.n_epoch

    # ======================== Data & Model ==========================#
    nf_size = 4
    ef_size = 3
    nnode = 7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eff_model_state_dict, data_loader = torch.load(args.eff_model)
    eff_model = initialize_model(model_index=args.model_index,
                                 gnn_nodes=args.gnn_nodes,
                                 gnn_layers=args.gnn_layers,
                                 pred_nodes=args.predictor_nodes,
                                 nf_size=nf_size,
                                 ef_size=ef_size,
                                 device=device)
    eff_model.load_state_dict(eff_model_state_dict)

    vout_model_state_dict, data_loader = torch.load(args.vout_model)
    vout_model = initialize_model(model_index=args.model_index,
                                  gnn_nodes=args.gnn_nodes,
                                  gnn_layers=args.gnn_layers,
                                  pred_nodes=args.predictor_nodes,
                                  nf_size=nf_size,
                                  ef_size=ef_size,
                                  device=device)
    vout_model.load_state_dict(vout_model_state_dict)

    print(optimize_reward(data_loader, eff_model, vout_model, nnode, args.model_index, device))
