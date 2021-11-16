import os

import torch
from ml_utils import initialize_model, optimize_reward
import argparse


if __name__ == '__main__':

    # ======================== Arguments ==========================#

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', type=str, default="../0_rawdata", help='raw data path')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=10, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=100, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=100,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=3, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')
    parser.add_argument('-threshold', type=float, default=0, help='classification threshold')

    parser.add_argument('-eff_model', type=str, default='reg_eff.pt', help='eff model file name')
    parser.add_argument('-vout_model', type=str, default='reg_vout.pt', help='vout model file name')

    args = parser.parse_args()

    batch_size = args.batch_size
    n_epoch = args.n_epoch
    th = args.threshold

    # ======================== Data & Model ==========================#
    nf_size = 4
    ef_size = 3
    nnode = 6
    if args.model_index == 0:
        ef_size = 6

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
