import torch
# from TransformerModel.get_transformer_reward import *
from AnalyticModel.get_analytic_reward import *
from ml_utils import initialize_model
import numpy as np

from arguments import get_args
from reward_fn import compute_batch_reward
from topo_data import Autopo
from data_preprocessing import *
from metrics import *
from dataset_processing import *
from utils import *


def clear_files(save_data_folder, raw_data_folder, ncomp):
    '''

    @param save_data_folder:
    @param raw_data_folder:
    @param ncomp:
    @return:
    '''

    os.system('rm ' + raw_data_folder + "/dataset" + "_" + str(ncomp) + ".json")
    os.system('rm ' + save_data_folder + '/processed/data.pt')
    os.system('rm ' + save_data_folder + '/processed/pre_filter.pt')
    os.system('rm ' + save_data_folder + '/processed/pre_transform.ptb')


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
    # Be careful about the order
    if reward_model is not None:
        out = reward_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
            gnn_layers)).cpu().detach().numpy()

        # all_* variables are updated here, instead of end of for loop
        # todo refactor
        # continue
        gnn_reward = out[:, 0]
        return gnn_reward[0]

    elif eff_vout_model is not None:
        # using a model that can predict both eff and vout
        out = eff_vout_model(input=(
            node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
            gnn_layers)).cpu().detach().numpy()
        gnn_eff, gnn_vout = out[:, 0], out[:, 1]

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


def init_gnn_models(args):
    '''

    @param sweep:
    @param num_component:
    @param args:
    @param dataset:
    @return:
    '''

    batch_size = args.batch_size
    n_epoch = args.n_epoch

    # ======================== Data & Model ==========================#
    nf_size = 4
    ef_size = 3
    # single_data_folder = './datasets/single_data_datasets/2_row_dataset'
    # single_data_path = './datasets/single_data_datasets'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model = None, None, None, None, None
    if args.reward_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        reward_model_state_dict, _ = torch.load(args.reward_model)

        reward_model = initialize_model(model_index=args.model_index,
                                        gnn_nodes=args.gnn_nodes,
                                        gnn_layers=args.gnn_layers,
                                        pred_nodes=args.predictor_nodes,
                                        nf_size=nf_size,
                                        ef_size=ef_size,
                                        device=device,
                                        output_size=1)  # need to set output size of the network here
        reward_model.load_state_dict(reward_model_state_dict)
        output_file = args.reward_model.replace('.pt', '.csv')
    elif args.eff_vout_model is not None:
        # if this argument is set, load one model that predicts both eff and vout
        # model_state_dict, data_loader = torch.load(args.eff_vout_model)
        eff_vout_model_state_dict, _ = torch.load(args.eff_vout_model)

        eff_vout_model = initialize_model(model_index=args.model_index,
                                          gnn_nodes=args.gnn_nodes,
                                          gnn_layers=args.gnn_layers,
                                          pred_nodes=args.predictor_nodes,
                                          nf_size=nf_size,
                                          ef_size=ef_size,
                                          device=device,
                                          output_size=2)  # need to set output size of the network here
        eff_vout_model.load_state_dict(eff_vout_model_state_dict)
        output_file = args.eff_vout_model.replace('.pt', '.csv')





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
        output_file = args.cls_vout_model.replace('.pt', '.csv')

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
        output_file = args.eff_model.replace('.pt', '.csv')

    print('finish init model and result are writen in ', output_file)
    return eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model, output_file


def generate_gnn_rewards(key_order, dataset, sweep, args,
                         eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_rewards_dict = {}
    if sweep:

        # note that, if we want to test the model that predict the max reward of a topology, we need to append, 'only_max' into the model name
        if (args.reward_model is not None) and ('only_max' in args.reward_model):
            gnn_dataset, _ = generate_dataset_for_gnn_max_reward_prediction(dataset=dataset, target_vout=50)
        else:
            gnn_dataset = dataset

        for key_para, topo_info in gnn_dataset.items():
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)
            generate_single_data_file(key_para=key_para, topo_info=topo_info,
                                      single_data_path=args.single_data_path, ncomp=args.num_component)
            if topo_info_valid(topo_info=topo_info):
                # if nn_node change, this must be changed too
                auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
                gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model,
                                                        vout_model=vout_model,
                                                        eff_vout_model=eff_vout_model, reward_model=reward_model,
                                                        cls_vout_model=cls_vout_model,
                                                        num_node=args.nnode, model_index=args.model_index,
                                                        gnn_layers=args.gnn_layers, device=device)
                gnn_reward = gnn_reward.item()
            else:
                gnn_reward = 0
            key = key_para.split('$')[0]
            if (key not in gnn_rewards_dict) or (gnn_reward > gnn_rewards_dict[key]):
                gnn_rewards_dict[key] = gnn_reward
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)
    else:
        for key_para, topo_info in dataset.items():
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)

            generate_single_data_file(key_para=key_para, topo_info=topo_info,
                                      single_data_path=args.single_data_path, ncomp=args.num_component)
            if topo_info_valid(topo_info):
                auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
                gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model,
                                                        vout_model=vout_model,
                                                        eff_vout_model=eff_vout_model, reward_model=reward_model,
                                                        cls_vout_model=cls_vout_model,
                                                        num_node=args.nnode, model_index=args.model_index,
                                                        gnn_layers=args.gnn_layers, device=device)
                gnn_reward = gnn_reward.item()
            else:
                gnn_reward = 0
            gnn_rewards_dict[key_para] = gnn_reward
            clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
                        ncomp=args.num_component)

    gnn_rewards = reordered_rewards(key_order=key_order, rewards_dict=gnn_rewards_dict)
    return gnn_rewards
