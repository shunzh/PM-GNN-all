import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-sweep', action='store_true', default=True, help='sweep parameters')
    parser.add_argument('-num_component', type=int, default=3, help='raw data path')
    parser.add_argument('-transformer_eff_model_seed', type=int, default=6, help='transformer eff model seed')
    parser.add_argument('-transformer_vout_model_seed', type=int, default=4, help='transformer vout model seed')

    parser.add_argument('-path', type=str, default="./datasets", help='raw data path')
    parser.add_argument('-single_data_path', type=str, default='./datasets/single_data_datasets', help='raw data path')
    parser.add_argument('-single_data_folder', type=str, default='./datasets/single_data_datasets/2_row_dataset', help='raw data path')
    parser.add_argument('-output_file_name', type=str, default='result', help='output_file_name')

    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-n_epoch', type=int, default=100, help='number of training epoch')
    parser.add_argument('-gnn_nodes', type=int, default=20, help='number of nodes in hidden layer in GNN')
    parser.add_argument('-predictor_nodes', type=int, default=10,
                        help='number of MLP predictor nodes at output of GNN')
    parser.add_argument('-gnn_layers', type=int, default=2, help='number of layer')
    parser.add_argument('-model_index', type=int, default=1, help='model index')
    parser.add_argument('--nnode', type=int, default=7, help='number of node')

    parser.add_argument('-eff_model', type=str, default='reg_eff_4Mod_2layers_20nodes_3comp.pt', help='eff model file name')
    parser.add_argument('-vout_model', type=str, default='reg_vout_4Mod_2layers_20nodes_3comp.pt', help='vout model file name')
    parser.add_argument('-eff_vout_model', type=str, default='reg_both_3Mod_2layers_20nodes_3comp.pt', help='file of model that predicts both eff and vout')
    parser.add_argument('-reward_model', type=str, default='only_max_reg_reward_model1.pt', help='file of model that predicts both eff and vout')

    parser.add_argument('-cls_vout_model', type=str, default=None, help='eff model file name')

    parser.add_argument('-y_select', type=str, default='reg_eff', help='define target label')
    parser.add_argument('-train_rate', type=float, default=1, help='# components')


    args = parser.parse_args()

    return args
