import sys

from TransformerModel.get_transformer_reward import *
from AnalyticModel.get_analytic_reward import *
from TransformerModel.get_transformer_reward import *
from GnnModel.get_gnn_reward import *

from arguments import get_args
from data_preprocessing import *
from metrics import *
from utils import *

sys.path.append(os.path.join(sys.path[0], 'TransformerModel/UCFTopo_dev'))
sys.path.append(os.path.join(sys.path[0], 'TransformerModel/transformer_SVGP'))
sys.path.append(os.path.join(sys.path[0], 'TransformerModel/trans_topo_data'))

def analyze_model(pred_rewards, gt_rewards, klist, output_folder, output_file='result.csv'):
    k_results, n_results = {}, {}
    for eval_k in [i for i in range(1, len(gt_rewards))]:
        print('getting result of threshold: ', eval_k)
        k_results[eval_k] = evaluate_top_K(preds=pred_rewards, ground_truth=gt_rewards, k=eval_k)
    write_results_to_csv_file(output_folder + 'diff_k_' + output_file, k_results)
    for eval_k in klist:
        for i in range(1, len(pred_rewards)):
            print(i, '\'s in ', len(pred_rewards))
            # print(pred_rewards[:i])
            # print(gt_rewards[:i])
            n_results[i] = evaluate_top_K(preds=pred_rewards[:i], ground_truth=gt_rewards[:i], k=eval_k)
        write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
    return k_results, n_results, pred_rewards, gt_rewards


def write_results_to_csv_file(file_name, results):
    """

    @param file_name:
    @param results: {k:{'max':max, 'mean':mean, 'std':std, 'precision', precision, 'recall', recall}}
    @return:
    """
    header = ['k', 'max', 'mean', 'std', 'precision', 'recall']
    output_for_ks = []
    index = 0
    for k, values in results.items():
        csv_list = [k]
        csv_list.extend(list(values.values()))
        output_for_ks.append(csv_list)
    with open(file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(output_for_ks)
    f.close()


if __name__ == '__main__':
    # ======================== Arguments ==========================#

    args = get_args()
    # dataset = json.load(open('./datasets/dataset_3.json'))
    # generate_statistic_rewards(sweep=False, dataset=dataset, key_order=[],
    #                                    target_vout=50)
    # print(len(dataset))
    # exit(1)
    random.seed(1)
    # for i in range(10):
    #     print(random.random())
    # dataset = json.load(open('./datasets/dataset_3.json'))
    dataset = json.load(open('./datasets/dataset_5.json'))
    split_data_set = {}

    idx = 0
    class_number = 20
    for k, v in dataset.items():
        if args.split_start <= idx < args.split_end:
            split_data_set[k] = v
        idx += 1

    results_to_save = {}

    # split_data_set = random_dic(split_data_set)
    # random_dataset_with_key(split_data_set)

    # klist = [i for i in range(1, 1 + int(len(split_data_set)))]
    klist = [i for i in [1, 3, 5, 10, 20, 50, 100, 200]]
    result_folder = './results/' + str(args.num_component) + 'comp/'
    verification_folder = result_folder + 'verification/'
    shuffled_keys = generate_keys_for_consistent_rewards(args.sweep, split_data_set)

    sim_rewards = generate_sim_rewards(sweep=args.sweep, dataset=split_data_set, key_order=shuffled_keys,
                                       target_vout=50)


    # # transformer
    transformer_rewards = generate_transformer_rewards(sweep=args.sweep, num_component=5,
                                                       eff_model_seed=args.transformer_eff_model_seed,
                                                       vout_model_seed=args.transformer_vout_model_seed,
                                                       dataset=split_data_set, key_order=shuffled_keys, target_vout=50)

    k_transformer_results, n_transformer_results, transformer_rewards, sim_rewards = \
        analyze_model(pred_rewards=transformer_rewards, gt_rewards=sim_rewards, klist=klist,
                      output_folder=result_folder, output_file='transformer.csv')

    save_verification_results(result_folder=verification_folder, model_name='transformer',
                              pred_rewards=transformer_rewards, ground_truth=sim_rewards, ground_truth_name='sim',
                              dataset_size=len(split_data_set), class_number=class_number)

    # anal
    # analytic_rewards = generate_analytic_rewards(sweep=args.sweep, dataset=split_data_set,
    #                                              key_order=shuffled_keys, target_vout=50)
    #
    # k_anal_results, n_anal_results, analytic_rewards, sim_rewards = \
    #     analyze_model(pred_rewards=analytic_rewards, gt_rewards=sim_rewards, klist=klist,
    #                   output_folder=result_folder, output_file='analytic.csv')
    #
    # save_verification_results(result_folder=verification_folder, model_name='anal',
    #                           pred_rewards=analytic_rewards, ground_truth=sim_rewards, ground_truth_name='sim',
    #                           dataset_size=len(split_data_set), class_number=class_number)

    # # gnn
    # model_list = ['only_max_reg_reward-5-0.pt', 'only_max_reg_reward-8-4.pt']
    # for model_name in model_list:
    #     args.reward_model = model_name
    #     eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model, output_file = \
    #         init_gnn_models(args=args)
    #     gnn_rewards = generate_gnn_rewards(key_order=shuffled_keys, dataset=split_data_set, sweep=args.sweep,
    #                                        args=args, eff_model=eff_model, vout_model=vout_model,
    #                                        eff_vout_model=eff_vout_model, reward_model=reward_model,
    #                                        cls_vout_model=cls_vout_model)
    #     k_gnn_results, n_gnn_results, gnn_rewards, sim_rewards = \
    #         analyze_model(pred_rewards=gnn_rewards, gt_rewards=sim_rewards, klist=klist,
    #                       output_folder=result_folder + 'gnn/', output_file=output_file)
    #     '''Be careful about the model name every time run gnn, especially the models that using eff'''
    #     save_verification_results(result_folder=verification_folder, model_name=model_name,
    #                               pred_rewards=gnn_rewards, ground_truth=sim_rewards, ground_truth_name='sim',
    #                               dataset_size=len(split_data_set), class_number=class_number)

    # note that, if we want to test the model that predict the max reward of a topology, we need to append
    # 'only_max' into the model name
    # k_gnn_results, n_gnn_results, gnn_rewards, sim_rewards = analyze_gnn(sweep=args.sweep,
    #                                                                      num_component=args.num_component,
    #                                                                      args=args,
    #                                                                      dataset=split_data_set, klist=klist,
    #                                                                      output_folder=result_folder + 'gnn/')
    #
    # save_verification_results(result_folder=verification_folder, model_name='gnn', pred_rewards=gnn_rewards,
    #                           ground_truth=sim_rewards, ground_truth_name='sim',
    #                           dataset_size=len(split_data_set), class_number=class_number)
    # test transformer

    # analytic
    # k_analytic_results, n_analytic_results, analytic_rewards, sim_rewards = analyze_analytic(sweep=args.sweep,
    #                                                                                          num_component=args.num_component,
    #                                                                                          dataset=split_data_set,
    #                                                                                          klist=klist,
    #                                                                                          output_folder=result_folder)
    # save_verification_results(result_folder=verification_folder, model_name='anal', pred_rewards=analytic_rewards,
    #                           ground_truth=sim_rewards, ground_truth_name='sim',
    #                           dataset_size=len(split_data_set), class_number=class_number)

    '''
    5 component component combination analysis
    dataset = json.load(open('./datasets/dataset_5.json'))
    dataset_statistic(dataset, 50)
    '''

    # def analyze_transformer(sweep, num_component, eff_model_seed, vout_model_seed, dataset, klist, target_vout=50,
#                         output_folder='results'):
#     args_file_name = './TransformerGP/UCFTopo_dev/config'
#     sim_configs = {}
#     transformer_rewards = []
#     sim_rewards = []
#     from TransformerModel.TransformerGP.UCFTopo_dev.utils.util import get_args
#     get_args(args_file_name, sim_configs)
#     sim = init_transformer_sim(num_component=num_component, sim_configs=sim_configs,
#                                eff_model_seed=eff_model_seed, vout_model_seed=vout_model_seed)
#     t = len(dataset)
#     if sweep:
#         transformer_sweep_rewards = {}
#         sim_sweep_data, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
#         sim_rewards = list(sim_sweep_rewards.values())
#         for key_para, topo_info in dataset.items():
#             transformer_reward, transformer_effi, transformer_vout = \
#                 metric_get_trans_prediction_from_topo_info(simulator=sim,
#                                                            list_of_node=topo_info['list_of_node'],
#                                                            list_of_edge=topo_info['list_of_edge'],
#                                                            param=[0.1, 10, 100])
#             if transformer_vout > 1:
#                 print(transformer_vout)
#             key = key_para.split('$')[0]
#             if (key not in transformer_rewards) or (transformer_reward >= transformer_rewards[key]):
#                 transformer_sweep_rewards[key] = transformer_reward
#             t -= 1
#             if t % 500 == 0:
#                 print(t, ' remaining')
#         print(len(transformer_sweep_rewards))
#         transformer_rewards = list(transformer_sweep_rewards.values())
#     else:
#
#         for key_para, topo_info in dataset.items():
#             transformer_reward, transformer_effi, transformer_vout = \
#                 metric_get_trans_prediction_from_topo_info(simulator=sim,
#                                                            list_of_node=topo_info['list_of_node'],
#                                                            list_of_edge=topo_info['list_of_edge'],
#                                                            param=[0.1, 10, 100])
#             transformer_rewards.append(transformer_reward)
#             sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
#                                                 'output_voltage': topo_info['vout']}, target_vout=target_vout)
#             sim_rewards.append(sim_reward)
#             t -= 1
#             if t % 500 == 0:
#                 print(t, ' remaining')
#     k_results, n_results = {}, {}
#     output_file = 'transformer.csv'
#     for eval_k in klist:
#         print('getting result of threshold: ', eval_k)
#         k_results[eval_k] = evaluate_top_K(transformer_rewards, sim_rewards, k=eval_k)
#     write_results_to_csv_file(output_folder + output_file, k_results)
#     for eval_k in klist:
#         for i in range(1, len(transformer_rewards)):
#             print(transformer_rewards[:i])
#             print(sim_rewards[:i])
#             n_results[i] = evaluate_top_K(transformer_rewards[:i], sim_rewards[:i], k=eval_k)
#         write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
#     return k_results, n_results, transformer_rewards, sim_rewards
#


# def analyze_gnn(sweep, num_component, args, dataset, klist, target_vout=50, output_folder='./results/'):
#     '''
#
#     @param sweep:
#     @param num_component:
#     @param args:
#     @param dataset:
#     @return:
#     '''
#     args = get_args()
#
#     batch_size = args.batch_size
#     n_epoch = args.n_epoch
#
#     # ======================== Data & Model ==========================#
#     nf_size = 4
#     ef_size = 3
#     # single_data_folder = './datasets/single_data_datasets/2_row_dataset'
#     # single_data_path = './datasets/single_data_datasets'
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     eff_model, vout_model, eff_vout_model, reward_model, cls_vout_model = None, None, None, None, None
#     if args.reward_model is not None:
#         # if this argument is set, load one model that predicts both eff and vout
#         reward_model_state_dict, _ = torch.load(args.reward_model)
#
#         reward_model = initialize_model(model_index=args.model_index,
#                                         gnn_nodes=args.gnn_nodes,
#                                         gnn_layers=args.gnn_layers,
#                                         pred_nodes=args.predictor_nodes,
#                                         nf_size=nf_size,
#                                         ef_size=ef_size,
#                                         device=device,
#                                         output_size=1)  # need to set output size of the network here
#         reward_model.load_state_dict(reward_model_state_dict)
#         output_file = args.reward_model.replace('.pt', '.csv')
#     elif args.eff_vout_model is not None:
#         # if this argument is set, load one model that predicts both eff and vout
#         # model_state_dict, data_loader = torch.load(args.eff_vout_model)
#         eff_vout_model_state_dict, _ = torch.load(args.eff_vout_model)
#
#         eff_vout_model = initialize_model(model_index=args.model_index,
#                                           gnn_nodes=args.gnn_nodes,
#                                           gnn_layers=args.gnn_layers,
#                                           pred_nodes=args.predictor_nodes,
#                                           nf_size=nf_size,
#                                           ef_size=ef_size,
#                                           device=device,
#                                           output_size=2)  # need to set output size of the network here
#         eff_vout_model.load_state_dict(eff_vout_model_state_dict)
#         output_file = args.eff_vout_model.replace('.pt', '.csv')
#
#
#
#
#
#     elif args.cls_vout_model is not None:
#         # if this argument is set, load one model that predicts both eff and vout
#         cls_vout_model_state_dict, _ = torch.load(args.cls_vout_model)
#         cls_vout_model = initialize_model(model_index=args.model_index,
#                                           gnn_nodes=args.gnn_nodes,
#                                           gnn_layers=args.gnn_layers,
#                                           pred_nodes=args.predictor_nodes,
#                                           nf_size=nf_size,
#                                           ef_size=ef_size,
#                                           device=device)
#         cls_vout_model.load_state_dict(cls_vout_model_state_dict)
#
#         eff_model_state_dict, data_loader = torch.load(args.eff_model)
#         eff_model = initialize_model(model_index=args.model_index,
#                                      gnn_nodes=args.gnn_nodes,
#                                      gnn_layers=args.gnn_layers,
#                                      pred_nodes=args.predictor_nodes,
#                                      nf_size=nf_size,
#                                      ef_size=ef_size,
#                                      device=device)
#         eff_model.load_state_dict(eff_model_state_dict)
#         output_file = args.cls_vout_model.replace('.pt', '.csv')
#
#     else:
#         vout_model_state_dict, _ = torch.load(args.vout_model)
#         vout_model = initialize_model(model_index=args.model_index,
#                                       gnn_nodes=args.gnn_nodes,
#                                       gnn_layers=args.gnn_layers,
#                                       pred_nodes=args.predictor_nodes,
#                                       nf_size=nf_size,
#                                       ef_size=ef_size,
#                                       device=device)
#         vout_model.load_state_dict(vout_model_state_dict)
#
#         eff_model_state_dict, data_loader = torch.load(args.eff_model)
#         eff_model = initialize_model(model_index=args.model_index,
#                                      gnn_nodes=args.gnn_nodes,
#                                      gnn_layers=args.gnn_layers,
#                                      pred_nodes=args.predictor_nodes,
#                                      nf_size=nf_size,
#                                      ef_size=ef_size,
#                                      device=device)
#         eff_model.load_state_dict(eff_model_state_dict)
#         output_file = args.eff_model.replace('.pt', '.csv')
#
#     gnn_rewards = []
#     sim_rewards = []
#     t = len(dataset)
#     if sweep:
#         sim_sweep_data, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
#         sim_rewards = list(sim_sweep_rewards.values())
#         gnn_sweep_rewards = {}
#         # note that, if we want to test the model that predict the max reward of a topology, we need to append
#         # 'only_max' into the model name
#         if (args.reward_model is not None) and ('only_max' in args.reward_model):
#             gnn_dataset, _ = generate_dataset_for_gnn_max_reward_prediction(dataset=dataset, target_vout=50)
#         else:
#             gnn_dataset = dataset
#
#         for key_para, topo_info in gnn_dataset.items():
#             clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
#                         ncomp=args.num_component)
#             generate_single_data_file(key_para=key_para, topo_info=topo_info,
#                                       single_data_path=args.single_data_path, ncomp=args.num_component)
#             if topo_info_valid(topo_info=topo_info):
#                 # if nn_node change, this must be changed too
#                 auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
#                 gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model,
#                                                         vout_model=vout_model,
#                                                         eff_vout_model=eff_vout_model, reward_model=reward_model,
#                                                         cls_vout_model=cls_vout_model,
#                                                         num_node=args.nnode, model_index=args.model_index,
#                                                         gnn_layers=args.gnn_layers, device=device)
#                 gnn_reward = gnn_reward.item()
#             else:
#                 gnn_reward = 0
#             key = key_para.split('$')[0]
#             if (key not in gnn_sweep_rewards) or (gnn_reward >= gnn_sweep_rewards[key]):
#                 gnn_sweep_rewards[key] = gnn_reward
#             clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
#                         ncomp=args.num_component)
#             t -= 1
#             if t % 500 == 0:
#                 print(t, ' remaining')
#         print(len(gnn_sweep_rewards))
#         gnn_rewards = list(gnn_sweep_rewards.values())
#
#     else:
#         for key_para, topo_info in dataset.items():
#             clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
#                         ncomp=args.num_component)
#
#             generate_single_data_file(key_para=key_para, topo_info=topo_info,
#                                       single_data_path=args.single_data_path, ncomp=args.num_component)
#             if topo_info_valid(topo_info):
#                 auto_dataset = Autopo(args.single_data_folder, args.single_data_path, args.y_select, args.num_component)
#                 gnn_reward = get_gnn_single_data_reward(dataset=auto_dataset, eff_model=eff_model,
#                                                         vout_model=vout_model,
#                                                         eff_vout_model=eff_vout_model, reward_model=reward_model,
#                                                         cls_vout_model=cls_vout_model,
#                                                         num_node=args.nnode, model_index=args.model_index,
#                                                         gnn_layers=args.gnn_layers, device=device)
#                 gnn_reward = gnn_reward.item()
#             else:
#                 gnn_reward = 0
#
#             gnn_rewards.append(gnn_reward)
#             sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
#                                                 'output_voltage': topo_info['vout']}, target_vout=target_vout)
#             sim_rewards.append(sim_reward)
#             clear_files(save_data_folder=args.single_data_folder, raw_data_folder=args.single_data_path,
#                         ncomp=args.num_component)
#             t -= 1
#             if t % 500 == 0:
#                 print(t, ' remaining')
#     results = {}
#     k_results, n_results = {}, {}
#     for eval_k in klist:
#         k_results[eval_k] = evaluate_top_K(gnn_rewards, sim_rewards, k=eval_k)
#     write_results_to_csv_file(output_folder + 'diff_k_' + output_file, k_results)
#     for eval_k in klist:
#         for i in range(1, len(gnn_rewards)):
#             print(gnn_rewards[:i])
#             print(sim_rewards[:i])
#             n_results[i] = evaluate_top_K(gnn_rewards[:i], sim_rewards[:i], k=eval_k)
#         write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
#     return k_results, n_results, gnn_rewards, sim_rewards


# def analyze_gnn(gnn_rewards, sim_rewards, klist, output_folder='./results/', output_file=None):
#     k_results, n_results = {}, {}
#     for eval_k in klist:
#         k_results[eval_k] = evaluate_top_K(gnn_rewards, sim_rewards, k=eval_k)
#     write_results_to_csv_file(output_folder + 'diff_k_' + output_file, k_results)
#     for eval_k in klist:
#         for i in range(1, len(gnn_rewards)):
#             print(gnn_rewards[:i])
#             print(sim_rewards[:i])
#             n_results[i] = evaluate_top_K(gnn_rewards[:i], sim_rewards[:i], k=eval_k)
#         write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
#     return k_results, n_results, gnn_rewards, sim_rewards


# def analyze_analytic(sweep, ):


# def analyze_analytic(sweep, num_component, dataset, klist, target_vout=50, output_folder='results'):
#     analytic_rewards = []
#     sim_rewards = []
#     outer_expression_dict = json.load(open('no-sweep-analytic-expression.json'))
#     # outer_expression_dict = correct_expression_dict(outer_expression_dict)
#     # key+$+'[C,L]': {'Expression': expression,'duty cycle':{'Efficiency':efficiency, 'Vout':vout}}
#     pre_expression_length = len(outer_expression_dict)
#     if sweep:
#         anal_sweep_data, anal_sweep_rewards, outer_expression_dict = \
#             generate_anal_sweep_prediction(dataset, target_vout, outer_expression_dict)
#         analytic_rewards = list(anal_sweep_rewards.values())
#
#         _, sim_sweep_rewards = generate_sweep_dataset(dataset=dataset, target_vout=50)
#         sim_rewards = list(sim_sweep_rewards.values())
#     else:
#         anal_sweep_rewards, outer_expression_dict = \
#             generate_anal_not_sweep_prediction(dataset, target_vout, outer_expression_dict)
#
#         for key_para, topo_info in dataset.items():
#             sim_reward = calculate_reward(effi={'efficiency': topo_info['eff'],
#                                                 'output_voltage': topo_info['vout']}, target_vout=target_vout)
#             sim_rewards.append(sim_reward)
#     # rewrite expression dict to json
#     if len(outer_expression_dict) > pre_expression_length:
#         with open('no-sweep-analytic-expression.json', 'w') as f:
#             json.dump(outer_expression_dict, f)
#             f.close()
#     k_results, n_results = {}, {}
#     output_file = 'analytic.csv'
#     for eval_k in klist:
#         print('getting result of threshold: ', eval_k)
#         k_results[eval_k] = evaluate_top_K(analytic_rewards, sim_rewards, k=eval_k)
#     write_results_to_csv_file(output_folder + output_file, k_results)
#     for eval_k in klist:
#         for i in range(1, len(analytic_rewards)):
#             print(analytic_rewards[:i])
#             print(sim_rewards[:i])
#             n_results[i] = evaluate_top_K(analytic_rewards[:i], sim_rewards[:i], k=eval_k)
#         write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
#     return k_results, n_results, analytic_rewards, sim_rewards


#
# def analyze_transformer(transformer_rewards, sim_rewards, klist, output_folder, output_file='transformer.csv'):
#     k_results, n_results = {}, {}
#     for eval_k in klist:
#         print('getting result of threshold: ', eval_k)
#         k_results[eval_k] = evaluate_top_K(transformer_rewards, sim_rewards, k=eval_k)
#     write_results_to_csv_file(output_folder + output_file, k_results)
#     for eval_k in klist:
#         for i in range(1, len(transformer_rewards)):
#             print(transformer_rewards[:i])
#             print(sim_rewards[:i])
#             n_results[i] = evaluate_top_K(transformer_rewards[:i], sim_rewards[:i], k=eval_k)
#         write_results_to_csv_file(output_folder + 'diff_n_' + str(eval_k) + '_' + output_file, n_results)
#     return k_results, n_results, transformer_rewards, sim_rewards
