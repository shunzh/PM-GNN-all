import math

from GnnModel.get_gnn_reward import *

from arguments import get_args
from data_preprocessing import *
from metrics import *
from utils import *


def save_results_to_csv(out_file_name, result_rows):
    with open(out_file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(result_rows)
    f.close()


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


def split_dataset(data_file, number, component_num=5):
    dataset = json.load(open(data_file))
    print(len(dataset))
    each_length = int(len(dataset) / number)
    print('each ', each_length)
    splited_datasets = [{} for _ in range(math.ceil(len(dataset) / each_length))]
    t = 0
    for k, v in dataset.items():
        splited_datasets[int(t / each_length)][k] = v
        t += 1
        if t % 5000 == 0: print(t)
    total_length = 0
    for i in range(len(splited_datasets)):
        json.dump(splited_datasets[i],
                  open('./datasets/dataset_ ' + str(component_num) + ' _sub_' + str(i) + '.json', 'w'))
        total_length += len(splited_datasets[i])
        print(i)
    print(total_length)


if __name__ == '__main__':
    # ======================== Arguments ==========================#

    args = get_args()
    random.seed(1)

    # results = json.load(open("./plots/24/output_jsons_24/with_times/2022-08-08-21-46-29-1_24-time-vout-eff.json"))
    # # # save_times_to_json(times, results[0], results[1],
    # # #                    "./plots/00/output_jsons-0/2022-08-08-10-38-17-1_0-time_vout-eff.json")
    # # json_to_csv(results[0], results[1], results[2],
    # #             "./plots/00/output_jsons-0/with_times/2022-08-08-21-45-17-1_0-time-vout-eff.csv")
    # plot_with_vouts_effs(times=results[0], Vouts=results[1], effs=results[2],
    #                      eff_range=[0, 2], Vout_range=[-500, 500],
    #                      plot_file="./plots/24/output_jsons_24/with_times/2022-08-08-21-46-29-1_24-time-vout-eff")
    # plot_with_vouts_effs(times=results[0], Vouts=results[1], effs=results[2],
    #                      eff_range=[2, 8], Vout_range=[-500, 500],
    #                      plot_file="./plots/24/output_jsons_24/with_times/2022-08-08-21-46-29-1_24-time-vout-eff")
    # plot_with_vouts_effs(times=results[0], Vouts=results[1], effs=results[2],
    #                      eff_range=[8, 16], Vout_range=[-500, 500],
    #                      plot_file="./plots/24/output_jsons_24/with_times/2022-08-08-21-46-29-1_24-time-vout-eff")
    # plot_with_vouts_effs(times=results[0], Vouts=results[1], effs=results[2],
    #                      eff_range=[16, 64], Vout_range=[-500, 500],
    #                      plot_file="./plots/24/output_jsons_24/with_times/2022-08-08-21-46-29-1_24-time-vout-eff")
    # exit()

    # exit()
    split_data_set = {}
    dataset = json.load(open('./datasets/diff-eff-4.json'))
    #dataset = json.load(open('./datasets/4-comp-diff/dataset4tobesimulated.json'))
    # dataset = json.load(open('./datasets/invalid-3/json0-500.json'))
    # dataset = json.load(open('./datasets/invalid-3/json00.json'))
    # dataset = json.load(open('./datasets/invalid-3/json-1-1.json'))
    # dataset = json.load(open('./datasets/invalid-3/json-1-500.json'))

    acc_time, settle_times_set = 10, [5]
    head = []
    for i in settle_times_set:
        head.append('sim eff ' + str(i))
        head.append('sim vout ' + str(i))
    collected_reward = [head + ['eff', 'vout', 'anal eff', 'anal vout', 'time']]
    times = []

    from SimulatorModel.simulate_with_topology import get_single_sim_result_with_topo_info, \
        simulate_one_result_circuit_infos

    for count, single_data in enumerate(dataset):
        print(count)
        # if count not in [20, 24]:
        if count not in [0]:
            continue

        for key in single_data:
            performs = []
            file_sufix = ''
            for settle_times in settle_times_set:
                #list_of_node, list_of_edge, netlist, para, key_sim_effi_, target_vout=50
                reward, eff, vout, time = get_single_sim_result_with_topo_info(
                    list_of_node=single_data[key]["list_of_node"],
                    list_of_edge=single_data[key]["list_of_edge"],
                    netlist=single_data[key]["netlist"],
                    para=[single_data[key]["duty_cycle"], 10, 100],
                    key_sim_effi_={}, acc_time=acc_time,
                    settle_times=settle_times)

                # reward, eff, vout, error_msg = simulate_one_result_circuit_infos(
                #     list_of_node=single_data[key]["list_of_node"],
                #     list_of_edge=single_data[key]["list_of_edge"],
                #     file_name='one-invalid-simu',
                #     netlist=single_data[key]["netlist"],
                #     para=[single_data[key]["duty_cycle"], 10, 100],
                #     acc_time=acc_time,
                #     settle_times=settle_times)
                # visualize(list_of_node=single_data[key]["list_of_node"],
                #           list_of_edge=single_data[key]["list_of_edge"],
                #           title=0, figure_folder="datasets/invalid-3/can-not-inverse-matrix/", file_name=str(count))

                plot_simu_vout_eff(path="one_circuit-0.simu", V_in=100, rin=0.1, rout=50,
                                   plot_file=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') +
                                             str(settle_times) + '_' + str(count),
                                   title=str(eff))

                # split_times, times = get_times(path="one_circuit-0.simu", V_in=100, rin=0.1, rout=50,
                #                                plot_file=datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') +
                #                                          str(settle_times) + '_' + str(count),
                #                                title=str(eff))
                #

    #             performs.append(eff)
    #             performs.append(vout)
    #             file_sufix = file_sufix + '_' + str(settle_times)
    #         collected_reward.append(performs + [single_data[key]["eff"], single_data[key]["vout"],
    #                                             single_data[key]["eff_analytic"], single_data[key]["vout_analytic"],
    #                                             time, count, error_msg])
    #
    # save_results_to_csv(str(acc_time) + '_' + file_sufix + '_' + 'can-not-inverse-matrix-results_simu_results.csv',
    #                     collected_reward)
