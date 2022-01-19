import argparse
import json
import random

from analysis.topoGraph import TopoGraph


def parse_json_data(data_path, training_ratio, dev_ratio, seed=0):
    """
    Convert the dataset json to formats that can be loaded by transformer
    """
    raw_data = json.load(open(data_path, 'r'))
    data = []

    for name, datum in raw_data.items():
        if datum['rout'] == 50 and datum['duty_cycle'] == 0.5:
            paths = TopoGraph(node_list=datum['list_of_node'],
                              edge_list=datum['list_of_edge']).find_end_points_paths_as_str()
            datum['paths'] = paths
            data.append({"name": name, "node_list": datum['list_of_node'], "edge_list": datum['list_of_edge'],
                         "paths": datum["paths"], "eff": datum["eff"], "vout": datum["vout"] })

    data_size = len(data)
    train_data_size = int(data_size * training_ratio)
    dev_data_size = int(data_size * dev_ratio)

    print('total data', data_size)
    print('training size', train_data_size)
    print('dev size', dev_data_size)

    # randomly permuatate the data
    random.seed(args.seed)
    random.shuffle(data)
    data_train = data[:train_data_size]
    data_dev = data[train_data_size:train_data_size + dev_data_size]
    data_test = data[train_data_size + dev_data_size:]

    # get rid of the last element
    file_name = data_path.split('.')[0]

    json.dump(data, open(file_name + '_all.json', 'w'))
    json.dump(data_train, open(file_name + '_train.json', 'w'))
    json.dump(data_dev, open(file_name + '_dev.json', 'w'))
    json.dump(data_test, open(file_name + '_test.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default="")
    parser.add_argument('-train', type=float, default=0.6)
    parser.add_argument('-dev', type=float, default=0.2)
    parser.add_argument('-seed', type=int, default=0, help='random seed (default: 0)')

    args = parser.parse_args()
    parse_json_data(data_path=args.data, training_ratio=args.train, dev_ratio=args.dev, seed=args.seed)
