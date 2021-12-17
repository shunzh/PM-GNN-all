import argparse
import json
import logging
import os
import random
import sys
from types import SimpleNamespace

sys.path.append(os.path.join(sys.path[0], 'TransformerGP/trans_topo_data/'))
from build_vocab import main as build_vocab_fn
from train import main as train_fn
from analysis.topoGraph import TopoGraph


def parse_json_data(data_path, training_ratio, dev_ratio, test_ratio, duty, seed=0):
    """
    Convert the dataset json to formats that can be loaded by transformer
    """
    raw_data = json.load(open(data_path, 'r'))
    data = raw_data
    """
    data = []

    for name, datum in raw_data.items():
        if datum['rout'] == 50 and datum['duty_cycle'] == duty:
            paths = TopoGraph(node_list=datum['list_of_node'],
                              edge_list=datum['list_of_edge']).find_end_points_paths_as_str()
            datum['paths'] = paths
            data.append({"name": name,
                         "paths": datum["paths"],
                         "eff": datum["eff"],
                         "vout": datum["vout"] })
    """
    data_size = len(data)
    train_data_size = int(data_size * training_ratio)
    dev_data_size = int(data_size * dev_ratio)
    test_data_size = data_size - train_data_size - dev_data_size if test_ratio == 0 else int(data_size * test_ratio)

    print('total data', data_size)
    print('training size', train_data_size)
    print('dev size', dev_data_size)
    print('test size', test_data_size)

    # randomly permuatate the data
    random.seed(seed)
    random.shuffle(data)
    data_train = data[:train_data_size]
    data_dev = data[train_data_size:train_data_size + dev_data_size]
    data_test = data[train_data_size + dev_data_size:train_data_size + dev_data_size + test_data_size]

    # get rid of the last element
    file_name = data_path.split('.')[0]

    json.dump(data, open(file_name + '_all.json', 'w'))
    json.dump(data_train, open(file_name + '_train.json', 'w'))
    json.dump(data_dev, open(file_name + '_dev.json', 'w'))
    json.dump(data_test, open(file_name + '_test.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, required=True, help='dataset json file')
    parser.add_argument('-train-ratio', type=float, default=0.6, help='proportion of data used for training (default 0.6)')
    parser.add_argument('-dev-ratio', type=float, default=0.2, help='proportion of data used for validation (default 0.2)')
    parser.add_argument('-test-ratio', type=float, default=0, help='proportion of data used for testing, (default 1 - train_ratio - dev_ratio)')
    parser.add_argument('-keep-data', default=False, action='store_true', help="don't reprocess data, enable this for multiple runs to save time")
    parser.add_argument('-duty', type=float, default=0.5, help='only use this duty cycle (default 0.5)')
    parser.add_argument('-target', type=str, default='eff', choices=['eff', 'vout'], help='training label')
    parser.add_argument('-seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('-save-model', type=str, default=None, help='save the trained model to this file, if provided')

    args = parser.parse_args()
    file_name = args.data.split('.')[0]

    logging.basicConfig(filename=file_name + '_' + args.target + '_' + str(args.seed) + '.log',
                        filemode='w',
                        level=logging.INFO)

    if args.keep_data:
        if not (os.path.exists(file_name + '_train.json') and os.path.exists(file_name + '_dev.json') and os.path.exists(file_name + '_test.json')\
                and os.path.exists(file_name + '_vocab.json')):
            raise Exception('processed files missing. remove -keep-data.')
    else:
        print('process topo data.')
        parse_json_data(data_path=args.data, training_ratio=args.train_ratio, dev_ratio=args.dev_ratio, test_ratio=args.test_ratio,
                        duty=args.duty, seed=args.seed)
        print('generate vocab file.')
        build_vocab_fn(SimpleNamespace(data_set_path=file_name + '_all.json',
                                       save_output_path=file_name + '_vocab.json',
                                       threshold=1))

    transformer_args = ['-data_train=' + file_name + '_train.json',
                        '-data_dev=' + file_name + '_dev.json',
                        '-data_test=' + file_name + '_test.json',
                        '-vocab=' + file_name + '_vocab.json',
                        '-target=' + args.target,
                        '-seed=' + str(args.seed)]
    if args.save_model:
        transformer_args.append('-save_model=' + args.save_model)

    train_fn(args=transformer_args)
