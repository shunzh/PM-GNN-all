import os.path as osp
import json

from torch_geometric.data import InMemoryDataset, Data

from torch.utils.data import DataLoader
from torch_geometric.data import Data, DataLoader
import torch

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from decimal import Decimal
from reward_fn import compute_reward

import torch_geometric


class Autopo(InMemoryDataset):

    def __init__(self, root, path, y_select, ncomp, transform=None, pre_transform=None, data_path_root=None):
        self.data_path_root = path
        self.y_select = y_select
        self.ncomp = ncomp
        super(Autopo, self).__init__(root, transform, pre_transform)
        # tmp = self.get_tmp()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.json']

    @property
    def processed_file_names(self):
        return ['data.pt']

    # process sample
    def get_tmp(self):

        print("get_tmp running")
        y_select = self.y_select
        ncomp = self.ncomp

        if ncomp == 3 or ncomp == 5:
            json_file = json.load(open(self.data_path_root + "/dataset" + "_" + str(ncomp) + ".json"))
        else:
            json_file = json.load(open(self.data_path_root + "/dataset_5-simu.json"))

        tmp = {}

        max_nodes = 7

        empty_node = [0, 0, 0, 0]
        empty_edge = [0, 0, 0]
        empty_edge0 = [0, 0, 0, 0, 0, 0]

        nn = 0

        for item in json_file:

            file_name = item

            list_of_edge = json_file[item]["list_of_edge"]
            list_of_node = json_file[item]["list_of_node"]

            edge_attr = json_file[item]["edge_attr"]
            node_attr = json_file[item]["node_attr"]
            edge_attr0 = json_file[item]["edge_attr0"]

            netlist = json_file[item]["netlist"]

            nn = nn + 1
            # print(nn)
            # if nn>2:
            #    break

            # print(file_name)
            # print(list_of_node)
            # print(list_of_edge)

            if json_file[item]["vout"] / 100 > 1 or json_file[item]["vout"] / 100 < 0:
                # print(json_file[item])
                continue

            target_vout = []
            target_eff = []
            target_rewards = []

            analytic_vout = []
            analytic_eff = []
            try:
                analytic_eff.append(float(json_file[item]["eff_analytic"]))
            except:
                analytic_eff.append(0)
            label_analytic_eff = analytic_eff
            try:
                analytic_vout.append(float(json_file[item]["vout_analytic"]) / 100)
            except:
                analytic_vout.append(0)
            label_analytic_vout = analytic_vout

            eff = json_file[item]["eff"]
            vout = json_file[item]["vout"] / 100
            r = compute_reward(eff, vout)

            target_eff.append(eff)
            target_vout.append(vout)
            target_rewards.append(r)

            if y_select == 'reg_eff':
                label = target_eff

            elif y_select == 'reg_vout':
                label = target_vout

            elif y_select == 'reg_reward':
                label = target_rewards

            elif y_select == 'reg_both':
                label = [[eff, vout]]

            elif y_select == 'cls_boost':
                target_vout = [float(json_file[item]["vout"] > 110)]
                label = target_vout

            elif y_select == 'cls_buck':
                target_vout = []
                # target_vout.append(float(json_file[item]["vout"] / 100))
                temp = float(json_file[item]["vout"])
                if temp < 30:
                    target_vout.append(0)
                elif temp < 50:
                    target_vout.append((temp - 30) / 20)
                elif temp < 70:
                    target_vout.append((70 - temp) / 20)
                else:
                    target_vout.append(0)
                label = target_vout

            else:
                print("Wrong select input")
                continue

            if json_file[item]["vout"] == -1:
                continue

            if json_file[item]["eff"] < 0 or json_file[item]["eff"] > 1:
                # print(json_file[item])
                continue

            rout = json_file[item]["rout"]
            cout = json_file[item]["cout"]
            freq = json_file[item]["freq"]
            duty_cycle = json_file[item]["duty_cycle"]

            tmp_list_of_edge = []

            list_of_node_name = []
            list_of_edge_name = []

            for node in node_attr:
                list_of_node_name.append(node)

            for edge in edge_attr:
                list_of_edge_name.append(edge)

            for edge in list_of_edge:
                tmp_list_of_edge.append(edge[:])

            # print("edge_attr:",edge_attr)
            # print(list_of_node_name)
            # print(list_of_edge_name)
            # print(tmp_list_of_edge)

            node_to_delete = []
            node_to_replace = []

            for edge in tmp_list_of_edge:
                if edge[0] not in list_of_edge_name:
                    node_to_delete.append(str(edge[1]))
                    node_to_replace.append(edge[0])

            list_of_edge_new = []
            for edge in tmp_list_of_edge:
                if str(edge[1]) in node_to_delete:
                    index = node_to_delete.index(str(edge[1]))
                    edge[1] = node_to_replace[index]
                if edge[0] not in node_to_replace:
                    list_of_edge_new.append(edge[:])

            list_of_node_new = []

            for node in list_of_node_name:
                if node not in node_to_delete:
                    list_of_node_new.append(node)

            # print(list_of_node_name)
            # print(list_of_node_new)

            # print(list_of_edge_new)

            node_attr_new = []
            for node in list_of_node_new:
                node_attr_new.append(node_attr[node][:])

            edge_start = []
            edge_end = []

            edge_attr_new0 = []
            edge_attr_new1 = []
            edge_attr_new2 = []

            for e1 in edge_attr:
                counter = 0
                start = -1
                end = -1
                for e2 in list_of_edge_new:
                    if e2[0] == e1 and counter == 0:
                        counter = counter + 1
                        start = list_of_node_new.index(str(e2[1]))
                    if e2[0] == e1 and counter == 1:
                        end = list_of_node_new.index(str(e2[1]))
                if start == -1 or end == -1:
                    continue
                edge_start.append(start)
                edge_start.append(end)
                edge_end.append(end)
                edge_end.append(start)
                edge_attr_new0.append(edge_attr0[e1])
                edge_attr_new0.append(edge_attr0[e1])

                if e1[0] != 'S':
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new2.append(edge_attr[e1])
                    edge_attr_new2.append(edge_attr[e1])

                else:
                    edge_attr_new1.append(edge_attr[e1])
                    edge_attr_new1.append(edge_attr[e1])
                    tmp_name = ''
                    if e1[:2] == 'Sa':
                        tmp_name = 'Sb0'
                    else:
                        tmp_name = 'Sa0'
                    edge_attr_new2.append(edge_attr[tmp_name])
                    edge_attr_new2.append(edge_attr[tmp_name])

            edge_attr_new1.append([0, 1 / cout, 0])
            edge_attr_new1.append([0, 1 / cout, 0])
            edge_attr_new2.append([0, 1 / cout, 0])
            edge_attr_new2.append([0, 1 / cout, 0])
            edge_attr_new1.append([1 / rout, 0, 0])
            edge_attr_new1.append([1 / rout, 0, 0])
            edge_attr_new2.append([1 / rout, 0, 0])
            edge_attr_new2.append([1 / rout, 0, 0])

            edge_attr_new0.append([0, 0, cout, 0, 0, 0])
            edge_attr_new0.append([0, 0, cout, 0, 0, 0])
            edge_attr_new0.append([0, 0, 0, 0, rout, 0])
            edge_attr_new0.append([0, 0, 0, 0, rout, 0])

            VOUT_index = list_of_node_new.index('VOUT')
            GND_index = list_of_node_new.index('GND')
            VIN_index = list_of_node_new.index('VIN')

            edge_start.append(VOUT_index)
            edge_start.append(GND_index)
            edge_end.append(GND_index)
            edge_end.append(VOUT_index)
            edge_start.append(VOUT_index)
            edge_start.append(GND_index)
            edge_end.append(GND_index)
            edge_end.append(VOUT_index)

            edge_index = []
            edge_index = [edge_start, edge_end]

            tmp[file_name] = {}

            node_feature_size = len(empty_node)
            edge_feature_size = len(empty_edge)
            edge0_feature_size = len(empty_edge0)

            edge_attr1_padded = np.zeros((max_nodes, max_nodes, edge_feature_size))
            edge_attr2_padded = np.zeros((max_nodes, max_nodes, edge_feature_size))
            node_attr_padded = np.zeros((max_nodes, node_feature_size))
            edge_attr0_padded = np.zeros((max_nodes, max_nodes, edge0_feature_size))

            n = len(node_attr_new)

            node_attr_padded[:n, :] = node_attr_new

            edge_attr0 = \
                torch_geometric.utils.to_dense_adj(torch.tensor(edge_index), None, torch.tensor(edge_attr_new0),
                                                   len(node_attr_new))[0]

            edge_attr1 = \
                torch_geometric.utils.to_dense_adj(torch.tensor(edge_index), None, torch.tensor(edge_attr_new1),
                                                   len(node_attr_new))[0]

            edge_attr2 = \
                torch_geometric.utils.to_dense_adj(torch.tensor(edge_index), None, torch.torch.tensor(edge_attr_new2),
                                                   len(node_attr_new))[0]

            r = edge_attr1.shape[0]
            c = edge_attr1.shape[1]
            edge_attr0_padded[:r, :r, :] = edge_attr0
            edge_attr1_padded[:r, :r, :] = edge_attr1
            edge_attr2_padded[:r, :r, :] = edge_attr2

            adjacent_matrix = np.zeros((max_nodes, max_nodes))

            for i in range(max_nodes):
                for j in range(max_nodes):
                    if not all(v == 0 for v in edge_attr1_padded[i, j]):
                        adjacent_matrix[i, j] = 1

            tmp[file_name]['node_attr'] = node_attr_padded
            tmp[file_name]['edge0_attr'] = edge_attr0_padded
            tmp[file_name]['edge1_attr'] = edge_attr1_padded
            tmp[file_name]['edge2_attr'] = edge_attr2_padded
            tmp[file_name]['adjacent_matrix'] = adjacent_matrix
            tmp[file_name]['label'] = label

            tmp[file_name]['sim_eff'] = target_eff
            tmp[file_name]['sim_vout'] = target_vout
            tmp[file_name]['analytic_eff'] = label_analytic_eff
            tmp[file_name]['analytic_vout'] = label_analytic_vout

            tmp[file_name]['list_of_node'] = list_of_node
            tmp[file_name]['list_of_edge'] = list_of_edge
            tmp[file_name]['netlist'] = netlist
            tmp[file_name]['duty_cycle'] = duty_cycle

        return tmp

    def process(self):

        print("process running")

        count = 0

        tmp = self.get_tmp()
        data_list = []
        for fn in tmp:
            node_attr = torch.tensor(tmp[fn]["node_attr"], dtype=torch.float)
            edge0_attr = torch.tensor(tmp[fn]["edge0_attr"], dtype=torch.float)
            edge1_attr = torch.tensor(tmp[fn]["edge1_attr"], dtype=torch.float)
            edge2_attr = torch.tensor(tmp[fn]["edge2_attr"], dtype=torch.float)
            adj = torch.tensor(tmp[fn]["adjacent_matrix"], dtype=torch.float)
            label = torch.tensor(tmp[fn]["label"], dtype=torch.float)
            analytic_eff = torch.tensor(tmp[fn]["analytic_eff"], dtype=torch.float)
            analytic_vout = torch.tensor(tmp[fn]["analytic_vout"], dtype=torch.float)
            sim_eff = torch.tensor(tmp[fn]["sim_eff"], dtype=torch.float)
            sim_vout = torch.tensor(tmp[fn]["sim_vout"], dtype=torch.float)
            duty_cycle = torch.tensor(tmp[fn]["duty_cycle"], dtype=torch.float)
            # list_of_node = torch.tensor(tmp[fn]["list_of_node"])
            # list_of_edge = torch.tensor(tmp[fn]["list_of_edge"],dtype=torch.CharTensor)
            # netlist = torch.tensor(tmp[fn]["netlist"],dtype=torch.CharTensor)

            #            print(node_attr)
            #            print(edge1_attr)
            data = Data(node_attr=node_attr, edge0_attr=edge0_attr, edge1_attr=edge1_attr, edge2_attr=edge2_attr,
                        adj=adj, label=label,
                        analytic_eff=analytic_eff, analytic_vout=analytic_vout,
                        sim_eff=sim_eff, sim_vout=sim_vout, duty_cycle=duty_cycle)
            # list_of_node=list_of_node, list_of_edge=list_of_edge, netlist=netlist
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def split_balance_data(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    # print("train_ratio", train_ratio)
    # print("val_ratio", val_ratio)
    # print("test_ratio", test_ratio)

    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    n_train = int(dataset_size * train_ratio)
    n_val = int(dataset_size * val_ratio)
    n_test = int(dataset_size * test_ratio)

    train_indices, val_indices, test_indices = indices[:n_train], indices[n_train + 1:n_train + n_val], indices[
                                                                                                        n_train + n_val + 1:n_train + n_val + n_test]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def split_imbalance_data(dataset, batch_size, rtrain, rval, rtest):
    train_ratio = rtrain
    val_ratio = rval
    test_ratio = rtest
    # print("train_ratio", train_ratio)
    # print("val_ratio", val_ratio)
    # print("test_ratio", test_ratio)

    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    ind_positive = []
    ind_negative = []
    ind = 0

    for data in dataset:
        # print(data)
        # print("1:", data['analytic_vout'].tolist())
        # print("2:", data['sim_vout'].tolist())
        flag_cls = data['label'].tolist()[0]
        # print("flag_cls",flag_cls)
        if 0.3 < flag_cls < 0.7:
            ind_positive.append(ind)
            # print(ind_positive)
        else:
            ind_negative.append(ind)
        # 通过判断eff是否在0.3-0.7
        # 判断vout是否在0.1-0.9之间
        ind += 1
    indices_new = []

    # for i in range(int(len(ind_negative) / len(ind_positive))):
    #     indices_new.extend(ind_positive)

    # for i in range(5-int(len(ind_positive) / len(ind_negative)*100)):
    #     print(5-int(len(ind_positive) / len(ind_negative)*100))
    #     indices_new.extend(ind_positive)

    A = len(ind_positive)
    B = len(ind_negative)
    a = B / A / 4 - 1
    i = int(a * A)
    positive_percentage = A / (A + B)
    # a = Decimal(a).quantize(Decimal('0.0'))
    print("percent: ", positive_percentage)
    print("A", A)
    print("B", B)
    print("range", round((B - 4 * A) / 4))

    if positive_percentage > 0.2:
        indices_new.extend(list(np.random.choice(ind_positive, int((-a) * A))))
        indices_new.extend(ind_negative)
    else:
        if a < 0.5:
            indices_new.extend(list(np.random.choice(ind_positive, int(a * A))))
        else:
            for i in range(round(a)):
                indices_new.extend(ind_positive)
            indices_new.extend(ind_positive)
            indices_new.extend(ind_negative)
    # for i in range(len(list(np.random.choice(ind_positive, int(a * A))))):
    #     indices_new.append(list(np.random.choice(ind_positive, int(a * A)))[i])

    # indices_new.extend(list(np.random.choice(ind_positive, i)))

    print("new positve percentage: ", (len(indices_new) - dataset_size) / len(indices_new))

    dataset_size_new = len(indices_new)
    # print("dataset_size_new: ", dataset_size_new)
    # print("dataset_size: ", dataset_size)

    n_train = int(dataset_size_new * train_ratio)
    n_val = int(dataset_size_new * val_ratio)
    n_test = int(dataset_size_new * test_ratio)
    # print("n_train", type(n_train))
    # print("n_val", n_val)
    # print("n_test", n_test)

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices_new)

    train_indices, val_indices, test_indices = indices_new[:n_train], indices_new[n_train + 1:n_train + n_val], \
                                               indices_new[n_train + n_val + 1:n_train + n_val + n_test]
    # print("train_indeces: ",type(train_indices))
    # print("valid_indeces: ",type(val_indices))
    # print("test_indeces: ", len(test_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
