from abc import abstractmethod, ABC
from copy import deepcopy

import config

from UCFTopo_dev.ucts.TopoPlanner import TopoGenSimulator, calculate_reward, sort_dict_string

from trans_topo_data.analysis.topoGraph import TopoGraph
from trans_topo_data.analysis.graphUtils import nodes_and_edges_to_adjacency_matrix


class SurrogateRewardTopologySim(TopoGenSimulator, ABC):
    def __init__(self, *args):

        # for fair comparison with simulator, create a hash table to save surrogate rewards here
        self.surrogate_hash_table = {}
        # for queried topologies, use this hash table rather than using surrogate model
        self.queried_hash_table = {}

        super().__init__(*args)

    def find_paths(self, state=None):
        """
        Useful for GP and transformer based surrogate model
        Return the list of paths in the current state
        e.g. ['VIN - inductor - VOUT', ...]
        """
        if state is None:
            state = self.get_state()

        node_list, edge_list = state.get_nodes_and_edges()

        adjacency_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)

        # convert graph to paths, and find embedding
        topo = TopoGraph(adj_matrix=adjacency_matrix, node_list=node_list)
        return topo.find_end_points_paths_as_str()

    def find_paths_with_topo_info(self, node_list, edge_list, state=None):
        """
        Useful for GP and transformer based surrogate model with list of node and edge
        Return the list of paths in the current state
        e.g. ['VIN - inductor - VOUT', ...]
        """
        if state is None:
            state = self.get_state()

        # node_list, edge_list = state.get_nodes_and_edges()

        adjacency_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)

        # convert graph to paths, and find embedding
        topo = TopoGraph(adj_matrix=adjacency_matrix, node_list=node_list)
        return topo.find_end_points_paths_as_str()

    def get_topo_key(self, state=None):
        """
        the key of topology used by hash table

        :return:  the key representation of the state (self.current if state == None)
        """
        if state is None:
            state = self.get_state()

        topo_key = sort_dict_string(state.graph)

        return topo_key

    def get_reward(self, state=None, node_list=None, edge_list=None):
        """
        Use surrogate reward function
        imp-wise, not sure why keeping a reward attribute
        """
        if state is not None:
            self.set_state(state)

        if not self.current.graph_is_valid():
            self.reward = 0
            return self.reward

        topo_key = self.get_topo_key()

        if topo_key in self.queried_hash_table:
            # when this topology is queried, now return ground-truth values
            return self.queried_hash_table[topo_key][0]
        elif topo_key in self.surrogate_hash_table:
            # return surrogate rewards that exist in the surrogate hash table
            self.hash_counter += 1

            return self.surrogate_hash_table[topo_key][0]
        else:
            # in this case, run the surrogate model
            self.query_counter += 1

            if node_list and edge_list:
                eff = self.get_surrogate_eff_with_topo_info(node_list=node_list, edge_list=edge_list)
                vout = self.get_surrogate_vout_with_topo_info(node_list=node_list, edge_list=edge_list)
            else:
                eff = self.get_surrogate_eff()
                vout = self.get_surrogate_vout()

            # an object for computing reward
            eff_obj = {'efficiency': eff,
                       'output_voltage': vout}
            self.reward = calculate_reward(eff_obj)

            self.surrogate_hash_table[topo_key] = [self.reward, eff, vout]
            self.seen_state_list.append(deepcopy(self.current))

        return self.reward

    def update_queried_state(self, state, true_reward, true_eff, true_vout):
        """
        when a state is queried, save it to the queried hash table so can be sued later
        """
        topo_key = self.get_topo_key(state)
        self.queried_hash_table[topo_key] = (true_reward, true_eff, true_vout)

    @abstractmethod
    def get_surrogate_eff(self, state=None):
        """
        return the mean of eff of state, use self.get_state() if state is None
        """
        pass

    @abstractmethod
    def get_surrogate_vout(self, state=None):
        """
        return the mean of vout of state, use self.get_state() if state is None
        """
        pass

    def get_surrogate_eff_with_topo_info(self, node_list, edge_list, state=None):
        """
        return the mean of eff of state, use self.get_state() if state is None
        """
        pass

    @abstractmethod
    def get_surrogate_vout_with_topo_info(self, node_list, edge_list, state=None):
        """
        return the mean of vout of state, use self.get_state() if state is None
        """
        pass

    @abstractmethod
    def get_surrogate_eff_std(self, state=None):
        """
        return the standard deviation of eff of state, useful for active learning
        """
        pass

    @abstractmethod
    def get_surrogate_vout_std(self, state=None):
        """
        return the standard deviation of vout of state, useful for active learning
        """
        pass

    def get_true_performance(self, state=None):
        """
        :return: [reward, eff, vout]
        """
        if state is not None:
            self.set_state(state)

        if not self.current.graph_is_valid():
            self.reward = 0
        else:
            hash = self.get_topo_key()

            # if not in hash table, call ngspice
            if not hash in self.graph_2_reward.keys():
                # compute true reward and save in graph_2_reward
                super().get_reward()

            return self.graph_2_reward[hash]

    def get_true_reward(self, state=None):
        if state is not None:
            self.set_state(state)

        return super().get_reward()

    def isomorphize_state_list(self, state_list):
        """
        Clean same topologies by path list. Useful for active learning selection.
        """
        paths_list = set()
        cleaned_states = []

        for state in state_list:
            # only consider potentially valid topos
            if state.graph_is_valid():
                self.set_state(state)
                paths = tuple(self.find_paths())

                if paths not in paths_list:
                    paths_list.add(paths)
                    cleaned_states.append(state)

        return cleaned_states
