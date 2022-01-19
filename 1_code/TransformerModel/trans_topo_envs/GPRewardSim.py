import torch

from algs.gp import GPModel
from TransformerModel.trans_topo_envs.surrogateRewardSim import SurrogateRewardTopologySim

from topo_data.analysis.embedding import tf_embed

import numpy as np

from TransformerModel.trans_topo_envs.surrogateRewardSimFactory import SurrogateRewardSimFactory


class GPRewardTopologySimFactory(SurrogateRewardSimFactory):
    def __init__(self, eff_file, vout_file):
        eff_gp, vec_of_paths = self.load_gp_model(eff_file)
        vout_gp, vec_of_paths = self.load_gp_model(vout_file)

        # init is called later
        self.eff_gp = eff_gp
        self.vout_gp = vout_gp
        self.vec_of_paths = vec_of_paths

    def load_gp_model(self, filename, debug=False):
        data = torch.load(filename)
        train_x = data['train_x']
        train_y = data['train_y']
        state_dict = data['model_state_dict']
        vec_of_paths = data['vec_of_paths']

        gp = GPModel(train_x, train_y, state_dict=state_dict)
        # check_gp(self.reward_model, train_x[:20], train_y[:20])

        if debug:
            print('check gp on training data')
            for idx in range(20):
                print(gp.get_mean(train_x[idx]), train_y[idx])

        return gp, vec_of_paths

    def get_sim_init(self):
        return lambda *args: GPRewardTopologySim(self.eff_gp, self.vout_gp, self.vec_of_paths, *args)


class GPRewardTopologySim(SurrogateRewardTopologySim):
    def __init__(self, eff_gp, vout_gp, vec_of_paths, *args):
        super().__init__(*args)

        # init is called later
        self.eff_gp = eff_gp
        self.vout_gp = vout_gp
        self.vec_of_paths = vec_of_paths

    def get_surrogate_eff(self, state=None):
        if state is not None:
            self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        eff = np.clip(self.eff_gp.get_mean(embedding), 0., 1.)

        return eff

    def get_surrogate_vout(self, state=None):
        if state is not None:
            self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        vout = np.clip(self.vout_gp.get_mean(embedding), 0., 50.)

        return vout

    def get_surrogate_eff_std(self, state=None):
        if state is not None:
            self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        return np.sqrt(self.eff_gp.get_variance(embedding))

    def get_surrogate_vout_std(self, state=None):
        if state is not None:
            self.set_state(state)

        # convert graph to paths, and find embedding
        paths = self.find_paths()
        embedding = tf_embed(paths, self.vec_of_paths)

        return np.sqrt(self.vout_gp.get_variance(embedding))
