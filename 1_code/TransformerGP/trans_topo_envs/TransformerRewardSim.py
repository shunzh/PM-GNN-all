import copy
from dataclasses import dataclass

import gpytorch
import torch
import numpy as np

from trans_topo_envs.surrogateRewardSimFactory import SurrogateRewardSimFactory
from transformer_SVGP.train import main as train_transformer
from transformer_SVGP.Models import get_model, GPModel
from trans_topo_envs.surrogateRewardSim import SurrogateRewardTopologySim

from build_vocab import Vocabulary


@dataclass
class ModelInfo:
    model: ...
    gp: ...
    likelihood: ...
    data_train: ...


class TransformerRewardSimFactory(SurrogateRewardSimFactory):
    """
    A class that can generate the transformer initializer
    """

    # args used for training the transformer, obtained from Yupeng.
    def __init__(self, eff_model_file, vout_model_file, vocab_file,
                 device, dev_file=None, eff_model_seed=0, vout_model_seed=0, epoch=10000, patience=50,
                 sample_ratio=0.1):
        """
        Initially, the pretrained models are loaded from files
        """
        self.device = device

        vocab = Vocabulary()
        self.vocab_file = vocab_file
        vocab.load(vocab_file)
        self.vocab = vocab

        self.dev_file = dev_file

        self.eff_model_seed = eff_model_seed
        self.vout_model_seed = vout_model_seed

        self.eff_model_file = eff_model_file
        self.vout_model_file = vout_model_file

        self.reset_model()

        self.epoch = epoch
        self.patience = patience
        self.sample_ratio = sample_ratio

    def reset_model(self):
        # loaded in (model, gp, likelihood) tuples
        self.eff_models = self.load_model_from_file(self.eff_model_file)
        self.vout_models = self.load_model_from_file(self.vout_model_file)

    def load_model_from_file(self, file_name):
        # load transformer model, using Yupeng's code
        # fixme get_model was only implemented for transformer, not transformer + gp
        model = get_model(cuda=(self.device == 'gpu'), pretrained_model=file_name, load_weights=True)
        model = model.to(self.device)

        checkpoint = torch.load(file_name + '.chkpt')

        gp_para = checkpoint["gp_model"]
        gp = GPModel(gp_para["variational_strategy.inducing_points"])
        gp.load_state_dict(gp_para)
        gp = gp.to(self.device)

        likelihood_para = checkpoint['likelihood']
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.load_state_dict(likelihood_para)
        likelihood = likelihood.to(self.device)

        if 'data_train' in checkpoint.keys():
            training_data = checkpoint['data_train']
        else:
            # okay to omit training data
            training_data = None

        model.eval()
        gp.eval()

        return ModelInfo(model, gp, likelihood, training_data)

    def add_data_to_model_and_train(self, path_set, effs, vouts):
        training_data = copy.deepcopy(self.eff_models.data_train)
        pre_training_data = copy.deepcopy(self.eff_models.data_train)
        assert training_data is not None

        training_data.random_sample_data(ratio=self.sample_ratio)

        training_data.append_data(path_set, effs, vouts)
        pre_training_data.append_data(path_set, effs, vouts)

        # keep training the transformer, while using the current trained models
        new_eff_models = train_transformer(args=['-data_dev=' + self.dev_file,
                                                 '-target=eff',
                                                 '-vocab=' + self.vocab_file,
                                                 '-seed=' + str(self.eff_model_seed)],
                                           training_data=training_data,
                                           transformer=self.eff_models.model,
                                           gp=self.eff_models.gp,
                                           likelihood=self.eff_models.likelihood,
                                           epoch=self.epoch,
                                           patience=self.patience)

        # self.eff_models = ModelInfo(*new_eff_models)
        self.eff_models = ModelInfo(*(new_eff_models[0], new_eff_models[1],
                                      new_eff_models[2], new_eff_models[3]))
        effi_early_stop = new_eff_models[4]

        new_vout_models = train_transformer(args=['-data_dev=' + self.dev_file,
                                                  '-target=vout',
                                                  '-vocab=' + self.vocab_file,
                                                  '-seed=' + str(self.vout_model_seed)],
                                            training_data=training_data,
                                            transformer=self.vout_models.model,
                                            gp=self.vout_models.gp,
                                            likelihood=self.vout_models.likelihood,
                                            epoch=self.epoch,
                                            patience=self.patience)
        self.vout_models = ModelInfo(*(new_vout_models[0], new_vout_models[1],
                                       new_vout_models[2], new_vout_models[3]))
        vout_early_stop = new_vout_models[4]

        self.eff_models.data_train = pre_training_data
        # print(self.eff_models)
        return effi_early_stop, vout_early_stop

    def update_sim_models(self, sim):
        sim.eff_models = self.eff_models
        sim.vout_models = self.vout_models

    def get_sim_init(self):
        return lambda *args: TransformerRewardSim(self.eff_models, self.vout_models, self.vocab, self.device, *args)


class TransformerRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_models, vout_models, vocab, device, *args):
        super().__init__(*args)

        self.eff_models = eff_models
        self.vout_models = vout_models

        self.vocab = vocab

        self.device = device

    def get_transformer_predict(self, model, gp, need_std=False):
        paths = self.find_paths()

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            path_list = [self.vocab(token) for token in paths]
            path_tensor = torch.Tensor(path_list).unsqueeze(0).long()

            padding_mask = (path_tensor != 0)

            path_tensor = path_tensor.to(self.device)
            padding_mask = padding_mask.to(self.device)

            pred, final = model(path_tensor, padding_mask)
            pred = gp(pred)

        if need_std:
            return np.sqrt(pred.variance.item())
        else:
            return pred.mean.item()

    def get_transformer_predict_with_topo_info(self, node_list, edge_list, model, gp, need_std=False):
        # paths = self.find_paths()
        paths = self.find_paths_with_topo_info(node_list=node_list, edge_list=edge_list)
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            path_list = [self.vocab(token) for token in paths]
            path_tensor = torch.Tensor(path_list).unsqueeze(0).long()

            padding_mask = (path_tensor != 0)

            path_tensor = path_tensor.to(self.device)
            padding_mask = padding_mask.to(self.device)

            pred, final = model(path_tensor, padding_mask)
            pred = gp(pred)

        if need_std:
            return np.sqrt(pred.variance.item())
        else:
            return pred.mean.item()

    def get_surrogate_eff(self, state=None):
        if state is not None:
            self.set_state(state)

        return np.clip(self.get_transformer_predict(self.eff_models.model, self.eff_models.gp), 0., 1.)

    def get_surrogate_vout(self, state=None):
        if state is not None:
            self.set_state(state)

        return np.clip(50 * self.get_transformer_predict(self.vout_models.model, self.vout_models.gp), 0., 50.)

    def get_surrogate_eff_with_topo_info(self, node_list, edge_list, state=None):
        if state is not None:
            self.set_state(state)
        if (not node_list) and (not edge_list):
            node_list, edge_list = self.current.get_nodes_and_edges()

        return np.clip(self.get_transformer_predict_with_topo_info(node_list=node_list, edge_list=edge_list,
                                                                   model=self.eff_models.model,
                                                                   gp=self.eff_models.gp),
                       0., 1.)

    def get_surrogate_vout_with_topo_info(self, node_list, edge_list, state=None):
        if state is not None:
            self.set_state(state)
        if (not node_list) and (not edge_list):
            node_list, edge_list = self.current.get_nodes_and_edges()

        return np.clip(50 * self.get_transformer_predict_with_topo_info(node_list=node_list, edge_list=edge_list,
                                                                        model=self.vout_models.model,
                                                                        gp=self.vout_models.gp),
                       0., 50.)

    def get_surrogate_eff_std(self, state=None):
        if state is not None:
            self.set_state(state)

        return self.get_transformer_predict(self.eff_models.model, self.eff_models.gp, need_std=True)

    def get_surrogate_vout_std(self, state=None):
        if state is not None:
            self.set_state(state)

        return 50 * self.get_transformer_predict(self.vout_models.model, self.vout_models.gp, need_std=True)
