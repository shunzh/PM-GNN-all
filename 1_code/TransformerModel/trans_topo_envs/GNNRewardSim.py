import torch

from TransformerModel.trans_topo_envs.surrogateRewardSim import SurrogateRewardTopologySim


class GNNRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, debug=False, *args):
        super().__init__(debug, *args)

        self.eff_model = torch.load(eff_model_file)
        self.vout_model = torch.load(vout_model_file)

    def get_surrogate_eff(self, state=None):
        # TODO
        pass

    def get_surrogate_vout(self, state=None):
        # TODO
        pass
