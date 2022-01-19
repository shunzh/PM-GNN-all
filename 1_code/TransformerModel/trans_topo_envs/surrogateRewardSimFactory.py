from abc import abstractmethod, ABC

from TransformerModel.UCFTopo_dev.ucts.TopoPlanner import TopoGenSimulator


class SurrogateRewardSimFactory(ABC):
    def reset_model(self):
        # assume no reset needed if not implemented by subclass
        pass

    @abstractmethod
    def get_sim_init(self):
        pass


class SoftwareSimulatorFactory(SurrogateRewardSimFactory):
    def get_sim_init(self):
        return TopoGenSimulator
