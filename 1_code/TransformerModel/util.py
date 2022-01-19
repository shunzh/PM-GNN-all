import random

import numpy
import torch


def feed_random_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

UCT_BASE_DIR = 'UCT_5_only_epr/'