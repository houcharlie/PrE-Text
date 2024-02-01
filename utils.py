
import torch
import torch.nn as nn
import numpy as np
import random
import os

from opacus.validators.module_validator import ModuleValidator
from torchvision import models
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_0

import math
import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Tuple

import torch
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.utils.data.data_utils import batchify
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm


def set_random_seed(seed_value, use_cuda: bool = True) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


