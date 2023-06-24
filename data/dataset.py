import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from rdkit import Chem

print(f"Torch Version: {torch.__version__}")
print(f"Is Cuda Available: {torch.cuda.is_available()}")
print(f"Torch Geometric Version: {torch_geometric.__version__}")

class HIVMoleculeDataset(Dataset):