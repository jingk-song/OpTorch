import os
import Sim as oNN
import torch,math
import numpy as np
import Sim as oNN
import matplotlib.pyplot as plt
from scipy.signal import hilbert 
import torch
from torch.autograd import Variable
import torch.optim as optim
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from tqdm import tqdm
thresholds = torch.linspace(0.6, 1, 2**1 + 1)[1:-1]
print(thresholds)