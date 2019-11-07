from torch import nn
from utils import Tester
from network import *

# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_300.pth' 
params.testdata_dir = './dataset/128/test.txt'

# models
model = StackedHourGlass(256, 1, 2, 4, 3)

# Test
tester = Tester(model, params)
tester.test()
