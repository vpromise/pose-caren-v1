from torch import nn
from utils import Tester
from network import *

# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_300.pth' 
params.testdata_dir = './dataset/_64/test.txt'

# models
model = StackedHourGlass(256, 1, 2, 4, 3)
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
# model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
# model.fc = nn.Linear(512*4, 6)

# Test
tester = Tester(model, params)
tester.test()