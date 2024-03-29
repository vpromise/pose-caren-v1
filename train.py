import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import CAREN
from utils import Trainer
from network import *

# Hyper-params
data_root = './dataset/128/' # root for data path txt files
model_path = './models/128/' # model save path
batch_size = 8 # for train 128-->8 256-->2
batch_size_valid = 2 # for valid
num_workers = 2

init_lr = 0.01
lr_decay = 0.8
momentum = 0.9
weight_decay = 0.000
nesterov = True

# Set Training parameters
params = Trainer.TrainParams()
params.max_epoch = 500
params.criterion = nn.MSELoss() # nn.CrossEntropyLoss()
params.gpus = [0,1] # set 'params.gpus=[]' to use CPU mode. examples: [0] [1] [0, 1]
params.save_dir = model_path
params.ckpt = None
params.save_freq_epoch = 10
params.start_save_epoch = 50

# load data
print("Loading dataset...")
train_data = CAREN(data_root,train=True)
val_data = CAREN(data_root,train=False)

batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)
batch_size_valid = batch_size_valid if len(params.gpus) == 0 else batch_size*len(params.gpus)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('train dataset len: {}'.format(len(train_dataloader.dataset)))

val_dataloader = DataLoader(val_data, batch_size=batch_size_valid, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(val_dataloader.dataset)))

# models
model = StackedHourGlass(256, 1, 2, 4, 3)

# optimizer
trainable_vars = [param for param in model.parameters() if param.requires_grad]
print("Training with sgd")
params.optimizer = torch.optim.SGD(trainable_vars, lr=init_lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   nesterov=nesterov)

# Train
params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=lr_decay, patience=10, cooldown=10, verbose=True)
trainer = Trainer(model, params, train_dataloader, val_dataloader)
trainer.train()