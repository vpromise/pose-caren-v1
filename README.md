# pose-caren.pytorch
human pose estimation in pytorch on CAREN datasets

## Datasets
CAREN Dataset

## Requirements
- pytorch
- torchvision
- visdom

## Traing
`python -m visdom.server` 打开 `Visdom` 服务器,可视化

`python train.py` 即可

## Test
params.ckpt = './models/ckpt_epoch_100.pth' # 修改保存模型路径

`python test.py`
