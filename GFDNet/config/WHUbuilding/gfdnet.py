from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.WHUbuilding_dataset import *
from geoseg.losses.dice import DiceLoss
from geoseg.losses.joint_loss import JointLoss
from geoseg.losses.soft_ce import SoftCrossEntropyLoss
from geoseg.losses.Loss import ModelLossModelLoss
from geoseg.models.GFDNet import GFDNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils

max_epoch = 105
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = 2  #
classes = CLASSES


weights_name = "whu-building-e105"
weights_path = "model_weights/whubuilding/{}".format(weights_name)
test_weights_name = "whu-building-e105"
log_name = 'whubuilding/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None

net = GFDNet(num_classes=num_classes)

loss = ModelLoss(ignore_index=ignore_index)


use_aux_loss = True


train_dataset = WHUBuildingDataset(
    data_root='data/WHUbuilding/train',
    mode='train',
    mosaic_ratio=0.25,
    transform=train_aug
)

val_dataset = WHUBuildingDataset(
    data_root='data/WHUbuilding/val',
    mode='val',
    transform=val_aug
)

test_dataset = WHUBuildingDataset(
    data_root='data/WHUbuilding/test',
    mode='test',
    transform=val_aug
)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True
                          )

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False
                        )

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)