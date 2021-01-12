#!/user/bin/python
# coding=utf-8

from utils.Logger import Logger
from torch.utils.tensorboard import SummaryWriter
from constant import TMP_ROOT
from os.path import join
import sys
import os
import time
# Create temporal directory for storing files.
if os.path.isdir(TMP_ROOT) is not True:
    os.makedirs(TMP_ROOT)
# Create temporal directory for saving checkpoints.
if os.path.isdir(join(TMP_ROOT, 'checkpoints')) is not True:
    os.makedirs(join(TMP_ROOT, 'checkpoints'))

logger = Logger(
    join(
        TMP_ROOT,
        'log_%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    )
)
sys.stdout = logger

epoch_writer = SummaryWriter(log_dir=join(TMP_ROOT, 'tensorboard/epoch'))
iteration_writer = SummaryWriter(log_dir=join(TMP_ROOT, 'tensorboard/iteration'))
 