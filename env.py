#!/user/bin/python
# coding=utf-8

from utils.Logger import Logger
from torch.utils.tensorboard import SummaryWriter
from constant import *
import sys
import os

if os.path.isdir(TMP_ROOT) is not True:
    os.makedirs(TMP_ROOT)

logger = Logger(join(TMP_ROOT, 'log.txt'))
sys.stdout = logger

writer = SummaryWriter(log_dir=os.path.join(TMP_ROOT, 'tensorboard'))
