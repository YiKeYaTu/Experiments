#!/user/bin/python
# coding=utf-8

from env import logger
from constant import *

def print_training_status(cur_epoch, cur_iteration, max_iteration, cur_loss, avg_loss):
    info = 'Epoch: [{0}/{1}][{2}/{3}]'.format(cur_epoch, EPOCH, cur_iteration, max_iteration) + \
        'Loss: {0:f} (Average: {1:f})'.format(cur_loss, avg_loss)
    print(info)

    return info
