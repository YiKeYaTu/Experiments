#!/user/bin/python
# coding=utf-8

from constant import *
import torch
import os


def gen_checkpoint_name(epoch, iteration):
    return 'checkpoint_%s_%s.pth' % (epoch, iteration)


def save_checkpoint(state, epoch=1, iteration=1):
    print('Start saving checkpoint...')
    torch.save(state, os.path.join(
        TMP_ROOT, 'checkpoints', gen_checkpoint_name(epoch, iteration)))
    print('Finish saving.')


def load_checkpoint(model, epoch=None, iteration=None):
    if epoch is not None and iteration is not None:
        model.load_state_dict(torch.load(os.path.join(
            TMP_ROOT, 'checkpoints', gen_checkpoint_name(epoch, iteration))))
        print('The "%s" checkpoint has been loaded.' %
              gen_checkpoint_name(epoch, iteration))
        return True

    checkpoints = sorted([checkpoint for checkpoint in os.listdir(os.path.join(TMP_ROOT, 'checkpoints'))
                          if checkpoint.split('_')[0] == 'checkpoint'],
                         key=lambda name: int(name.split('_')[1]) + int(name.split('_')[2].split('.')[0]))

    print('Candidate checkpoints are: %s.' % checkpoints)

    if len(checkpoints) > 0:
        print('start loading.')
        model.load_state_dict(torch.load(os.path.join(
            TMP_ROOT, 'checkpoints', checkpoints[-1]))['state_dict'])
        print('The "%s" checkpoint has been loaded.' % checkpoints[-1])
        return True
    else:
        print('There are no candidate checkpoints which the model can load.')
        return False
