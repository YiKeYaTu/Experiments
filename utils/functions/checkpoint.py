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


def load_checkpoint(model, optimizer=None, epoch=None, iteration=None):
    if epoch is not None and iteration is not None:
        print('Load target checkpoint %s-%s' % (epoch, iteration))
        checkpoint = torch.load(
            os.path.join(
                TMP_ROOT, 'checkpoints', gen_checkpoint_name(epoch, iteration)
            ),
            map_location=DEVICE
        )

        if checkpoint['state_dict'] is not None:
            model.load_state_dict(
                checkpoint['state_dict']
            )
        if checkpoint['optimizer'] is not None and optimizer is not None:
            optimizer.load_state_dict(
                checkpoint['optimizer']
            )
            
        print('The "%s" checkpoint has been loaded.' %
              gen_checkpoint_name(epoch, iteration))
        return True

    checkpoints = sorted([checkpoint for checkpoint in os.listdir(os.path.join(TMP_ROOT, 'checkpoints'))
                          if checkpoint.split('_')[0] == 'checkpoint'],
                         key=lambda name: int(name.split('_')[1]) + int(name.split('_')[2].split('.')[0]))

    print('Candidate checkpoints are: %s.' % checkpoints)

    if len(checkpoints) > 0:
        print('start loading.')
        model.load_state_dict(
            torch.load(
                os.path.join(
                    TMP_ROOT, 'checkpoints', checkpoints[-1],
                ),
                map_location=DEVICE
            )['state_dict']
        )
        print('The "%s" checkpoint has been loaded.' % checkpoints[-1])
        return True
    else:
        print('There are no candidate checkpoints which the model can load.')
        return False
