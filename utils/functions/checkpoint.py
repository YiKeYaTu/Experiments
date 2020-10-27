from constant import *
import torch
import os


def gen_checkpoint_name(epoch, iteration):
    return 'checkpoint_%s_%s.pth' % (epoch, iteration)


def save_checkpoint(state, epoch=1, iteration=1):
    print('Start saving checkpoint...')
    torch.save(state, os.path.join(
        TMP_ROOT, gen_checkpoint_name(epoch, iteration)))
    print('Finish saving.')


def load_checkpoint(model, epoch=None, iteration=None):
    if epoch is not None and iteration is not None:
        model.load_state_dict(torch.load(os.path.join(
            TMP_ROOT, gen_checkpoint_name(epoch, iteration))))
        print('The "%s" checkpoint has been loaded.' % gen_checkpoint_name(epoch, iteration))
        return True

    checkpoints = sorted([checkpoint for checkpoint in os.listdir(os.path.join(TMP_ROOT))
                    if checkpoint.split('_')[0] == 'checkpoint'])

    print('Candidate checkpoints are %s.' % checkpoints)

    if len(checkpoints) > 0:
        model.load_state_dict(torch.load(os.path.join(
            TMP_ROOT, checkpoints[-1])))
        print('The "%s" checkpoint has been loaded.' % checkpoints[-1])
        return True
    else:
        print('There are no candidate checkpoints which the model can load.')
        return False
