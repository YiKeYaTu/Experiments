from arguments import *
from constant import *
from importlib import import_module
from utils.functions.checkpoint import load_checkpoint, save_checkpoint
import os
import env
from env import logger


def resume_model(model):
    if RESUME is True:
        load_checkpoint(model)


def train():
    sub_module = import_module('tasks.%s.%s' %
                               (arguments.task, arguments.mode))
    model = sub_module.model
    resume_model(model)

    for epoch in range(EPOCH):
        epoch = epoch + 1
        sub_module.run(epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, epoch=epoch, iteration=1)
        logger.flush()


def test():
    sub_module = import_module('tasks.%s.%s' %
                               (arguments.task, arguments.mode))
    model = sub_module.model
    resume_model(model)


def default():
    import_module('tasks.%s.%s' %
                  (arguments.task, arguments.mode))


def main():
    if arguments.mode == 'train':
        train()
    elif arguments.mode == 'test':
        test()
    else:
        default()


if __name__ == '__main__':
    main()
