#!/user/bin/python
# coding=utf-8

from arguments import *
from constant import *
from importlib import import_module
from utils.functions.checkpoint import load_checkpoint, save_checkpoint
import os


def resume_model(model, optimizer=None, force=False):
    if RESUME is True or force is True:
        if RESUME_NO != -1:
            load_checkpoint(model, optimizer, RESUME_NO, 1)
        else:
            load_checkpoint(model, optimizer)


def train():
    from env import logger, epoch_writer
    sub_module = import_module('tasks.%s.%s' %
                               (arguments.task, 'train'))
    model = sub_module.model
    model.train()
    optimizer = sub_module.optimizer
    criterion = sub_module.criterion
    # Loading parameters for model.
    resume_model(model, optimizer)
    logger.flush()

    print('Current using device\' no is: %s' % DEVICE)
    # Recording model's information.
    print(model)
    print(optimizer)

    for epoch in range(EPOCH):
        statistical_losses = sub_module.run(epoch)
        save_checkpoint({
            'epoch': epoch,
            'optimizer': optimizer and optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }, epoch=epoch, iteration=1)

        if statistical_losses:
            print('Epoch %s\'s average loss value is: %f' %
                (epoch, statistical_losses.avg))
            epoch_writer.add_scalar('AverageLoss/Epoch', statistical_losses.avg, epoch)

        print('Epoch %s finished.' % (epoch))
            
        logger.flush()

        try:
            test()
        except:
            print('No test module can be loaded.')

        if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
            print('Train finished.')
            break

    logger.close()
    epoch_writer.close()
    iteration_writer.close()


def test():
    sub_module = import_module('tasks.%s.%s' %
                               (arguments.task, 'test'))
    model = sub_module.model
    resume_model(model, force=True)
    model.eval()
    sub_module.run()


def default():
    import_module('tasks.%s.%s' %
                  (arguments.task, arguments.mode))


def main():
    from env import logger, epoch_writer
    if arguments.mode == 'train':
        train()
    elif arguments.mode == 'test':
        test()
    else:
        default()


if __name__ == '__main__':
    main()
