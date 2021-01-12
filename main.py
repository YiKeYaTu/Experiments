#!/user/bin/python
# coding=utf-8

from arguments import *
from constant import *
from importlib import import_module
from utils.functions.checkpoint import load_checkpoint, save_checkpoint
import os


def resume_model(model, force=False):
    if RESUME is True or force is True:
        load_checkpoint(model)


def train():
    from env import logger, epoch_writer
    sub_module = import_module('tasks.%s.%s' %
                               (arguments.task, 'train'))
    model = sub_module.model
    model.train()
    optimizer = sub_module.optimizer
    criterion = sub_module.criterion
    # Loading parameters for model.
    resume_model(model)
    logger.flush()

    for epoch in range(EPOCH):
        # Recording model's information.
        print('Current using device\' no is: %s' % DEVICE)
        print(model)
        print(optimizer)
        
        epoch = epoch + 1
        statistical_losses = sub_module.run(epoch)
        save_checkpoint({
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
        }, epoch=epoch, iteration=1)

        if statistical_losses:
            print('Epoch %s\'s average loss value is: %f' %
                (epoch, statistical_losses.avg))
            epoch_writer.add_scalar('AverageLoss/Epoch', statistical_losses.avg, epoch)
            
        logger.flush()

        try:
            test()
        except:
            print('No test module can be loaded.')

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
