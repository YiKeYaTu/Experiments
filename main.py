from arguments import *
from constant import *
from importlib import import_module
from utils.functions.checkpoint import load_checkpoint, save_checkpoint
import os

def create_tmp_root():
    if os.path.isdir(TMP_ROOT) is not True:
        os.makedirs(TMP_ROOT)

def resume_model(model):
    if RESUME is True:
        load_checkpoint(model)

def main():
    sub_module = import_module('tasks.%s.%s' % (arguments.task, arguments.mode))
    
    create_tmp_root()
    resume_model(sub_module.model)
    
    for epoch in range(EPOCH):
        epoch = epoch + 1
        sub_module.run(epoch)
        save_checkpoint(sub_module.model, epoch=epoch, iteration=1)


if __name__ == '__main__':
    main()