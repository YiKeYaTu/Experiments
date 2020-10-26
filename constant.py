import os
import torch

from os.path import abspath, join, dirname
from arguments import *

# Datasets and Current path of directory..
DATASETS_ROOT = abspath(dirname(__file__) + os.sep + '../../DataSets')
PROJECT_ROOT = abspath(dirname(__file__))

# Current task's working directory.
WORKING_ROOT = join(PROJECT_ROOT, 'tasks', arguments.task)
TMP_ROOT = join(WORKING_ROOT, '__tmp__', arguments.sub_task)

# Options for the model
EPOCH = arguments.epoch
BATCH_SIZE = arguments.batch_size
ITERATION_SIZE = arguments.iteration_size
LEARNING_RATE = arguments.learning_rate
DEVICE = torch.device(arguments.device if arguments.device == 'cpu' else 'cuda:' + arguments.device)
RESUME = arguments.resume
