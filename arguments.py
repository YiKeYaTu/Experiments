import argparse
import torch
import os
import time

parser = argparse.ArgumentParser()
# Basic options to set the task.
parser.add_argument('-t', '--task', type=str, required=True, choices=[str(task) for task in os.listdir('tasks')],
                    help='The task which you want to excute.')
parser.add_argument('-s', '--sub_task', type=str, default=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                    help='')
parser.add_argument('-m', '--mode', type=str, default='train',
                    help='')
# Options for the model
parser.add_argument('-e', '--epoch', type=int, default=10000,
                    help='Epoch count')
parser.add_argument('-b', '--batch_size', type=int, default=10,
                    help='Batch size')
parser.add_argument('-i', '--iteration_size', type=int, default=1,
                    help='The iteration count for model to update parameters.')
parser.add_argument('-l', '--learning_rate', type=float, default=3e-5,
                    help='The learning rate for model.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='')
parser.add_argument('-d', '--device', type=str, default=('0' if torch.cuda.is_available() else 'cpu'), choices=['cpu'] + [str(cuda_id) for cuda_id in range(torch.cuda.device_count())],
                    help='')
parser.add_argument('-r', '--resume', type=bool, default=False,
                    help='')
parser.add_argument('-rn', '--resume_no', type=int, default=-1,
                    help='')

arguments = parser.parse_args()
