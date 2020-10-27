from env import logger

def print_training_status(cur_epoch, cur_iteration, max_iteration, cur_loss, avg_loss):
    info = 'Epoch: [{0}][{1}/{2}]'.format(cur_epoch, cur_iteration, max_iteration) + \
        'Loss: {0:f} (Average: {1:f})'.format(cur_loss, avg_loss)
    print(info)

    return info
