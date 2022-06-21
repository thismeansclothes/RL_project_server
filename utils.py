import os
import numpy as np

def set_log_dir(env_id):
    if not os.path.exists('./train_log/'):
        os.mkdir('./train_log/')
    if not os.path.exists('./eval_log/'):
        os.mkdir('./eval_log/')

    if not os.path.exists('./train_log/' + env_id + '/'):
        os.mkdir('./train_log/' + env_id + '/')
    if not os.path.exists('./eval_log/' + env_id + '/'):
        os.mkdir('./eval_log/' + env_id + '/')

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints')

    if not os.path.exists('./checkpoints/' + env_id + '/'):
        os.mkdir('./checkpoints/' + env_id + '/')
    return


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
