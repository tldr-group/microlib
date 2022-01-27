import numpy as np
import numpy as np
import torch
from torch import autograd
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import logging
from postprocessing import post_process


def initialise_folders(tag):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    try:
        os.mkdir(f'runs')
    except:
        pass
    try:
        os.mkdir(f'runs/{tag}')
    except:
        pass


def wandb_init(name, offline):
    """[summary]

    :param name: [description]
    :type name: [type]
    :param offline: [description]
    :type offline: [type]
    """
    if offline:
        mode = 'disabled'
    else:
        mode = None
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    print('initing')
    wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        # 'watch_called': False,
        'no_cuda': False,
        # 'seed': 42,
        'log_interval': 1000,

    }
    # wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    # wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']


def wandb_save_models(pth, fn):
    """[summary]

    :param pth: [description]
    :type pth: [type]
    :param fn: [description]
    :type fn: function
    """
    shutil.copy(pth+fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def batch_real(training_imgs):
    """[summary]

    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    return training_imgs

def progress(i, iters, n, num_epochs, timed):
    """[summary]

    :param i: [description]
    :type i: [type]
    :param iters: [description]
    :type iters: [type]
    :param n: [description]
    :type n: [type]
    :param num_epochs: [description]
    :type num_epochs: [type]
    :param timed: [description]
    :type timed: [type]
    """
    progress = 'iteration {} of {}, epoch {} of {}'.format(
        i, iters, n, num_epochs)
    logging.warning({"Progress": progress, "Time per iter": timed})

def plot_img(img, slcs=4):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    img = post_process(img)
    images = []
    for j in range(slcs):
        j = np.random.randint(0, img.shape[0])
        images.append(img[j])
    wandb.log({"slices": [wandb.Image(i) for i in images]})