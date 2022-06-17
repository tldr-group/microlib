from time import sleep
from tkinter import image_types
import numpy as np
import torch
from torch import autograd
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from torch import nn
import tifffile

# check for existing models and folders
def check_existence(tag):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    root = f'data/inpaint_runs/{tag}'
    check_D = os.path.exists(f'{root}/Disc.pt')
    check_G = os.path.exists(f'{root}/Gen.pt')
    if check_G or check_D:
        print(f'Models already exist for tag {tag}.')
        x = input("To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n")
        if x=='o':
            print("Overwriting")
            return True
        if x=='l':
            print("Loading previous model")
            return False
        elif x=='c':
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True


# set-up util
def initialise_folders(tag, overwrite):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f'data/inpaint_runs')
        except:
            pass
        try:
            os.mkdir(f'data/inpaint_runs/{tag}')
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
    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    if API_KEY is None or ENTITY is None or PROJECT is None:
        raise AssertionError('.env file arguments missing. Make sure WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are present.')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    
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

def wandb_save_models(fn):
    """[summary]

    :param pth: [description]
    :type pth: [type]
    :param fn: [description]
    :type fn: filename
    """
    shutil.copy(fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)

# training util
def preprocess(data_path, imtype, load=True):
    """[summary]

    :param imgs: [description]
    :type imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    # img = tifffile.imread(data_path)
    img = plt.imread(data_path)
    if imtype == 'colour':
            img = img[:,:,:3]
            img = torch.tensor(img)
            return img.permute(2,0,1), 3
    else:
        if len(img.shape) > 2:
            img = img[...,0]
        if imtype == 'n-phase':
            phases = np.unique(img)
            if len(phases) > 10:
                raise AssertionError('Image not one hot encoded.')
            # x, y, z = img.shape
            x, y = img.shape
            # img_oh = torch.zeros(len(phases), x, y, z)
            img_oh = torch.zeros(len(phases), x, y)
            for i, ph in enumerate(phases):
                img_oh[i][img == ph] = 1
            return img_oh, len(phases)
        elif imtype == 'grayscale':
            img = np.expand_dims(img, 0)
            img = torch.tensor(img)
            return img, 1
        # x, y, z = img.shape


def calculate_size_from_seed(seed, c):
    imsize = seed
    for (k, s, p) in zip(c.gk, c.gs, c.gp):
        imsize = (imsize-1)*s-2*p+k
    return imsize

def calculate_seed_from_size(imsize, c):
    for (k, s, p) in zip(c.gk, c.gs, c.gp):
        imsize = ((imsize-k+2*p)/s+1).to(int)
    return imsize

def make_mask(training_imgs, c):
    y1,y2,x1,x2 = c.mask_coords
    ydiff, xdiff = y2-y1, x2-x1
    seed = calculate_seed_from_size(torch.tensor([xdiff, ydiff]).to(int), c)
    img_seed = seed+2
    img_size = calculate_size_from_seed(img_seed, c)
    mask_size = calculate_size_from_seed(seed, c)
    D_size_dim = int(torch.div(mask_size.min(),32, rounding_mode='floor'))*16
    D_seed = calculate_seed_from_size(torch.tensor([D_size_dim, D_size_dim]).to(int), c)

    x2, y2 = x1+mask_size[0].item(), y1+mask_size[1].item()
    xmid, ymid = (x2+x1)//2, (y2+y1)//2
    x1_bound, x2_bound, y1_bound, y2_bound = xmid-img_size[0].item()//2, xmid+img_size[0].item()//2, ymid-img_size[1].item()//2, ymid+img_size[1].item()//2
    unmasked = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound].clone()
    training_imgs[:, x1:x2, y1:y2] = 0
    mask = training_imgs[:,x1_bound:x2_bound, y1_bound:y2_bound]
    mask_layer = torch.zeros_like(training_imgs[0]).unsqueeze(0)
    unmasked = torch.cat([unmasked, torch.zeros_like(unmasked[0]).unsqueeze(0)])
    mask_layer[:,x1:x2,y1:y2] = 1
    mask = torch.cat((mask, mask_layer[:,x1_bound:x2_bound, y1_bound:y2_bound]))

    # save coords to c
    c.mask_coords = (x1,x2,y1,y2)
    c.mask_size = (mask_size[0].item(), mask_size[1].item())
    c.D_seed_x = D_seed[0].item()
    c.D_seed_y = D_seed[1].item()
    
    # plot regions where discriminated
    # plt.figure()
    # plotter = mask.permute(1,2,0).numpy().copy()
    # plotter[(img_size[0].item()-D_size_dim)//2:(img_size[0].item()+D_size_dim)//2,(img_size[1].item()-D_size_dim)//2:(img_size[1].item()+D_size_dim)//2,:] = 0
    # plt.imshow(plotter)
    # plt.savefig('data/mask_plot.png')
    # plt.close()

    # plt.imsave('data/mask.png',mask.permute(1,2,0).numpy())
    # plt.imsave('data/unmasked.png',unmasked.permute(1,2,0).numpy())
    return mask, unmasked, D_size_dim, img_size, img_seed, c

def update_discriminator(c):
    out = c.dl
    layer = 0
    dk = [4]
    dp = [1]
    ds = [2]
    df = [c.n_phases]
    while out != 1:
        out_check = int(round((out+2*dp[layer]-dk[layer])/ds[layer]+1))
        if out_check>1:
            out = out_check
            dk.append(4)
            ds.append(2)
            dp.append(1)
            df.append(int(np.min([2**(layer+6), 512])))
            layer += 1
        elif out_check<1:
            dp[layer] = int(round((2+dk[layer]-out)/2))
        else:
            out = out_check
    df.append(1)
    c.df = df
    c.dk = dk
    c.ds = ds
    c.dp = dp
    return c

def update_pixmap_rect(raw, img, c):
    updated_pixmap = raw.clone().unsqueeze(0)
    x1, x2, y1, y2 = c.mask_coords
    lx, ly = c.mask_size
    x_1, x_2, y_1, y_2 = (img.shape[2]-lx)//2,(img.shape[2]+lx)//2, (img.shape[3]-ly)//2, (img.shape[3]+ly)//2
    updated_pixmap[:,:, x1:x2, y1:y2] = img[:,:,x_1:x_2, y_1:y_2]
    updated_pixmap = post_process(updated_pixmap, c).permute(0,2,3,1)
    if c.image_type=='grayscale':
        plt.imsave('data/temp/temp.png', updated_pixmap[0,...,0], cmap='gray')
    else:
        plt.imsave('data/temp/temp.png', updated_pixmap[0].numpy())

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
    alpha = alpha.view(batch_size, nc, l, l)
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

def batch_real_poly(img, l, bs, real_seeds):
    n_ph, _, _ = img.shape
    max_idx = len(real_seeds[0])
    idxs = torch.randint(max_idx, (bs,))
    data = torch.zeros((bs, n_ph, l, l))
    for i, idx in enumerate(idxs):
        x, y = real_seeds[0][idx], real_seeds[1][idx]
        data[i] = img[:, x:x+l, y:y+l]
    return data

def batch_real(img, l, bs, mask_coords):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    x1, x2, y1, y2 = mask_coords
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, l, l))
    for i in range(bs):
        x, y = torch.randint(x_max - l, (1,)), torch.randint(y_max - l, (1,))
        while (x1<x+l and x1>x-l) and (y1<y+l and y1>y-l):
            x, y = torch.randint(x_max - l, (1,)), torch.randint(y_max - l, (1,))
        data[i] = img[:, x:x+l, y:y+l]
    return data

def pixel_wise_loss(fake_img, real_img, coeff=1, device=None):
    mask = real_img.clone().permute(1,2,0)
    mask = (mask[...,-1]==0).unsqueeze(0)
    mask = mask.repeat(fake_img.shape[0], fake_img.shape[1],1,1)
    fake_img = torch.where(mask==True, fake_img, torch.tensor(0).float().to(device))
    real_img = real_img.unsqueeze(0).repeat(fake_img.shape[0], 1 ,1, 1)[:,0:-1]
    real_img = torch.where(mask==True, real_img, torch.tensor(0).float().to(device))
    return torch.nn.MSELoss(reduction='none')(fake_img, real_img)*coeff

# Evaluation util
def post_process(img, c):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    if c.image_type=='n-phase':
        phases = np.arange(c.n_phases)
        color = iter(cm.rainbow(np.linspace(0, 1, c.n_phases)))
        # color = iter([[0,0,0],[0.5,0.5,0.5], [1,1,1]])
        img = torch.argmax(img, dim=1)
        if len(phases) > 10:
            raise AssertionError('Image not one hot encoded.')
        bs, x, y = img.shape
        out = torch.zeros((bs, 3, x, y))
        for b in range(bs):
            for i, ph in enumerate(phases):
                col = next(color)
                col = torch.tile(torch.Tensor(col[0:3]).unsqueeze(1).unsqueeze(1), (x,y))
                out[b] = torch.where((img[b] == ph), col, out[b])
        out = out
    else:
        out = img
    return out

def crop(fake_data, l):
    w = fake_data.shape[2]
    return fake_data[:,:,w//2-l//2:w//2+l//2,w//2-l//2:w//2+l//2]

def make_noise(noise, seed_x, seed_y, c, device):
    # noise = torch.ones(bs, nz, seed_x, seed_y, device=device)
    mask = torch.zeros_like(noise).to(device)
    mask[:,:, (seed_x-c.D_seed_x)//2:(seed_x+c.D_seed_x)//2, (seed_y-c.D_seed_y)//2:(seed_y+c.D_seed_y)//2] = 1
    rand = torch.randn_like(noise).to(device)*mask
    noise = noise*(mask==0)+rand
    return noise