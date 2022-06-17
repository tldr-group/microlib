### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from src.slicegan import model, networks, util
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def slicegan_dataset():
    if not os.path.exists('data/slicegan_runs'):
        os.mkdir('data/slicegan_runs')
    tags = sorted(os.listdir('data/micrographs_final'))
    for tag in tags:
        slicegan_tag('train', tag)

def slicegan_gen_dataset():
    tags = sorted(os.listdir('data/slicegan_runs'))
    tags = [tag +'.png' for tag in tags]
    for tag in tags:
        try:
            slicegan_tag('gen', tag)
        except:
            print(f'couldnt generate {tag}')

def slicegan_tag(mode, tag):
# Define project name
    Project_name = tag[:-4]
    # Specify project folder.
    Project_dir = 'data/slicegan_runs/'
    if not os.path.exists(Project_dir):
        os.mkdir(Project_dir)
    # Run with False to show an image during or after training
    Training = True if mode == 'train' else False
    Project_path = util.mkdr(Project_name, Project_dir, Training,)
    if not Project_path:
        return
    ## Data Processing
    # Define image  type (colour, grayscale, three-phase or two-phase.
    # n-phase materials must be segmented)
    img = plt.imread(f'data/micrographs_final/{tag}')

    image_type = 'twophase' if len(np.unique(img)) == 2 else 'grayscale'
    # define data type (for colour/grayscale images, must be 'colour' / '
    # greyscale. nphase can be, 'tif', 'png', 'jpg','array')
    data_type = 'png' if image_type == 'twophase' else 'grayscale'
    # Path to your data. One string for isotrpic, 3 for anisotropic

    data_path = [f'data/micrographs_final/{tag}']

    ## Network Architectures
    # Training image size, no. channels and scale factor vs raw data
    img_size, img_channels, scale_factor = 64, 2 if image_type=='twophase' else 1,  1
    # z vector depth
    z_channels = 16
    # Layers in G and D
    lays = 5
    # kernals for each layer
    dk, gk = [4]*lays, [4]*lays
    # strides
    ds, gs = [2]*lays, [2]*lays
    # no. filters
    df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
    # paddings
    dp, gp = [1,1,1,1,0],[2,2,2,2,3]

    ## Create Networks
    netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp)
    print('training')
    # Train
    if Training:
        model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor)
    else:
        img, raw, netG = util.test_img(Project_path, image_type, netG(), z_channels, lf=12, periodic=False)
