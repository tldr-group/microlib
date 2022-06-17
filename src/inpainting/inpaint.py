from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
# from inpainting import *
import numpy as np
import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import time
from copy import deepcopy
import imageio
from src.inpainting.networks import *
from src.inpainting.config import *
from src.inpainting.util import *


def train(img, imtype, mask, mask_ip, rect, pth, tag):
    h, w = img.shape
    c = ConfigPoly(tag)
    c.data_path = pth

    # rh, rw = r.height() * sf, r.width() * sf
    x, y = np.meshgrid(np.arange(w), np.arange(h)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    seeds_mask = np.zeros((h,w))
    for x in range(c.l):
        for y in range(c.l):
            seeds_mask += np.roll(np.roll(mask, -x, 0), -y, 1)
    seeds_mask[seeds_mask>1]=1
    real_seeds = np.where(seeds_mask[:-c.l, :-c.l]==0)
    overwrite = True
    initialise_folders(tag, overwrite)
    
    if imtype == 'n-phase':
        c.n_phases = len(np.unique(plt.imread(c.data_path)[...,0]))
        c.conv_resize=True
    else:
        c.n_phases = 1
    c.image_type =  imtype
    netD, netG = make_nets_poly(c, overwrite)
    real_seeds = real_seeds
    mask = mask
    c.save()
    l, batch_size, beta1, beta2, num_epochs, iters, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()
    # Read in data
    ngpu = 1
    training_imgs, nc = preprocess(c.data_path, c.image_type)
    device = torch.device('cuda:0')
    # Define Generator network
    netG = netG().to(device)
    netD = netD().to(device)
    pth = f'data/inpaint_runs/{tag}/'
   
    try:
        os.mkdir(f'{pth}')
        os.mkdir(f'{pth}imgs')
    except:
        pass
    c.frames = 100
    c.iters_per_epoch = 1000
    c.opt_iters = 1000
    c.epochs = 50 if imtype == 'n-phase' else 100
    c.mask = mask
    c.mask_ip = mask_ip
    c.poly_rects = rect
    c.save_inpaint = int(c.opt_iters/c.frames)
    c.pth = pth
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    max_iters = c.epochs*c.iters_per_epoch
    for ep in tqdm(range(c.epochs)):
        torch.cuda.synchronize()
        for i in range(c.iters_per_epoch):
            times = []
            # Discriminator Training
            if ('cuda' in str(device)) and (ngpu > 1):
                start_overall = torch.cuda.Event(enable_timing=True)
                end_overall = torch.cuda.Event(enable_timing=True)
                start_overall.record()
            else:
                start_overall = time.time()

            netD.zero_grad()

            noise = torch.randn(batch_size, nz, lz, lz, device=device)
            fake_data = netG(noise).detach()
            real_data = batch_real_poly(training_imgs, l, batch_size, real_seeds).to(device)

            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc)

            # Compute the discriminator loss and backprop
            disc_cost = out_fake - out_real + gradient_penalty
            disc_cost.backward()

            optD.step()

            # Generator training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                noise = torch.randn(batch_size, nz, lz, lz, device=device)
                # Forward pass through G with noise vector
                fake_data = netG(noise)
                output = -netD(fake_data).mean()

                # Calculate loss for G and backprop
                output.backward()
                optG.step()

            if ('cuda' in str(device)) and (ngpu > 1):
                end_overall.record()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall-start_overall)


            # Every 50 iters log images and useful metrics
            if i==0:
                torch.save(netG.state_dict(), f'{pth}/Gen.pt')
                torch.save(netD.state_dict(), f'{pth}/Disc.pt')
                mse, img = inpaint(netG, c)
                times = []
                plt.imsave(f'{pth}imgs/img{ep}.png', img)
                img_pth = f'data/micrographs_final/{tag}.png'
                plt.imsave(img_pth, img)

def inpaint(netG, c):
    img = preprocess(c.data_path,c.image_type)[0]
    img = torch.nn.functional.pad(img, (32, 32, 32, 32), value=-1)
    mask_ip = torch.nn.functional.pad(torch.tensor(c.mask_ip), (32, 32, 32, 32), value=0)
    
    for rect in c.poly_rects:
        x0, y0, x1, y1 = (int(i)+32 for i in rect)
        w, h = x1-x0, y1-y0
        x1 += 32 - w%32
        y1 += 32 - h%32

        w, h = x1-x0, y1-y0
        im_crop = img[:, x0-16:x1+16, y0-16:y1+16]
        mask_crop = mask_ip[x0-16:x1+16, y0-16:y1+16]
        ch, w, h = im_crop.shape
        # print(im_crop.shape, mask_crop.shape, img.shape, mask_ip.shape, x0, y0, x1, y1)
        if c.conv_resize:
            lx, ly = int(w/16), int(h/16)
        else:
            lx, ly = int(w/32) + 2, int(h/32) + 2
        inpaints, mse = optimise_noise(c, lx, ly, im_crop, mask_crop, netG)
        frames = len(inpaints)

        if c.image_type =='n-phase':
            final_imgs = [torch.argmax(img, dim=0) for i in range(frames)]
            final_img_fresh = torch.argmax(img, dim=0)
        else:
            final_img_fresh = img.permute(1, 2, 0)
            final_imgs = [deepcopy(img.permute(1, 2, 0)) for i in range(frames)]
        for fimg, inpaint in enumerate(inpaints):
            final_imgs[fimg][x0:x1,  y0:y1] = inpaint
    for i, final_img in enumerate(final_imgs):
        istr = f'00{i}'
        if c.image_type=='n-phase':
            final_img[mask_ip!=1] = final_img_fresh[mask_ip!=1]
            # final_img[mask_ip!=1] = 0.5
            
            final_img = (final_img.numpy()/final_img.max())
            plt.imsave(f'data/temp/temp{istr[-3:]}.png', np.stack([final_img for i in range(3)], -1))
        else:
            for ch in range(c.n_phases): 
                final_img[:,:,ch][mask_ip==0] = final_img_fresh[:,:,ch][mask_ip==0]
            final_img[final_img==-1] = 0.5
            final_img-= final_img.min()
            final_img/= final_img.max()
            if c.image_type=='grayscale':
                plt.imsave(f'data/temp/temp{istr[-3:]}.png', np.concatenate([final_img for i in range(3)], -1))
    filenames = sorted(os.listdir('data/temp'))
    with imageio.get_writer(f'{c.pth}/movie.gif', mode='I', duration=0.2) as writer:
        image = imageio.imread(f'data/temp/temp000.png')
        for ch in range(2):
            image[...,ch][mask_ip==1] = 0
        
        image[...,2][mask_ip==1] = 255
        for i in range(10):
            writer.append_data(image[32:-32,32:-32])

        for filename in filenames:
            image = imageio.imread(f'data/temp/{filename}')
            writer.append_data(image[32:-32,32:-32])
    
    return mse, image

def optimise_noise(c, lx, ly, img, mask, netG):
    
    device = torch.device("cuda:0" if(
        torch.cuda.is_available() and c.ngpu > 0) else "cpu")        
    target = img.to(device)
    for ch in range(c.n_phases):
        target[ch][mask==1] = -1
    # plt.imsave('test2.png', torch.cat([target.permute(1,2,0) for i in range(3)], -1).cpu().numpy())
    # plt.imsave('test.png', np.stack([mask for i in range(3)], -1))

    target = target.unsqueeze(0)
    noise = [torch.nn.Parameter(torch.randn(1, c.nz, lx, ly, requires_grad=True, device=device))]
    noise_opt = torch.optim.Adam(params=noise, lr=0.02, betas=(0.8, 0.8))
    inpaints = []
    # loss_min = 1000
    for i in range(c.opt_iters):
        raw = netG(noise[0])
        # print(raw.shape, target.shape)
        loss = (raw - target)**4
        loss[target==-1] = 0
        loss = loss.mean()
        loss.backward()
        noise_opt.step()
        with torch.no_grad():
            noise[0] -= torch.tile(torch.mean(noise[0], dim=[1]), (1, c.nz,1,1))
            noise[0] /= torch.tile(torch.std(noise[0], dim=[1]), (1, c.nz,1,1))
        if c.image_type == 'n-phase':
            raw = torch.argmax(raw[0], dim=0)[16:-16, 16:-16].detach().cpu()
        else:
            raw = raw[0].permute(1,2,0)[16:-16, 16:-16].detach().cpu()
        
        if (i%c.save_inpaint==0) or (i <20):
            inpaints.append(raw)
        # if  (loss < loss_min):
        #     best_img = deepcopy(raw)
        #     loss_min = loss
        
           
                
    # inpaints.append(best_img)
    return inpaints, loss.item()