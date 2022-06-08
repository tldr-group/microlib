from src.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import tifffile
import time

def train(c, Gen, Disc, offline=True, overwrite=True):
    """[summary]

    :param c: [description]
    :type c: [type]
    :param Gen: [description]
    :type Gen: [type]
    :param Disc: [description]
    :type Disc: [type]
    :param offline: [description], defaults to True
    :type offline: bool, optional
    """

    # Assign torch device
    ngpu = c.ngpu
    tag = c.tag
    path = c.path
    device = torch.device(c.device_name if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f'Using {ngpu} GPUs')
    print(device, " will be used.\n")
    cudnn.benchmark = True

    # Get train params
    l, batch_size, beta1, beta2, num_epochs, iters, lrg, lr, Lambda, critic_iters, lz, nz, = c.get_train_params()

    # Read in data
    training_imgs, nc = preprocess(c.data_path)

    # Define Generator network
    netG = Gen().to(device)
    netD = Disc().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    if not overwrite:
        netG.load_state_dict(torch.load(f"{path}/Gen.pt"))
        netD.load_state_dict(torch.load(f"{path}/Disc.pt"))

    wandb_init(tag, offline)
    wandb.watch(netD, log='all', log_freq=100)
    wandb.watch(netG, log='all', log_freq=100)

    for epoch in range(num_epochs):
        times = []
        for i in range(iters):
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

            real_data = batch_real(training_imgs, l, batch_size).to(device)

            # Train on real
            out_real = netD(real_data).mean()
            # train on fake images
            out_fake = netD(fake_data).mean()
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, Lambda, nc)

            # Compute the discriminator loss and backprop
            disc_cost = out_fake - out_real + gradient_penalty
            disc_cost.backward()

            optD.step()

            # Log results
            if not offline:
                wandb.log({'Gradient penalty': gradient_penalty.item()})
                wandb.log({'Wass': out_real.item() - out_fake.item()})
                wandb.log({'Discriminator real': out_real.item()})
                wandb.log({'Discriminator fake': out_fake.item()})

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
                torch.cuda.synchronize()
                times.append(start_overall.elapsed_time(end_overall))
            else:
                end_overall = time.time()
                times.append(end_overall-start_overall)


            # Every 50 iters log images and useful metrics
            if i % 50 == 0:
                with torch.no_grad():
                    torch.save(netG.state_dict(), f'{path}/Gen.pt')
                    torch.save(netD.state_dict(), f'{path}/Disc.pt')
                    # wandb_save_models(f'{path}/Disc.pt')
                    # wandb_save_models(f'{path}/Gen.pt')
                    noise = torch.randn(3, nz, lz, lz, device=device)
                    img = netG(noise).detach()
                    plot_img(img, i, epoch, path, offline)
                    progress(i, iters, epoch, num_epochs,
                                timed=np.mean(times))
                        
                    times = []


    print("TRAINING FINISHED")
