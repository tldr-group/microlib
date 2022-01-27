import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
import torch.nn as nn
import time


def post_process(img):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()

    return img



def generate(c, netG):
    """Generate an instance from generator, save to .tif

    :param c: Config object class
    :type c: Config
    :param netG: Generator instance
    :type netG: Generator
    :return: Post-processed generated instance
    :rtype: torch.Tensor
    """
    tag, ngpu, nz, lf, pth = c.tag, c.ngpu, c.nz, c.lf, c.path


    out_pth = f"runs/{tag}/out.tif"
    if torch.cuda.device_count() > 1 and c.ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
        torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{pth}/nets/Gen.pt"))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf, lf)
    raw = netG(noise)
    gb = post_process(raw)
    tif = np.array(gb[0], dtype=np.uint8)
    tifffile.imwrite(out_pth, tif, imagej=True)
    return tif

