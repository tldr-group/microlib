from copy import deepcopy
import json
import matplotlib.pyplot as plt
from src.inpainting.inpaint import train, generate
import numpy as np
import os
def inpaint_samples(mode):
    if not os.path.exists('inpaint_runs'):
        os.mkdir('inpaint_runs')
    with open(f'data/anns.json', 'r') as f:
        data_map = json.load(f)
    for key in data_map.keys():
        load_sample(data_map[key], mode)

def load_sample(s, mode):
    try:
        pth = s['data_path']
    except:
        return
    if not pth:
        return
    bar_box = s['barbox'] 
    if not bar_box:
        return
    
    crop = s['crop'] 
    phases = [ph / 255 for ph in s['phases']]
    img = plt.imread(pth)[...,0]
    x0, y0 = bar_box[0]
    x1, y1 = bar_box[1] 
    x1, y1 = x1, y1
    rect = [(y0, x0, y1, x1)]
    mask = np.zeros_like(img)
    mask[y0:y1, x0:x1] = 1
    if s['barcol'] != None:
        bar_col = s['barcol'][0] / 255
        bar_col_var = s['barcol'][1]/2550
        mask_ip = np.zeros_like(img)
        mask_cropped = abs(img[y0:y1, x0:x1]-bar_col) < bar_col_var
        mask_ip[y0:y1, x0:x1][mask_cropped] = 1
    else:
        mask_ip = deepcopy(mask)
    for sh in [1, -1]:
        mask_ip += np.roll(mask_ip, sh, axis=0)
        mask_ip += np.roll(mask_ip, sh, axis=1)
    mask_ip[mask_ip > 1] = 1
    if len(phases) != 0:
        img = oh(phases, pth)
    pth = 'data/temp.png'
    plt.imsave(pth, img)
    if crop:
        x0, y0 = crop[0]
        x1, y1 = crop[1]
        img = img[y0:y1, x0:x1]
    imtype = 'grayscale' if len(phases)==0 else 'n-phase'
    tag = s['data_path'][-7:-4]
    tag = f'micro{tag}'
    print(f'training {tag}')
    if mode == 'train':
        train(img, imtype, mask, mask_ip, rect, pth, tag)
        return
    if not os.path.exists('data/micrographs_final'):
        os.mkdir('data/micrographs_final')
    return generate(img, imtype, mask, mask_ip, rect, pth, tag)

def oh(phases, pth):
    img_oh = plt.imread(pth)[...,0]
    boundaries = [0]
    for ph1, ph2 in zip(phases[:-1], phases[1:]):
        boundaries.append(ph1 + (ph2 - ph1)/2)
    boundaries.append(1)
    for i, (b_low, b_high) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        img_oh[(img_oh >= b_low) & (img_oh <= b_high)] = i
    return img_oh


        
