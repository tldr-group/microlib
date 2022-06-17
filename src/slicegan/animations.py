from asyncio.log import logger
from msilib.schema import Error
from sympy import Q
from src.slicegan import networks, util
import argparse
import torch
import tifffile
import numpy as np
from plotoptix.materials import make_material
import plotoptix.materials as m
from plotoptix import NpOptiX, TkOptiX
from scipy import ndimage
import moviepy.editor as mp
from moviepy.editor import *
import matplotlib.pyplot as plt
import imageio
import os
from time import time
from time import sleep
import shutil

class Animator():
    def __init__(self):
        # Define project name
        res = 512
        min_accum = 500
        self.optix = NpOptiX(on_scene_compute=self.compute,
                        on_rt_completed=self.update,
                        width=res, height=res,
                        start_now=False)
        
        self.optix.set_param(min_accumulation_step=min_accum,
                        # 1 animation frame = 128 accumulation frames
                        max_accumulation_frames=5130,
                        light_shading="Hard")  # accumulate 512 frames when paused
        self.optix.set_uint("path_seg_range", 5, 10)
        exposure = 1
        gamma = 2.3
        self.optix.set_float("tonemap_exposure", exposure)  # sRGB tonning
        self.optix.set_float("tonemap_gamma", gamma)
        self.optix.set_float("denoiser_blend", 0.25)
        self.optix.add_postproc("Denoiser")
        self.optix.set_background(250)
        alpha = np.full((1, 1, 4), 1, dtype=np.float32)
        self.optix.set_texture_2d("mask", (255 * alpha).astype(np.uint8))
        m_diffuse_3 = make_material("Diffuse", color_tex="mask")
        self.optix.setup_material("3", m_diffuse_3)
        self.optix.start()
        self.optix.pause_compute()

    def new_animation(self, micro):
        self.Project_name = micro
        # Specify project folder.
        self.Project_path = f'data/slicegan_runs/{micro}'
        if not os.path.exists(f'{self.Project_path}/frames'):
            os.mkdir(f'{self.Project_path}/frames')
        
        self.frame_path = f'{self.Project_path}/frames'
        self.Project_path = f'data/slicegan_runs/{micro}/{micro}'
        img_size, img_channels, scale_factor = 64, 1, 1
        z_channels = 16
        lays = 6
        dk, gk = [4] * lays, [4] * lays
        ds, gs = [2] * lays, [2] * lays
        df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 64,
                                                        img_channels]
        dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

        ## Create Networks
        netD, netG = networks.slicegan_nets(self.Project_path, False, 'grayscale', dk, ds,
                                            df, dp, gk, gs, gf, gp)
        netG = netG()
        netG.eval()
        lf = 12
        n = (lf - 2) * 32
        noise = torch.randn(1, z_channels, lf, lf, lf)
        netG = netG.cuda()
        noise = noise.cuda()
        nseeds = 10
        netG.load_state_dict(torch.load(self.Project_path + '_Gen.pt'))
        
        img = netG(noise[0].unsqueeze(0))
        image_type = 'twophase' if img.shape[1] == 2 else 'grayscale'
        # image_type = 'colour'
        
        img = util.post_proc(img, image_type)
        self.img = img
        if image_type == 'twophase':
            self.ph = 0 if np.mean(img) > 0.5 else 1
        if image_type == 'twophase':
            bind = np.array(np.where(img == self.ph)).T
            c = 1 - (bind + 0.5)
            c = (0.3, 0.3, 0.3)
        elif image_type=='colour':
            bind = np.array(np.where(img[...,0] != -1)).T
            c = (img.reshape(-1, 3)) / 255
            print(c.shape, bind.shape)

        else:
            # img = ndimage.gaussian_filter(img, blur)
            bind = np.array(np.where(img != -1)).T
            # bind[:,1][img.reshape(-1) <  0] +=1000
            c = (img.reshape(-1)) / 255
        bind = bind / n - 0.5
        tf = 360
        self.f = 0
        self.fn=0
        self.s = int(360 / tf)
        self.e = [-3, 0, 0]
        self.l = [-5, 10, 0]
        self.bind = bind
        self.c = c
        self.fin = False
        self.rotating=False
        self.n = img.shape[0]
        self.image_type = image_type
        s = 1 / self.n
        self.optix.set_data("cubes_b", pos=self.bind, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
                    geom="Parallelepipeds",  # cubes, actually default geometry
                    mat="3",  # opaque, mat, default
                    c=self.c)
        self.optix.setup_camera("cam1", eye=self.e, target=[0, 0, 0], up=[0, 1, 0],
                        fov=30)
        self.optix.set_ambient((0.3, 0.3, 0.3))
        x = self.n / 2
        self.optix.resume_compute()

    def compute(self, rt: NpOptiX,
                delta: int) -> None:  # compute scene updates in parallel to the raytracing
        self.fn+=1
        if not self.rotating:
            self.f += self.s
            img = self.img[-self.f:]
            # print(img.shape)
            # print(np.array(np.where(img == self.ph)).T.shape)
        
            if self.image_type == 'twophase':
                bind = np.array(np.where(img == self.ph)).T
                c = 1 - (bind + 0.5)
                c = (0.3, 0.3, 0.3)
            elif self.image_type=='colour':
                bind = np.array(np.where(img[...,0] != -1)).T
                self.c = (img.reshape(-1, 3)) / 255

            else:
                # img = ndimage.gaussian_filter(img, blur)
                bind = np.array(np.where(img != -1)).T
                # bind[:,1][img.reshape(-1) <  0] +=1000
                self.c = (img.reshape(-1)) / 255
            self.bind = bind / self.n - 0.5
            if self.f==self.n:
                self.f=0
                self.rotating = True
            self.e = [-3, min(1.2 * (self.f/self.n), 1.2), 0]
            self.l = [-5, 10,0]
        # self.bind = bind/n - 0.5
        else:
            self.f += self.s
            f_step = self.f * np.pi * 2 / 360
            # self.e = [0.5*np.cos(self.f/360), 12, 20*np.sin(self.f/360)]
            x, y = np.cos(f_step), np.sin(f_step)

            self.e = [-3 * x, 1.2, -3 * y]
            self.l = [-5 * x, 5, -5 * y]


    # optionally, save every frame to a separate file using save_image() method
    def update(self, rt: NpOptiX) -> None:
        rt.update_camera('cam1', eye=self.e)
        # rt.update_light('light1', pos=self.l)
        rt.set_data("cubes_b", pos=self.bind, u=[1 / self.n, 0, 0], v=[0, 1 / self.n, 0],
                    w=[0, 0, 1 / self.n],
                    geom="Parallelepipeds",  # cubes, actually default geometry
                    mat="diffuse",  # opaque, mat, default
                    c=self.c)
        # rt.update_light('light1', pos=self.l)
        # print("frames/frame_{:05d}.png".format(self.f))

        # self.optix.close()
        # raise Error
        rt.save_image(self.frame_path + '/frame_{:05d}.png'.format(self.fn))
        if self.f == 360:
            self.optix.pause_compute()
            self.save_animation()
            # rt.close()
            self.fin = True
        
       
    def save_animation(self):
        
        frames = sorted(os.listdir(self.Project_path + 'frames'))[1:]
        frames = frames[:319] + frames[320:]

        end_frames = frames[:319]
        end_frames.reverse()

        frames = frames + end_frames
        fps = 45
        frame_duration = 1 / fps

        clips = [
            ImageClip(f'{self.Project_path}frames/{m}').set_duration(frame_duration)
            for m in frames]
        clips[0] = clips[0].set_duration(1)
        clip = concatenate_videoclips(clips, method="compose")
        clip.write_videofile(f'{self.Project_path}_long.mp4', fps=fps, verbose=False, logger=None, ffmpeg_params=['-movflags', 'faststart'])
def animate_dataset():
    dir = f'data/slicegan_runs'
    micros = sorted(os.listdir(dir))
    print(len(micros))
    a = Animator()
    for micro in micros:
        
        a.new_animation(micro)
        while not a.fin:
            sleep(1)
            pass
        print(f'{micro} finished')
        a.fin = False
    a.optix.close()


