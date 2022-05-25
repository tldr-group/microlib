from imageio import get_reader, get_writer
import numpy as np    
import os
from PIL import Image
from random import shuffle
#Create reader object for the gif
gifs_all = [get_reader(f'runs/{file}/movie.gif') for file in sorted(os.listdir('runs'))]
gifs = []
for gif in gifs_all:
    if len(np.unique(gif.get_data(0))) < 10:
        print(gif.get_data(0).shape)
        gifs.append(gif)

ngifs = len(gifs)
gifs = gifs[5:-1]
print(f'{ngifs} gifs loaded')
max_width = 8
x, y = gifs[0].get_data(0).shape[:2]
n_frames = gifs[0].get_length()
#If they don't have the same number of frame take the shorter

#Create writer object
new_gif = get_writer('output_short2.gif', duration = 0.1)
print(f'max width {max_width}')
for width in range(1, max_width + 1):
    print(width)
    shuffle(gifs)
    for frame_number in range(n_frames):
        frames = []
        for gif in gifs[:width**2]:
            frame = gif.get_data(frame_number)

            # if len(frame.shape) > 2:
            #     frame = frame[...,0]
            # frame = frame[16:-16, 16:-16]
            
            x, y =  frame.shape[:2]
            
            # new_frame = np.zeros((maxdim, maxdim, frame.shape[-1]), dtype=np.uint8)
            ys = (y-x)//2
            frame = frame[:, ys:ys+x]
            frame =np.array(Image.fromarray(frame).resize(size=(496,496)))
            border_frame = np.zeros((512, 512, frame.shape[-1]))
            border_frame[8:-8, 8:-8] = frame
            frames.append(border_frame)
        rows = []
        for i in range(width):
            st = i
            fin = i+1
            rows.append(np.hstack(frames[st*width:fin*width]))
        img = np.vstack(rows)
        
        new_gif.append_data(np.array(Image.fromarray(img.astype(np.uint8)).resize(size=(512,512))))
import moviepy.editor as mp

clip = mp.VideoFileClip("output_short2.gif")
clip = clip.speedx(final_duration=40) 
clip.write_videofile("inpaint_concat2.mp4", fps=24)
# new_gif.close()