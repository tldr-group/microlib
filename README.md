# Microlib

A repo for generating the dataset associated with microlib.io. 

Website: https://microlib.io/

Paper: ___

## Folder structure

```
microlib
 ┣ src
 ┃ ┣ preprocessing
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ annotations.gui.py
 ┃ ┃ ┣ inpaint_scalebars.py
 ┃ ┃ ┣ inpaint.py
 ┃ ┃ ┣ networks.py
 ┃ ┃ ┣ util.py
 ┃ ┣ inpainting
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ config.py
 ┃ ┃ ┣ inpaint_scalebars.py
 ┃ ┃ ┣ inpaint.py
 ┃ ┃ ┣ networks.py
 ┃ ┃ ┣ util.py
 ┃ ┣ slicegan
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ model.py
 ┃ ┃ ┣ network.py
 ┃ ┃ ┣ preprocessing.py
 ┃ ┃ ┣ run_slicegan.py
 ┃ ┃ ┣ util.py
 ┃ ┣ postprocessing
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ figures.ipynb
 ┃ ┃ ┣ mergegifs.py
 ┣ data
 ┃ ┣ prelabelled_anns.json
 ┃ ┣ micrographs_raw*
 ┃ ┣ micrographs_final*
 ┃ ┣ inpaint_runs*
 ┃ ┣ slicegan_runs*
 ┃ ┣ anns.json*
 ┣ .gitignore
 ┣ LICENSE.txt
 ┣ main.py
 ┣ README.md
 ┗ requirements.txt

*folders and files generated during the processing steps
```


## Repo setup

Prerequisites:

- conda
- python3

Create a new conda environment, activate and install pytorch

_Note: cudatoolkit version and pytorch install depends on system, see [PyTorch install](https://pytorch.org/get-started/locally/) for more info._

```
conda create --name microlib
conda activate microlib
conda install pytorch torchvision -c pytorch
conda install -r requirements.txt
```
## Dataset generation

You are now ready to run the repo. We will download images, annotate them, perform inpainting, run slicegan and finally generate some animations.

First, to download images run in import mode. This will create a series of requests to doitpoms. If you get cert errors, go to src/preprocessing/import_data.py and add verify=False to line 18 *at your own risk*.

```
python main.py import
```

Next, annotate the images by running in preprocess mode. You can skip this step and use our annotations by renaming data/prelabelled_anns.json to data/anns.json. If you quit the gui and rerun, you will automatically continue from where you left off - to restart, just delete anns.json.

```
python main.py preprocess
```

The following are the controls at different stages of the annotation GUI. At any time, press C to restart the current microstructure, or W to remove the current microstructure if it doesn't fit the exclusion criteria'. The stage you are on is shown at the top of the gui. These are the stages:

1. Scale bar col: click on the scale bar then use A and S keys to adjust thresholds, or press enter to skip.
2. Scale bar box: click on the top left then bottom right corners of the reqion containing the scale bar, or press enter to skip. You should not skip this if you have selected a scale bar col.
3. Crop region: click on the top left then bottom right corner to define the region you want to keep. Click a third time to reset. Press enter to skip.
4. Click on the different phases to segment. Use A and S to adjust threshold. Press enter to skip and select grayscale.
5. Voxel size: click on the left of the scalebar, then the right, then enter scale bar size in microns.


Now run in inpaint mode. This creates a repo called final_images with all the inpainted images ready for slicegan, as well as any images that didn't need inpainting

```
python main.py inpaint
```

Run in slicegan mode to train 3D generators. This creates the data/slicegan_runs folder and a subfolder for each run that will contain the generator and discriminator, params, and the animations and 3D volumes generated in the next step.

```
python main.py slicegan
```

Finally, run in animate mode to generate a 3D volume and animate it slice by slice and rotating. Note that th

```
python main.py animate
```

## Using pretrained generators

This repo is centred around how users can follow the steps we took to generate the full 3D dataset in microlib. If instead you are interested in using the pretrained generators to make more microstructures of different sizes or shapes, you should instead use the SliceGAN repo.

To do so, first clone SliceGAN from here: https://github.com/stke9/SliceGAN. Create the following additional folder within the repo, where microxxx.Gen and microxxx.params can be downloaded from microlib.io by clicking on a microstructure of interest:

```
SliceGAN
 ┣ TrainedGenerators
 ┃ ┣ microxxx
 ┃ ┃ ┣ microxxx.Gen
 ┃ ┃ ┣ microxxx.params
```




