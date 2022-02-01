# GAN-boilerplate

A boilerplate repo for GAN projects.

This repo is geared towards GANs for material microstructure projects, where the generator learned to output a homogeneous image. However, it can be easily adapted to any GAN project.

## Folder structure

```
GAN-boilerplate
 ┣ src
 ┃ ┣ __init__.py
 ┃ ┣ networks.py
 ┃ ┣ postprocessing.py
 ┃ ┣ preprocessing.py
 ┃ ┣ test.py
 ┃ ┣ train.py
 ┃ ┗ util.py
 ┣ data
 ┃ ┗ example.png
 ┣ .gitignore
 ┣ config.py
 ┣ main.py
 ┣ README.md
 ┗ requirements.txt
```

## Quickstart

Prerequisites:

- conda
- python3

Create a new conda environment, activate and install pytorch

_Note: cudatoolkit version and pytorch install depends on system, see [PyTorch install](https://pytorch.org/get-started/locally/) for more info._

```
conda create --name gan-boilerplate
conda activate gan-boilerplate
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -r requirements.txt
```

Create a .env file to hold secrets, the .env file must include

```
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=
```

You are now ready to run the repo. To start training

```
python main.py train -t test-run
```

This will track your run online with Weights and Biases and name your training run `test-run`. To run in offline mode

```
python main.py train -t test-run -o
```

To generate samples from a trained generator

```
python main.py generate -t test-run
```

To run unit tests

```
python main.py test
```

TODO

- [x] Quickstart
- [ ] Saving and loading models
- [ ] Network architectures
