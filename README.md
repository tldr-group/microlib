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
conda install pytorch torchvision -c pytorch
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

## Saving, loading and overwriting models

Models are saved to runs folder which is generated when training initiates. Inside runs, a new folder with the name of your training run tag will be generated, inside this the model params and training outputs are saved. This includes:

- **config.json** - this json holds the config paramaters of your training run, see config.py for more info
- **Gen.pt** - this holds the generator training parameters
- **Disc.pt** - this holds the discriminator parameters

### Training

If training for the first time, these files are created and updated during training.

If you initiate a training run with a tag of a run that already exists you will see the prompt

```
To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.
```

By entering `'o'` you will overwrite the existing models, deleting their saved parameters and config. `l` will load the existing model params and config, and continue training this model. `c` will abort the training run.

### Evaluation

When evaluating a trained model, the params and model config are loaded from files. Models are saved with their training tag, use this tag to evaluate specific models.

## TODO

- [x] Quickstart
- [x] Saving and loading models
- [ ] Training outputs
- [ ] Network architectures
- [ ] wandb
