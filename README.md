# M3-CFR

M3-CFR is a bi-static mmWave MIMO channel frequency response (CFR) dataset for fine-grained micro-gesture recognition under domain shifts. The dataset and paper target communication-native sensing in integrated sensing and communications (ISAC), using CFR captured from an 8 x 8 fully digital mmWave MIMO testbed rather than radar-specific sensing hardware.

This repository currently contains:

- a PyTorch dataset loader and windowing utility in `m3cfr_datagen.py`
- a VGG-16 baseline training script in `train_vgg16_v2.py`
- the accompanying paper describing the dataset and benchmark task

Dataset download: https://huggingface.co/datasets/foysalhaque/M3-CFR/tree/main


## Citation

If you use M3-CFR, please cite the paper:

```bibtex
@article{haque2026m3cfr,
  title={M3-CFR: A Domain-Adaptive Bi-Static mmWave MIMO CFR Dataset for Micro-Gesture Recognition},
  author={Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  journal={IEEE Sensors Letters},
  year={2026}
}
```



## Overview

M3-CFR dataset contains:

- 10 micro-gesture classes
- 3 subjects
- 3 indoor environments
- 3,000 labeled gesture instances
- about 1.5 million CFR frames in total
- 8 x 8 MIMO CFR measurements at 58 GHz with 1 GHz bandwidth
- 1024 OFDM subchannels per CFR snapshot


## Micro-gestures

The dataset covers the following gesture categories:

- thumbs up
- thumbs down
- victory
- okay
- pointing
- pinch
- tap fingers
- fingers crossed
- fist
- half open palm


## Expected Dataset Layout

The dataset loader expects a directory structure of the form:

```text
<root_dir>/
|-- <env>/
|   `-- <subject>/
|       |-- Finger-Crossed/
|       |-- Fist/
|       |-- half-palm/
|       |-- okay/
|       |-- pinch/
|       |-- pointing/
|       |-- tap-fingers/
|       |-- thumbs-down/
|       |-- thumbs-up/
|       `-- victory/
```

Each gesture folder contains `.mat` files. The loader assumes each file stores one CFR tensor with shape:

```text
1024 x 8 x 8 x frames
```

The code automatically picks the first non-metadata variable from each `.mat` file.

## Data Loader

`m3cfr_datagen.py` provides:

- `M3CFRDataset`, a PyTorch `Dataset` for loading gesture instances
- temporal window extraction with configurable `window_size` and `stride`
- conversion from complex CFR to concatenated real/imaginary features
- per-sample normalization
- `create_train_test_split()` for a simple random train/test split

For each sample, the loader:

1. loads a CFR tensor from a `.mat` file
2. extracts a temporal window
3. reshapes `8 x 8` MIMO links into `64` spatial streams
4. concatenates real and imaginary parts
5. returns a tensor of shape `Tw x 1024 x 128`

## Training Baseline

`train_vgg16_v2.py` is an example training script built around `torchvision.models.vgg16_bn`.

It currently:

- trains on one selected environment and one selected subject at a time
- uses `window_size=10` and `stride=5`
- performs a random 80/20 train/test split
- runs a one-batch overfit sanity check before full training
- saves the best checkpoint as `best_vgg16_bn_<env>_<subject>.pth`


## Run Training

After downloading and extracting the dataset into the expected folder structure:

```bash
python train_vgg16_v2.py
```