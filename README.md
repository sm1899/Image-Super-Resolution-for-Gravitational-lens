# SRGAN for Gravitational Lens Image Super-Resolution

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.7+-red.svg)](https://pytorch.org/)

This repository contains an implementation of the Super-Resolution Generative Adversarial Network (SRGAN) for enhancing the resolution of gravitational lens images. The model is trained on simulated gravitational lens images and then fine-tuned using transfer learning to super-resolve real gravitational lens images.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The SRGAN architecture consists of a generator and a discriminator network. The generator takes a low-resolution image as input and generates a high-resolution version of the image. The discriminator is trained to distinguish between real high-resolution images and generated high-resolution images. The generator and discriminator are trained adversarially, resulting in the generator producing realistic high-resolution images.

## Model Architecture

### Generator

The generator network is composed of the following components:

- Initial convolutional block
- Residual blocks
- Convolutional block
- Upsampling block
- Final convolutional layer

The generator takes a low-resolution image as input and progressively upscales it to produce a high-resolution output.

### Discriminator

The discriminator network consists of a series of convolutional blocks followed by a classification head. The convolutional blocks extract features from the input image, and the classification head predicts whether the input is a real high-resolution image or a generated one.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/yourusername/srgan-gravitational-lensing.git
   cd srgan-gravitational-lensing
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Usage


## Results

The SRGAN model achieves significant improvement in the resolution and clarity of gravitational lens images. The generated high-resolution images exhibit enhanced details and structures compared to the original low-resolution images.

### Simulated Data

Here are some example results of the SRGAN model on simulated gravitational lens images:

![Simulated Image 1](images/img1_sim.png)
![simulated Image 2](images/img2_sim.png)

### Real Data

Here are some example results of the SRGAN model with transfer learning on real gravitational lens images:

![Real Image 1](images/img1_real.png) 
![Real Image 2 ](images/img2_real.png)

The SRGAN model achieves significant improvement in the resolution and clarity of gravitational lens images. The generated high-resolution images exhibit enhanced details and structures compared to the original low-resolution images.


## Citation

If you use this code or find it helpful for your research, please cite the original SRGAN paper:

```bibtex
@inproceedings{ledig2017photo,
  title={Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
  author={Ledig, Christian and Theis, Lucas and Huszar, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4681--4690},
  year={2017}
}
```
