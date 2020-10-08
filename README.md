# IcoSweepNet using CrownConv
[![arXiv](https://img.shields.io/badge/arXiv-2007.06891-b31b1b.svg)](https://arxiv.org/abs/2007.06891)
![GitHub](https://img.shields.io/github/license/matsuren/crownconv360depth)
![YouTube Video Views](https://img.shields.io/youtube/views/_vVD-zDMvyM)

PyTorch implementation of our IROS 2020 paper 
[360° Depth Estimation from Multiple Fisheye Images with Origami Crown Representation of Icosahedron](#). 
The preprint is available in [arXiv](https://arxiv.org/abs/2007.06891).

[![iros_video](https://img.youtube.com/vi/_vVD-zDMvyM/0.jpg)](https://youtu.be/_vVD-zDMvyM)


## Publication
Ren Komatsu, Hiromitsu Fujii, Yusuke Tamura, Atsushi Yamashita and Hajime Asama, "360° Depth Estimation from Multiple Fisheye Images with Origami Crown Representation of Icosahedron", Proceedings of the 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2020), 2020.

## Installation
We recommend you to use `conda` to install dependency packages.
First, run the following command to create virtual environment with dependency packages.
```bash
conda env create -f environment.yml
# enter virtual env
conda activate crownconv
```

Next, install PyTorch based on your cuda version. If you are using CUDA 9.2, run the following command:
```bash
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

Also, you need additional library to undistort fisheye images, which is installed by running the following command: 
```bash
pip install git+git://github.com/matsuren/ocamcalib_undistort.git
```

## Dataset
Please download datasets from [Omnidirectional Stereo Dataset](http://cvlab.hanyang.ac.kr/project/omnistereo/).
We use `OmniThings` for training and `OmniHouse` for evaluation.

:exclamation:Attention:exclamation:  
For some reasons, some filenames are inconsistent in `OmniThings`.
For instance, the first image is named `00001.png` in `cam1`, but, it is named `0001.png` for `cam2`, `cam3`, and `cam4`. So please rename `0001.png`, `0002.png`, and `0003.png` so that they have five-digit numbers.


## Training
```bash
python train.py $DATASETS/omnithings 
```
_Type `python train.py -h` to display other available options._

## Evaluation

One of the pretrained models is available [here](https://1drv.ms/u/s!Ao6oo2sMuht4tyrqjJbchskcqM-r?e=9TbnH5).

```bash
python evaluation.py $DATASETS/omnihouse checkpoints/checkpoints_{i}.pth --save_depth
```
_Type `python train.py -h` to display other available options._

The more detailed information is coming soon!