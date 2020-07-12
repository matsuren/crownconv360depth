# IcoSweepNet using CrownConv
PyTorch implementation of our IROS 2020 paper [360° Depth Estimation from Multiple Fisheye Images with Origami Crown Representation of Icosahedron](https://github.com/matsuren/crownconv360depth). 

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
The dataset is available from [here](http://cvlab.hanyang.ac.kr/project/omnistereo/).
We use `OmniThings` for training and `OmniHouse` for evaluation.


## Training
Coming soon!