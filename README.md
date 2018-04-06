# Semantic Segmentation on Cilia Images using Tiramisu Network

This is a PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf). This repository is created over the course of two weeks for the project 4 of CSCI 8360 Data Science Practicum at University of Georgia in Spring 2018. The goal of this project is to conduct semantic segmentation on time-series cilia motion images.

This repository is benefited from Bendan Fortuner(@bfortuner)'s implementation (https://github.com/bfortuner/pytorch_tiramisu) and ZijunDeng(@ZijunDeng)'s implementation (https://github.com/ZijunDeng/pytorch-semantic-segmentation). Huge thanks to them!

For other variations of DenseNet and the references for other preprocessing methods, check our [Wiki](https://github.com/dsp-uga/kampf/wiki) tab. (or press `g` `w` on your keyboard).

<img src="media/cilia.png" width="800" class="center">
(from the project write-up by Dr. Shannon Quinn. See https://quinngroup.github.io/people.html.)

For more detailed information, see the notebook `demo.ipynb`.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Again, the best way to get started is from the `demo.ipynb`.

### My hardware setting
- CPU: Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz
- RAM: 64 GB
- GPU: GeForce GTX 1080 Ti

### Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](http://pytorch.org/docs/master/)
- [Jupyter Notebook](http://jupyter.org/)

For other required libraries, please check `environment.yml` file.

### Environment Setting

1. Clone this repository.
```
$ git clone https://github.com/dsp-uga/kampf.git
$ cd kampf
```

2. Create conda environment based on `environments.yml` offered in this repository.
```
$ conda env create -f environments.yml -n cilia python=3.6
$ source activate cilia
```
It will build a conda environment named `cilia`, and of course you can create an environment with your favorite name.

3. Run (**in native Python**)
```
$ python main.py [-p <your-data-path>]
```
For the requirement of data folder path and its inner structure, please check the header of `main.py`.

4. Run (**in Jupyter Notebook**)
Alternatively, you can also start a Jupyter Notebook environment, just navigate to the project folder (if you are not there already), and
```
$ jupyter notebook
```
It should pop up your default browser, and the next step is just to open `demo.ipynb` and follow the instructions there.


## Results
Our best final result is 45.81168, which is an average IoU score for all 114 testing cilia videos/images. Here is an example of what our results are like:

<img src="media/result1.png">

## TODO
- More parameter tuning for the Tiramisu network.
- Trying other different methods in preprocessing and integrate more preprocessing methods before feeding processed images into the network.


## Authors
- [Maulik Shah](https://github.com/mauliknshah)
- [Yuanming Shi](https://github.com/whusym)
- [Jin Wang](https://github.com/SundayWang)

See the [contributors.md](https://github.com/dsp-uga/kampf/blob/master/contributors.md) file for detailed contributions by each team member.

## How to Contribute
We are welcome to any kind of contribution. If we want to contribute, just create a ticket!

## License
LGPL-3.0. See [LICENSE](https://github.com/dsp-uga/kampf/blob/master/LICENSE) for details.
