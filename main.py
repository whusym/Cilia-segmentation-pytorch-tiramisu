'''
Main function for training and testing.

Inspired from: https://github.com/bfortuner/pytorch_tiramisu/blob/master/train.ipynb,
and https://github.com/ZijunDeng/pytorch-semantic-segmentation.

I highly recommend using the interactive Jupyter Notebook to run this pipeline. It is
provided in the demo.ipynb under the same root directory in the repo. This Python file is
mainly for the purpose of making this project complete and helping people without access to
Jupyter Notebook.

To start, just run
    python main.py -p <your-data-folder-path>

The data folder structure should be
--train
--|--data
--|--masks
--validate
--|--data
--|--masks
--test
--|--data

The file will automatically store the weight
NB: this file does not support CPU in training and testing. To see how to load a model on
your laptop and use the CPU to generate the results, check the Juypter Notebook.
'''

from datasets import getCilia
from torchvision import transforms
from torch.utils import data
from utils import joint_transforms
from utils import training_utils
import torch.nn as nn
from models import tiramisu
import torchvision
from imageio import imwrite
from imageio import imread
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import numpy as np
import argparse


# Some global params for training
LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 3000
torch.cuda.manual_seed(0) # use manual seed for reproductivity
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)


if __name__ == '__main__':

    # Check GPU
    try:
        print ('Your GPU is {}'.format(torch.cuda.get_device_name(0)))
    except AttributeError:
        print ('No GPU found!')

    parser = argparse.ArgumentParser(description='P4-Celia Segmentation')
    parser.add_argument('-p','--path', dest='root', type=str,
                    help='Root path for the data folder',
                    default='/media/data2TB/jeremyshi/data/cilia/')

    # Specify the path for folder
    args = parser.parse_args()
    ROOT = args.root

    # Joint_tranformation for training inputs and targets
    train_joint_transformer = joint_transforms.Compose([
        joint_transforms.RandomSizedCrop(256),
        joint_transforms.RandomHorizontallyFlip()
        ])

    # Tranformation for training inputs and targets (change them to tensors)
    img_transform = transforms.Compose([
        transforms.ToTensor()
        ])

    cilia = getCilia.CiliaData(ROOT,
                joint_transform = train_joint_transformer,
                input_transform = img_transform,
                target_transform = img_transform,
                remove_cell = False
                )

    # Loading the training data using native PyTorch loader (same for valiation and testing later)
    train_loader = data.DataLoader(cilia, batch_size = 1, shuffle = True)
    print ("Loaded training set!")

    val_cilia = getCilia.CiliaData(ROOT, 'validate',
                joint_transform = None,
                input_transform=img_transform,
                target_transform=img_transform,
                remove_cell = False
                )

    val_loader = torch.utils.data.DataLoader(
                    val_cilia, batch_size=1, shuffle=True)
    print ("Loaded training set!")

    test_cilia = getCilia.CiliaData(ROOT, 'test',
                joint_transform = None,
                input_transform = img_transform
                )

    test_loader = torch.utils.data.DataLoader(
                test_cilia,
                batch_size=1,
                shuffle=False
                )
    print ("Loaded testing set!")

    model = tiramisu.FCDenseNet103(n_classes=3, in_channels=1).cuda()
    model.apply(training_utils.weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.NLLLoss().cuda()

    # Training process
    for epoch in range(1, N_EPOCHS+1):
        since = time.time()
        ### Train ###
        trn_loss, trn_err = training_utils.train(
            model, train_loader, optimizer, criterion, epoch)
        print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
            epoch, trn_loss, 1-trn_err))
        time_elapsed = time.time() - since
        print('Train Time {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        ### Test ###
        val_loss, val_err = training_utils.test(model, val_loader, criterion, epoch)
        print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
        time_elapsed = time.time() - since
        print('Total Time {:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        ### Checkpoint ###
        training_utils.save_weights(model, epoch, val_loss, val_err)
        ### Adjust Lr ###
        training_utils.adjust_learning_rate(LR, LR_DECAY, optimizer,
                                         epoch, DECAY_EVERY_N_EPOCHS)

    # post-processing -- put png results into .results/ folder
    test_dir = sorted(os.listdir(ROOT + 'test' + '/data/'))
    for i, pic in enumerate(test_loader):
        pred = training_utils.get_test_results(model, pic)
        pred_img = pred[0, :, :]
        pred_img[pred_img == 1] = 0
        imwrite('.results/' + test_dir[i] + '.png', pred_img.numpy().astype(np.uint8))
