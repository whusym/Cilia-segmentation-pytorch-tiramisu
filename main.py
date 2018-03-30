'''
Main function for training and testing.
For a more interactive exploration, check the Jupyter notebook.

Inspired from: https://github.com/bfortuner/pytorch_tiramisu/blob/master/train.ipynb

TODO: better preprocessing and transformation (RandomCrop? RandomVerticalFlip? ..)
TODO: write training methods (including validation)
TODO: write postprocessing methods. (i.e. how to get labels and put them in a file?)
'''

from datasets import getCilia
from torchvision import transforms
from torch.utils import data
from utils import joint_transforms
from utils import training_utils
import torch.nn as nn
from models import tiramisu
import torchvision

# Check GPU
if torch.cuda.get_device_name(0):
    print ('Your GPU is {}'.format(torch.cuda.get_device_name(0)))
else:
    print ('No GPU found!')

# Specify the path for folder
ROOT = '/media/data2TB/jeremyshi/data/cilia/'

# Joint_tranformation for training inputs and targets
train_joint_transformer = joint_transforms.Compose([
    joint_transforms.RandomSizedCrop(256),
    joint_transforms.RandomHorizontallyFlip()
    ])

# Tranformation for training inputs and targets (change them to tensors)
img_transform = transforms.Compose([
    transforms.ToTensor()
])

# using the same transformation process now; just for test-run.
cilia = getCilia.CiliaData(ROOT,
                joint_transform = train_joint_transformer,
                input_transform = img_transform,
                target_transform = img_transform)

# loading the data using native PyTorch loader
train_loader = data.DataLoader(cilia, batch_size=5, shuffle=True)
