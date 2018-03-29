'''
Main function for training and testing.

Inspired from: https://github.com/bfortuner/pytorch_tiramisu/blob/master/train.ipynb

TODO: better preprocessing and transformation (RandomCrop? RandomVerticalFlip? ..)
TODO: write training methods (including validation)
TODO: write postprocessing methods. (i.e. how to get labels and put them in a file?)
'''

from datasets import getCilia
from torchvision import transforms
from torch.utils import data

ROOT = '/media/data2TB/jeremyshi/data/cilia/'

train_joint_transformer = joint_transforms.Compose([
    joint_transforms.RandomSizedCrop(256),
    joint_transforms.RandomHorizontallyFlip()
    ])

# transform input
img_transform = transforms.Compose([
    transforms.ToTensor()
])

# using the same transformation process now; just for test-run.
cilia = getCilia.CiliaData(ROOT,
                input_transform=img_transform,
                target_transform=img_transform)

# loading the data using native PyTorch loader
train_loader = data.DataLoader(cilia, batch_size=5, shuffle=True)

# To test whether cilia can be iterated.
input_a, target_a = next(iter(cilia))

# if we want to see the results
plt.imshow(input_a[1, 0, :, :], cmap='gray')
plt.imshow(target_a[1, 0, :, :], cmap='gray')
