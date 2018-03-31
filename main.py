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

cilia = getCilia.CiliaData(ROOT,
                joint_transform = train_joint_transformer,
                input_transform = img_transform,
                target_transform = img_transform)
train_loader = data.DataLoader(cilia, batch_size=5, shuffle=True)


val_cilia = CiliaData(ROOT, 'validate',
                  joint_transform = None,
                  input_transform=img_transform,
                  target_transform=img_transform
                 )

val_loader = torch.utils.data.DataLoader(
    val_cilia, batch_size=1, shuffle=True)

# using the same transformation process now; just for test-run.


test_cilia = CiliaData(ROOT, 'test',
                  joint_transform = None,
                  input_transform=img_transform
                 )

# loading the data using native PyTorch loader


model = tiramisu.FCDenseNet67(n_classes=3).cuda()
model.apply(training_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss().cuda()

for epoch in range(1, N_EPOCHS+1):
    ### Train ###
    trn_loss, trn_err = training_utils.train(
        model, train_loader, optimizer, criterion, epoch)
    print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch, trn_loss, 1-trn_err))

    ### Test ###
    val_loss, val_err = training_utils.test(model, val_loader, criterion, epoch)
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))

    ### Checkpoint ###
    training_utils.save_weights(model, epoch, val_loss, val_err)

    ### Adjust Lr ###
    training_utils.adjust_learning_rate(LR, LR_DECAY, optimizer,
                                     epoch, DECAY_EVERY_N_EPOCHS)


# post-processing -- put png results into .results/ folder
for i, pic in enumerate(test_loader):
    pred = training_utils.get_test_results(model, pic)
    pred_img = pred[0, :, :]
    pred_img[pred_img == 1] = 0
    imwrite('.results/' + test_dir[i] + '.png', pred_img.numpy().astype(np.uint8))
