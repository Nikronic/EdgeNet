# %% import library
from EdgeNet import EdgeNet
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip
from utils.preprocess import *
import torch
from torch.utils.data import DataLoader
from utils.loss import EdgeLoss

import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% config parser
parser = argparse.ArgumentParser()
parser.add_argument("--txt", help='path to the text file', default='data/filelist.txt')
parser.add_argument("--img", help='path to the images tar archive (uncompressed)', default='data/data.tar')
parser.add_argument("--bs", help='int number as batch size', default=128, type=int)
parser.add_argument("--es", help='int number as number of epochs', default=25, type=int)
parser.add_argument("--nw", help='number of workers (1 to 8 recommended)', default=4, type=int)
parser.add_argument("--lr", help='learning rate of optimizer (=0.0001)', default=0.0001, type=float)
parser.add_argument("--cudnn", help='enable(1) cudnn.benchmark or not(0)', default=0, type=int)
parser.add_argument("--pm", help='enable(1) pin_memory or not(0)', default=0, type=int)
args = parser.parse_args()

if args.cudnn == 1:
    cudnn.benchmark = True
else:
    cudnn.benchmark = False

if args.pm == 1:
    pin_memory = True
else:
    pin_memory = False


# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    RandomNoise(p=0.5, mean=0, std=0.1)])

train_dataset = PlacesDataset(txt_path=args.txt,
                              img_dir=args.img,
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.bs,
                          shuffle=True,
                          num_workers=args.nw,
                          pin_memory=pin_memory)

# %% initialize network, loss and optimizer
def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"

    :param m: Layer to initialize
    :return: None
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm2d):  # reference: https://github.com/pytorch/pytorch/issues/12259
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

criterion = EdgeLoss()
edgenet = EdgeNet().to(device)
optimizer = optim.Adam(edgenet.parameters(), lr=args.lr)
edgenet.apply(init_weights)

# %% train model
def train_model(net, data_loader, optimizer, criterion, epochs=128):
    """
    Train model

    :param net: Parameters of defined neural network
    :param data_loader: A data loader object defined on train data set
    :param epochs: Number of epochs to train model
    :param optimizer: Optimizer to train network
    :param criterion: The loss function to minimize by optimizer
    :return: None
    """

    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            X = data['X']
            y_e = data['y_edge']

            X = X.to(device)
            y_e = y_e.to(device, dtype=torch.int64)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(X)
            loss = criterion(outputs, y_e)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
    print('Finished Training')


train_model(edgenet, train_loader, optimizer, criterion, epochs=args.es)

# %% test model

for i in range(len(train_dataset)):
    sample = train_dataset[i]

    X = sample['X']
    y_e = sample['y_edge']

    print(X.size())
    print(y_e.size())

    if i == 0:
        break

# %% test
def test_model(net, data_loader):
    """
    Return loss on test

    :param net: The trained NN network
    :param data_loader: Data loader containing test set
    :return: Print loss value over test set in console
    """
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            X = data['X']
            y_e = data['y_edge']
            X = X.to(device)
            y_e = y_e.to(device)
            outputs = net(X)
            loss = criterion(outputs, y_e)
            running_loss += loss

            print('loss: %.3f' % running_loss)
    return running_loss, outputs

# test_model(coarsenet, train_loader)
