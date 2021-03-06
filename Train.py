# %% import library
from EdgeNet import EdgeNet
from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomResizedCrop, RandomRotation, \
    RandomHorizontalFlip, Normalize
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
parser.add_argument("--txt", help='path to the text file', default='filelist.txt')
parser.add_argument("--img", help='path to the images tar archive (uncompressed) or extracted folder', default='data')
parser.add_argument("--txt_t", help='path to the text file of test set', default='filelist.txt')
parser.add_argument("--img_t", help='path to the images tar archive (uncompressed) or extracted folder of test set',
                    default='data')
parser.add_argument("--bs", help='int number as batch size', default=128, type=int)
parser.add_argument("--es", help='int number as number of epochs', default=10, type=int)
parser.add_argument("--nw", help='number of workers (1 to 8 recommended)', default=4, type=int)
parser.add_argument("--lr", help='learning rate of optimizer (=0.0001)', default=0.0001, type=float)
parser.add_argument("--cudnn", help='enable(1) cudnn.benchmark or not(0)', default=0, type=int)
parser.add_argument("--pm", help='enable(1) pin_memory or not(0)', default=0, type=int)
args = parser.parse_args()

cudnn.benchmark = False
pin_memory = False
cudnn.benchmark = True if args.cudnn == 1 else False
pin_memory = True if args.cudnn == 1 else False


# %% define data sets and their loaders
custom_transforms = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.2)),
    RandomRotation(degrees=(-30, 30)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    RandomNoise(p=0.5, mean=0, std=0.1)])

train_dataset = PlacesDataset(txt_path=args.txt,
                              img_dir=args.img,
                              transform=custom_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.bs,
                          shuffle=True,
                          num_workers=args.nw,
                          pin_memory=pin_memory)

test_dataset = PlacesDataset(txt_path=args.txt_t,
                             img_dir=args.img_t,
                             transform=ToTensor())

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=False)


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


# %% train model
def train_model(net, data_loader, optimizer, criterion, epochs=10):
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
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            y_descreen = data['y_descreen']
            y_e = data['y_edge']

            y_descreen = y_descreen.to(device)
            y_e = y_e.to(device)

            optimizer.zero_grad()

            outputs = net(y_descreen)
            loss = criterion(outputs, y_e.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
    print('Finished Training')


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
            y_descreen = data['y_descreen']
            y_e = data['y_edge']

            y_descreen = y_descreen.to(device)
            y_e = y_e.to(device)
            outputs = net(y_descreen)
            loss = criterion(outputs, y_e)
            running_loss += loss
            print('loss: %.3f' % running_loss)
    return outputs


def show_batch_image(image_batch):
    """
    Show a sample grid image which contains some sample of test set result

    :param image_batch: The output batch of test set
    :return: PIL image of all images of the input batch
    """

    to_pil = ToPILImage()
    fs = []
    for i in range(len(image_batch)):
        img = to_pil(image_batch[i].cpu())
        fs.append(img)
    x, y = fs[0].size
    ncol = int(np.ceil(np.sqrt(len(image_batch))))
    nrow = int(np.ceil(np.sqrt(len(image_batch))))
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * int(i / nrow), y * (i % nrow)
        cvs.paste((fs[i]), (px, py))
    cvs.save('out.png', format='png')
    cvs.show()


# %% run model
criterion = EdgeLoss().to(device)
edgenet = EdgeNet().to(device)
optimizer = optim.Adam(edgenet.parameters(), lr=args.lr)
edgenet.apply(init_weights)
train_model(edgenet, train_loader, optimizer, criterion, epochs=args.es)
show_batch_image(test_model(edgenet, test_loader))
