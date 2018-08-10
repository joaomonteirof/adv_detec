from __future__ import print_function
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet

# Training settings
parser = argparse.ArgumentParser(description='10 classes Imagenet Classification')
parser.add_argument('--train-data-path', type=str, default='./data/train/', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/val/', metavar='Path', help='Path to data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N', help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='lambda', help='Momentum (default: 0.9)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['vgg', 'resnet'], default='vgg')
parser.add_argument('--pretrained', action='store_true', default=False, help='downloads pretrained features model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(args.train_data_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

valid_data = datasets.ImageFolder(args.valid_data_path, transform=valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.workers)

if args.model == 'vgg':
	features_model = torchvision.models.vgg19_bn(pretrained=args.pretrained)
	model = vgg(features_model)
elif args.model == 'resnet':
	features_model = torchvision.models.resnet50(pretrained=args.pretrained)
	model = resnet(features_model)

if args.cuda:
	model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs)
