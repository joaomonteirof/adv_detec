from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop_oltl import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Cifar10 Classification')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--valid-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--patience', type=int, default=10, metavar='N', help='How many epochs without improvement to wait before reducing the LR (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--l2', type=float, default=5e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='lambda', help='Momentum (default: 0.9)')
parser.add_argument('--ngpus', type=int, default=0, help='Number of GPUs to use. Default=0 (no GPU)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data .hdf')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
args = parser.parse_args()
args.cuda = True if args.ngpus>0 and torch.cuda.is_available() else False

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
transform_test = transforms.ToTensor()

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.valid_batch_size, shuffle=False, num_workers=2)

model_1 = vgg.VGG('VGG16', soft=False)
model_2 = resnet.ResNet18(soft=False)

if args.cuda:
	model_1 = model_1.cuda()
	model_2 = model_2.cuda()

optimizer_1 = optim.SGD(model_1.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.momentum)
optimizer_2 = optim.SGD(model_2.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.momentum)

trainer = TrainLoop(model_1, model_2, optimizer_1, optimizer_2, train_loader, test_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, patience=args.patience)
