from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Adv attacks/defenses on CIFAR10')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--l2', type=float, default=0.00001, metavar='lambda', help='L2 wheight decay coefficient (default: 0.00001)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='vgg')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# https://github.com/kuangliu/pytorch-cifar

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if args.model == 'vgg':
	model = vgg.VGG('VGG16', soft=args.soft)
elif args.model == 'resnet':
	model = resnet.ResNet18(soft=args.soft)
elif args.model == 'densenet':
	model = resnet.densenet_cifar(soft=args.soft)

model_id = 'cifar10_'+args.model+('_soft' if args.soft else '')

if args.cuda:
	model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

best_test_acc = -np.inf

def save_model():
	print('Saving model...')
	ckpt = {'model_state': model.state_dict()}
	torch.save(ckpt, model_id+'.pt')

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data_v, target_v = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data_v)
		loss = F.nll_loss(output, target_v)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data = (data-data.max(0)[0])/(data.max(0)[0]-data.min(0)[0])
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss
	test_loss /= len(test_loader) # loss function already averages over batch size

	if correct>best_test_acc:
		best_test_acc = correct
		save_model()

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
	train(epoch)
	test(epoch)
