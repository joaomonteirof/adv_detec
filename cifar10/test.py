from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import vgg, resnet, densenet

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Training settings
parser = argparse.ArgumentParser(description='Test CIFAR10 model')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='vgg')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# https://github.com/kuangliu/pytorch-cifar

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform_test = transforms.ToTensor()

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

if args.model == 'vgg':
	model = vgg.VGG('VGG16', soft=args.soft)
elif args.model == 'resnet':
	model = resnet.ResNet18(soft=args.soft)
elif args.model == 'densenet':
	model = resnet.densenet_cifar(soft=args.soft)

model_id = args.model_path + 'cifar_' + args.model + ('_soft' if args.soft else '') + '.pt'

mod_state = torch.load(model_id, map_location = lambda storage, loc: storage)
model.load_state_dict(mod_state['model_state'])

if args.cuda:
	model.cuda()

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for t, data, target in enumerate(test_loader):
		#data = (data-data.max(0)[0])/(data.max(0)[0]-data.min(0)[0])
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss
	test_loss /= len(test_loader) # loss function already averages over batch size

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])
	
	plt.show()


history = mod_state['history']

plot_learningcurves(history, 'train_loss')
plot_learningcurves(history, 'valid_loss')
plot_learningcurves(history, 'valid_acc')

test()
