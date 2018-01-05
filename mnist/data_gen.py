from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import attacks
from attacks import fgsm, igsm, deepfool, c_w_L2

# Training settings
parser = argparse.ArgumentParser(description='Adv attacks/defenses on MNIST')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model-path', type=str, default='./models/', metavar='Path', help='Path for model load')
parser.add_argument('--model', choices=['cnn', 'mlp'], default='cnn')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
parser.add_argument('--method', choices=['fgsm', 'igsm', 'deepfool', 'cw'], default='fgsm')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--test', action='store_true', default=False, help='Test data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=not args.test, download=True, transform=transforms.ToTensor()), batch_size=1, shuffle=True, **kwargs)

if args.model == 'cnn':
	if args.soft:
		model = models.cnn_soft()
	else:
		model = models.cnn()
elif args.model == 'mlp':
	if args.soft:
		model = models.mlp_soft()
	else:
		model = models.mlp()

'''
if args.method == 'fgsm':
	attacker = fgsm
elif args.method == 'igsm':
	attacker = igsm
elif args.method == 'deepfool':
	attacker = deepfool
elif args.method == 'cw':
	attacker = c_w_L2
'''

model_id = args.model_path + 'mnist_' + args.model + ('_soft' if args.soft else '') + '.pt'

mod_state = torch.load(model_id)
model.load_state_dict(mod_state['model_state'])

if args.cuda:
	model.cuda()

to_pil = transforms.ToPILImage()

for i, data in enumerate(loader):

	clean_sample, target = data

	attack_sample = fgsm(model=model, x=clean_sample, y=target, eps=0.25)
	pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
	sample_pil = to_pil(attack_sample[0])
	sample_pil.save('./data/{}.bmp'.format(i))
	print(i)
