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
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model-path', type=str, default='./models/', metavar='Path', help='Path for model load')
parser.add_argument('--model', choices=['cnn', 'mlp'], default='cnn')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

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

model_id = args.model_path + 'mnist_' + args.model + ('_soft' if args.soft else '') + '.pt'

mod_state = torch.load(model_id)
model.load_state_dict(mod_state['model_state'])

if args.cuda:
	model.cuda()

to_pil = transforms.ToPILImage()

ntrials = 10

for i in range(ntrials):

	index = np.random.randint(len(train_loader))

	clean_sample, target = list(train_loader)[index]

	pred = model.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()
	sample_pil = to_pil(clean_sample[0])
	sample_pil.save('./samples/{}_clean_{}_{}.png'.format(i, pred.argmax(), target[0]))

	print('Clean sample -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))


	attack_sample = fgsm(model=model, x=clean_sample, y=target, eps=0.2)
	pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
	sample_pil = to_pil(attack_sample[0])
	sample_pil.save('./samples/{}_fgsm_{}_{}.png'.format(i, pred.argmax(), target[0]))

	print('After fgsm -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

	'''

	attack_sample = igsm(model=model, x=clean_sample, y=target, lr=0.1, eps=0.25, niter=10)
	pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
	sample_pil = to_pil(attack_sample[0])
	sample_pil.save('./samples/{}_igsm_{}_{}.png'.format(i, pred.argmax(), target[0]))

	print('After igsm -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

	'''

	'''
	attack_sample = deepfool(model=model, x=clean_sample, num_classes=10)
	pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
	sample_pil = to_pil(attack_sample[0])
	sample_pil.save('./samples/{}_deepfool_{}_{}.png'.format(i, pred.argmax(), pred.argmax()))

	print('After deepfool -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))
	'''
