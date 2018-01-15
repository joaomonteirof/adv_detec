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

import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM



# Training settings
parser = argparse.ArgumentParser(description='Adv attacks/defenses on MNIST')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./models/', metavar='Path', help='Path for model load')
parser.add_argument('--model', choices=['cnn', 'mlp'], default='cnn')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

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
model.eval()

fool_model = PyTorchModel(model, bounds=(0,1), num_classes=10, cuda=False)
attack = FGSM(fool_model)

if args.cuda:
	model.cuda()

print(model_id[:-3])

successes = 0
ntrials = 5

for i in range(ntrials):

	#index = np.random.randint(len(train_loader))
	index = i + np.random.randint(10)

	clean_sample, target = list(train_loader)[index]

	if args.cuda:
		clean_sample, target = clean_sample.cuda(), target.cuda()

	pred = model.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()

	print('Clean sample -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

	attack_sample = attack(image=clean_sample.numpy()[0], label=target[0])

	pred = model.forward(Variable(torch.from_numpy(attack_sample).unsqueeze(0), requires_grad=False)).data.cpu().numpy()

	print('Attack sample -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

	if pred.argmax() != target[0]:
		successes += 1

print('Success rate: {}'.format(successes/ntrials))
