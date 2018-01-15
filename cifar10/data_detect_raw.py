from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import vgg, resnet, densenet
import pickle
import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM, DeepFoolAttack, SaliencyMapAttack
import sys

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean Cifar10 samples')
parser.add_argument('--data-size', type=int, default=10000, metavar='N', help='Number of samples in the final dataset (default: 1e4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='vgg')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

if args.model == 'vgg':
	model = vgg.VGG('VGG16', soft=args.soft)
elif args.model == 'resnet':
	model = resnet.ResNet18(soft=args.soft)
elif args.model == 'densenet':
	model = resnet.densenet_cifar(soft=args.soft)

model_id = args.model_path + 'cifar_' + args.model + ('_soft' if args.soft else '') + '.pt'

mod_state = torch.load(model_id, map_location = lambda storage, loc: storage)
model.load_state_dict(mod_state['model_state'])
model.eval()

fool_model = PyTorchModel(model, bounds=(0,1), num_classes=10, cuda=False)
attack = DeepFoolAttack(fool_model)
#attack = FGSM(fool_model)
#attack = SaliencyMapAttack(fool_model)


if args.cuda:
	model.cuda()

print(model_id[:-3])

data = []

for i in range(args.data_size):

	sys.stdout.flush()
	sys.stdout.write('\rSample {}/{}'.format(i+1, args.data_size))

	index = i + np.random.randint(10)

	clean_sample, target = trainset[index]

	clean_sample, target = clean_sample.unsqueeze(0), np.asarray([target]).reshape(1,1)

	if args.cuda:
		clean_sample = clean_sample.cuda(), target.cuda()

	pred_clean = model.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()

	if np.random.rand() > 0.5:

		attack_sample = attack(image=clean_sample.numpy()[0], label=pred_clean.argmax())
		try:
			pred_attack = model.forward(Variable(torch.from_numpy(attack_sample).unsqueeze(0), requires_grad=False)).data.cpu().numpy()
			sample = np.concatenate([attack_sample.reshape(1,-1), np.ones([1,1]), pred_clean.argmax().reshape(1,-1), pred_attack.argmax().reshape(1,-1), target ], 1)
		except RuntimeError:
			sample = np.concatenate([clean_sample[0].cpu().numpy().reshape(1,-1), np.zeros([1,1]), pred_clean.argmax().reshape(1,-1), pred_clean.argmax().reshape(1,-1), target ], 1)
	else:
		sample = np.concatenate([clean_sample[0].cpu().numpy().reshape(1,-1), np.zeros([1,1]), pred_clean.argmax().reshape(1,-1), pred_clean.argmax().reshape(1,-1), target ], 1)

	data.append(sample)

data = np.asarray(data)

pfile = open('./detec.p', 'wb')
pickle.dump(data.squeeze(), pfile)
pfile.close()
