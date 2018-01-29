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
from foolbox.attacks import FGSM, IterativeGradientSignAttack, DeepFoolAttack, SaliencyMapAttack, GaussianBlurAttack, SaltAndPepperNoiseAttack, AdditiveGaussianNoiseAttack
import sys

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean mnist samples')
parser.add_argument('--n-samples', type=int, default=500, metavar='N', help='Number of samples per attack (default: 500)')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed (default: 10)')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()

torch.manual_seed(args.seed)

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

model = vgg.VGG('VGG16', soft=args.soft)

model_id = args.model_path + 'cifar_vgg' + ('_soft' if args.soft else '') + '.pt'

mod_state = torch.load(model_id, map_location = lambda storage, loc: storage)
model.load_state_dict(mod_state['model_state'])
model.eval()

fool_model = PyTorchModel(model, bounds=(0,1), num_classes=10, cuda=False)
attack_dict = {'fgsm':FGSM(fool_model), 'igsm':IterativeGradientSignAttack(fool_model), 'jsma':SaliencyMapAttack(fool_model), 'deepfool':DeepFoolAttack(fool_model), 'gaussianblur':GaussianBlurAttack(fool_model), 'gaussiannoise':AdditiveGaussianNoiseAttack(fool_model), 'saltandpepper':SaltAndPepperNoiseAttack(fool_model)}

mse_dict = {'fgsm':0.0, 'igsm':0.0, 'jsma':0.0, 'deepfool':0.0, 'gaussianblur':0.0, 'gaussiannoise':0.0, 'saltandpepper':0.0}
success_rate_dict = {'fgsm':0.0, 'igsm':0.0, 'jsma':0.0, 'deepfool':0.0, 'gaussianblur':0.0, 'gaussiannoise':0.0, 'saltandpepper':0.0}

print(model_id[:-3])

for i in range(args.n_samples):

	sys.stdout.flush()
	sys.stdout.write('\rSample {}/{}'.format(i+1, args.n_samples))

	index = np.random.randint(trainset.__len__())

	clean_sample, target = trainset[index]
	clean_sample, target = clean_sample, np.asarray([target]).reshape(1,1)

	for attack in attack_dict:

		attacker = attack_dict[attack]
		attack_sample = attacker(image=clean_sample.numpy(), label=target[0,0])
		try:
			mse = ((attack_sample-clean_sample)**2).mean() / args.n_samples
			success_rate = 1 / args.n_samples
		except TypeError:
			mse = 0.0
			success_rate = 0.0
		mse_dict[attack] += mse
		success_rate_dict[attack] += success_rate

print(mse_dict)
print(success_rate_dict)
