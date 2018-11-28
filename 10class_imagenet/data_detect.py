from __future__ import print_function
import argparse
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import vgg, resnet
import pickle
import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM, LinfinityBasicIterativeAttack, DeepFoolAttack, SaliencyMapAttack, GaussianBlurAttack, SaltAndPepperNoiseAttack, AdditiveGaussianNoiseAttack, CarliniWagnerL2Attack
import sys
import os

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean Cifar10 samples')
parser.add_argument('--data-path', type=str, default='./data/train/', metavar='Path', help='Path to data')
parser.add_argument('--data-size', type=int, default=10000, metavar='N', help='Number of samples in the final dataset (default: 1e4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
parser.add_argument('--attack', choices=['fgsm', 'igsm', 'jsma', 'deepfool', 'cw', 'gaussianblur', 'gaussiannoise', 'saltandpepper'], default='fgsm')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('generating attack: ' + args.attack)

id2att = {'fgsm':FGSM, 'igsm':LinfinityBasicIterativeAttack, 'jsma':SaliencyMapAttack, 'deepfool':DeepFoolAttack, 'cw':CarliniWagnerL2Attack, 'gaussianblur':GaussianBlurAttack, 'gaussiannoise':AdditiveGaussianNoiseAttack, 's:AdditiveGaussianNoiseAttackaltandpepper':SaltAndPepperNoiseAttack}

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

trainset = datasets.ImageFolder(args.data_path, transform=data_transform)

features_model_1 = torchvision.models.vgg19_bn(pretrained=False)
features_model_2 = torchvision.models.resnet50(pretrained=False)

model_1 = vgg(features_model_1)
model_2 = resnet(features_model_2)

model_id_1 = args.model_path + 'img_vgg.pt'
model_id_2 = args.model_path + 'img_resnet.pt'

mod_state_1 = torch.load(model_id_1, map_location = lambda storage, loc: storage)
model_1.load_state_dict(mod_state_1['model_state'])
model_1.eval()

mod_state_2 = torch.load(model_id_2, map_location = lambda storage, loc: storage)
model_2.load_state_dict(mod_state_2['model_state'])
model_2.eval()

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

fool_model_1 = PyTorchModel(model_1, bounds=(0,1), num_classes=10, preprocessing=(mean, std), device='cuda:0' if args.cuda else 'cpu')
fool_model_2 = PyTorchModel(model_2, bounds=(0,1), num_classes=10, preprocessing=(mean, std), device='cuda:0' if args.cuda else 'cpu')

if args.attack == 'igsm':
	attack_1 = id2att[args.attack](fool_model_1, distance=foolbox.distances.Linfinity)
	attack_2 = id2att[args.attack](fool_model_2, distance=foolbox.distances.Linfinity)
else:
	attack_1 = id2att[args.attack](fool_model_1)
	attack_2 = id2att[args.attack](fool_model_2)

data = []
images = []

for i in range(args.data_size):

	sys.stdout.flush()
	sys.stdout.write('\rSample {}/{}'.format(i+1, args.data_size))

	index = (i + np.random.randint(10))%len(trainset)

	clean_sample, target = trainset[index]

	clean_sample, target = clean_sample.unsqueeze(0), np.asarray([target]).reshape(1,1)

	if np.random.rand() > 0.5:
		if np.random.rand() > 0.5:
			attack_sample = attack_1(input_or_adv=clean_sample.cpu().numpy()[0], label=target[0,0])
		else:
			attack_sample = attack_2(input_or_adv=clean_sample.cpu().numpy()[0], label=target[0,0])

		try:

			attack_sample = torch.from_numpy((attack_sample-mean)/std).unsqueeze(0).float()

			if args.cuda:
				attack_sample = attack_sample.cuda()

			pred_attack_1 = model_1.forward(attack_sample).detach().cpu().numpy()
			pred_attack_2 = model_2.forward(attack_sample).detach().cpu().numpy()
			sample = np.concatenate([pred_attack_1, pred_attack_2, np.ones([1,1])], 1)
			image_sample = attack_sample.cpu().numpy()[0]
		except:

			if args.cuda:
				clean_sample = clean_sample.cuda()

			pred_clean_1 = model_1.forward(clean_sample).detach().cpu().numpy()
			pred_clean_2 = model_2.forward(clean_sample).detach().cpu().numpy()
			sample = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)
			image_sample = clean_sample.cpu().numpy()[0]
	else:

		if args.cuda:
			clean_sample = clean_sample.cuda()

		pred_clean_1 = model_1.forward(clean_sample).detach().cpu().numpy()
		pred_clean_2 = model_2.forward(clean_sample).detach().cpu().numpy()
		sample = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)
		image_sample = clean_sample.cpu().numpy()[0]

	data.append(sample)
	images.append(image_sample)

data = np.asarray(data)
image_data = np.asarray(images)

data_file = './detec_'+args.attack+'.p'
image_data_file = './raw_imgnet_'+args.attack+'.p'

if os.path.isfile(data_file):
	os.remove(data_file)
	print(data_file+' Removed')

if os.path.isfile(image_data_file):
	os.remove(image_data_file)
	print(image_data_file+' Removed')

pfile = open(data_file, 'wb')
pickle.dump(data.squeeze(), pfile)
pfile.close()

pfile = open(image_data_file, 'wb')
pickle.dump(image_data, pfile)
pfile.close()
