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
from foolbox.attacks import FGSM, IterativeGradientSignAttack, DeepFoolAttack, SaliencyMapAttack, GaussianBlurAttack, SaltAndPepperNoiseAttack, AdditiveGaussianNoiseAttack, CarliniWagnerL2Attack
import sys

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean Cifar10 samples')
parser.add_argument('--data-path', type=str, default='./data/train/', metavar='Path', help='Path to data')
parser.add_argument('--data-size', type=int, default=10000, metavar='N', help='Number of samples in the final dataset (default: 1e4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

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

if args.cuda:
	model_1=model_1.cuda()
	model_2=model_2.cuda()
else:
	model_1=model_1.cpu()
	model_2=model_2.cpu()

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

fool_model_1 = PyTorchModel(model_1, bounds=(0,1), num_classes=10, preprocessing=(mean, std))
attack_1 = FGSM(fool_model_1)
fool_model_2 = PyTorchModel(model_2, bounds=(0,1), num_classes=10, preprocessing=(mean, std))
attack_2 = FGSM(fool_model_2)
#attack = FGSM(fool_model)
#attack = IterativeGradientSignAttack(fool_model)
#attack = DeepFoolAttack(fool_model)
#attack = SaliencyMapAttack(fool_model)

print(model_id_1[:-3])
print(model_id_2[:-3])

data = []

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

		if args.cuda:
			clean_sample = clean_sample.cuda()

		try:

			pred_attack_1 = model_1.forward(torch.from_numpy(attack_sample).unsqueeze(0)).detach().cpu().numpy()
			pred_attack_2 = model_2.forward(torch.from_numpy(attack_sample).unsqueeze(0)).detach().cpu().numpy()
			sample = np.concatenate([pred_attack_1, pred_attack_2, np.ones([1,1])], 1)
		except:
			pred_clean_1 = model_1.forward(clean_sample).detach().cpu().numpy()
			pred_clean_2 = model_2.forward(clean_sample).detach().cpu().numpy()
			sample = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)
	else:

		if args.cuda:
			clean_sample = clean_sample.cuda()

		pred_clean_1 = model_1.forward(clean_sample).detach().cpu().numpy()
		pred_clean_2 = model_2.forward(clean_sample).detach().cpu().numpy()
		sample = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)

	data.append(sample)

data = np.asarray(data)

pfile = open('./detec_fgsm.p', 'wb')
pickle.dump(data.squeeze(), pfile)
pfile.close()
