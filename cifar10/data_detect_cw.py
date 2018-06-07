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
import sys
from cw import AttackCarliniWagnerL2

# Training settings
parser = argparse.ArgumentParser(description='Adversarial/clean Cifar10 samples')
parser.add_argument('--data-size', type=int, default=10000, metavar='N', help='Number of samples in the final dataset (default: 1e4)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./trained_models/', metavar='Path', help='Path for model load')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

model_1 = vgg.VGG('VGG16', soft=args.soft)
model_2 = resnet.ResNet18(soft=args.soft)

model_id_1 = args.model_path + 'cifar_vgg' + ('_soft' if args.soft else '') + '.pt'
model_id_2 = args.model_path + 'cifar_resnet' + ('_soft' if args.soft else '') + '.pt'

mod_state_1 = torch.load(model_id_1, map_location = lambda storage, loc: storage)
model_1.load_state_dict(mod_state_1['model_state'])
model_1.eval()

mod_state_2 = torch.load(model_id_2, map_location = lambda storage, loc: storage)
model_2.load_state_dict(mod_state_2['model_state'])
model_2.eval()

if args.cuda:
	model_1.cuda()
	model_2.cuda()

print(model_id_1[:-3])
print(model_id_2[:-3])

attack = AttackCarliniWagnerL2(targeted=False, max_steps=1000, search_steps=6, cuda=args.cuda)

data_logits = []
data_images = []

for i in range(args.data_size):

	sys.stdout.flush()
	sys.stdout.write('\rSample {}/{}'.format(i+1, args.data_size))

	index = i + np.random.randint(10)

	clean_sample, target = trainset[index]

	clean_sample, target = clean_sample.unsqueeze(0), torch.LongTensor([target])

	if args.cuda:
		clean_sample = clean_sample.cuda(), target.cuda()

	clean_sample, target = Variable(clean_sample), Variable(target)

	if np.random.rand() > 0.5:
		if np.random.rand() > 0.5:
			attack_sample = attack.run(model_1, clean_sample.data, target.data)
		else:
			attack_sample = attack.run(model_2, clean_sample.data, target.data)
		try:
			pred_attack_1 = model_1.forward(attack_sample).data.cpu().numpy()
			pred_attack_2 = model_2.forward(attack_sample).data.cpu().numpy()
			sample_logits = np.concatenate([pred_attack_1, pred_attack_2, np.ones([1,1])], 1)
			sample_image = attack_sample.data.contiguous().cpu().numpy()
		except RuntimeError:
			pred_clean_1 = model_1.forward(clean_sample).data.cpu().numpy()
			pred_clean_2 = model_2.forward(clean_sample).data.cpu().numpy()
			sample_logits = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)
			sample_image = clean_sample.data.cpu().numpy()
	else:
			pred_clean_1 = model_1.forward(clean_sample).data.cpu().numpy()
			pred_clean_2 = model_2.forward(clean_sample).data.cpu().numpy()
			sample_logits = np.concatenate([pred_clean_1, pred_clean_2, np.zeros([1,1])], 1)
			sample_image = clean_sample.data.cpu().numpy()

	data_logits.append(sample_logits)
	data_images.append(sample_image)

data_logits = np.asarray(data_logits)
data_images = np.asarray(data_images)

pfile = open('./cw_logits.p', 'wb')
pickle.dump(data_logits.squeeze(), pfile)
pfile.close()

pfile = open('./cw_images.p', 'wb')
pickle.dump(data_images.squeeze(), pfile)
pfile.close()
