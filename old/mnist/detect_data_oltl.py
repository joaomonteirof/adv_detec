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
import pickle

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
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

model_1 = models.cnn()
model_2 = models.mlp()

if args.cuda:
	model_1.cuda()
	model_2.cuda()

model_id = args.model_path + 'mnist_oltl.pt'

mod_state = torch.load(model_id)
model_1.load_state_dict(mod_state['model1_state'])
model_2.load_state_dict(mod_state['model2_state'])

if args.cuda:
	model_1.cuda()
	model_2.cuda()

to_pil = transforms.ToPILImage()

eps_ = 0.25
ndata = 10000
data = []

for i in range(ndata):

	index = i + np.random.randint(10)

	clean_sample, target = list(train_loader)[index]

	if args.cuda:
		clean_sample, target = clean_sample.cuda(), target.cuda()

	if np.random.rand() > 0.5:

		attack_sample_1 = fgsm(model=model_1, x=clean_sample, y=target, eps=eps_)
		pred_1 = model_1.forward(Variable(attack_sample_1, requires_grad=False)).data.cpu().numpy()

		attack_sample_2 = fgsm(model=model_2, x=clean_sample, y=target, eps=eps_)
		pred_2 = model_2.forward(Variable(attack_sample_2, requires_grad=False)).data.cpu().numpy()

		pred = np.concatenate([pred_1, pred_2, np.ones([1,1])], 1)

	else:
		pred_1 = model_1.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()
		pred_2 = model_2.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()

		pred = np.concatenate([pred_1, pred_2, np.zeros([1,1])], 1)

	data.append(pred)

data = np.asarray(data)

pfile = open('./oltl_detec.p', 'wb')
pickle.dump(data.squeeze(), pfile)
pfile.close()
