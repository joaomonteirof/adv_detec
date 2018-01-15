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
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--model-path', type=str, default='./models/', metavar='Path', help='Path for model load')
parser.add_argument('--soft', action='store_true', default=False, help='Adds extra softmax layer')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

if args.soft:
	model_1 = models.cnn_soft()
	model_2 = models.mlp_soft()
else:
	model_1 = models.cnn()
	model_2 = models.mlp()

model_id_1 = args.model_path + 'mnist_cnn' + ('_soft' if args.soft else '') + '.pt'
model_id_2 = args.model_path + 'mnist_mlp' + ('_soft' if args.soft else '') + '.pt'

mod1_state = torch.load(model_id_1)
mod2_state = torch.load(model_id_2)
model_1.load_state_dict(mod1_state['model_state'])
model_2.load_state_dict(mod2_state['model_state'])

if args.cuda:
	model_1.cuda()
	model_2.cuda()

to_pil = transforms.ToPILImage()

eps_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
ntrials = 1000

print(model_id_1)
print(model_id_2)

for eps_ in eps_list:

	successes_1_1 = 0
	successes_1_2 = 0
	successes_2_1 = 0
	successes_2_2 = 0
	successes_1_ = 0
	successes_2_ = 0

	for i in range(ntrials):

		#index = np.random.randint(len(train_loader))
		index = i + np.random.randint(10)

		clean_sample, target = list(train_loader)[index]

		if args.cuda:
			clean_sample, target = clean_sample.cuda(), target.cuda()

		#pred = model.forward(Variable(clean_sample, requires_grad=False)).data.cpu().numpy()

		attack_sample_1 = fgsm(model=model_1, x=clean_sample, y=target, eps=eps_)


		attack_sample_2 = fgsm(model=model_2, x=clean_sample, y=target, eps=eps_)

		pred_1_1 = model_1.forward(Variable(attack_sample_1, requires_grad=False)).data.cpu().numpy()
		pred_1_2 = model_1.forward(Variable(attack_sample_2, requires_grad=False)).data.cpu().numpy()
		pred_2_2 = model_2.forward(Variable(attack_sample_2, requires_grad=False)).data.cpu().numpy()
		pred_2_1 = model_2.forward(Variable(attack_sample_1, requires_grad=False)).data.cpu().numpy()

		if pred_1_1.argmax() != target[0]:
			successes_1_1 += 1

		if pred_2_2.argmax() != target[0]:
			successes_2_2 += 1

		if pred_1_2.argmax() != target[0]:
			successes_1_2 += 1

		if pred_2_1.argmax() != target[0]:
			successes_2_1 += 1

		if pred_1_1.argmax() != pred_2_1.argmax():
			successes_1_ += 1

		if pred_1_2.argmax() != pred_2_2.argmax():
			successes_2_ += 1

		'''

		print('Clean sample -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

		sample_pil = to_pil(clean_sample[0])
		sample_pil.save('./samples/{}_clean_{}_{}.png'.format(i, pred.argmax(), target[0]))

		attack_sample = fgsm(model=model, x=clean_sample, y=target, eps=0.2)
		pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
		sample_pil = to_pil(attack_sample[0])
		sample_pil.save('./samples/{}_fgsm_{}_{}.png'.format(i, pred.argmax(), target[0]))

		print('After fgsm -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))


		attack_sample = igsm(model=model, x=clean_sample, y=target, lr=0.1, eps=0.25, niter=10)
		pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
		sample_pil = to_pil(attack_sample[0])
		sample_pil.save('./samples/{}_igsm_{}_{}.png'.format(i, pred.argmax(), target[0]))

		print('After igsm -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

		attack_sample = deepfool(model=model, x=clean_sample, num_classes=10)
		pred = model.forward(Variable(attack_sample, requires_grad=False)).data.cpu().numpy()
		sample_pil = to_pil(attack_sample[0])
		sample_pil.save('./samples/{}_deepfool_{}_{}.png'.format(i, pred.argmax(), pred.argmax()))

		print('After deepfool -- Class prediction: {}, Actual class: {}'.format(pred.argmax(), target[0]))

		'''

	print('Epsilon: {}, Success rate - m1a1, m1a2, m2a1, m2a2, m1m2_1, m2m1_2: {}, {}, {}, {}, {}, {}'.format(eps_, successes_1_1/ntrials, successes_1_2/ntrials, successes_2_1/ntrials, successes_2_2/ntrials, successes_1_/ntrials, successes_2_/ntrials))
