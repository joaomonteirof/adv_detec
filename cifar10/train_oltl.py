from __future__ import print_function
import argparse
import numpy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models

# Training settings
parser = argparse.ArgumentParser(description='Adv attacks/defenses on MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))   ])), batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.batch_size, shuffle=True, **kwargs)

model_1 = models.cnn()
model_2 = models.mlp()

if args.cuda:
	model_1.cuda()
	model_2.cuda()

optimizer_1 = optim.SGD(model_1.parameters(), lr=args.lr, momentum=args.momentum)
optimizer_2 = optim.SGD(model_2.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
	model_1.train()
	model_2.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data_v, target_v = Variable(data), Variable(target)
		optimizer_1.zero_grad()
		optimizer_2.zero_grad()

		output1_log, output1 = model_1.forward_oltl(data_v)
		output2_log, output2 = model_2.forward_oltl(data_v)

		target_1 = Variable(output1.data, requires_grad=False)
		target_2 = Variable(output2.data, requires_grad=False)

		loss1 = F.nll_loss(output1_log, target_v) + F.kl_div(output1_log, target_2)
		loss2 = F.nll_loss(output2_log, target_v) + F.kl_div(output2_log, target_1)

		loss1.backward()
		loss2.backward()
		optimizer_1.step()
		optimizer_2.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses: {:.6f}, {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss1.data[0], loss2.data[0]))

def test(epoch):
	model_1.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output, _ = model_1.forward_oltl(data)
		test_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss
	test_loss /= len(test_loader) # loss function already averages over batch size
	print('\nModel 1  --  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

	model_2.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output, _ = model_2.forward_oltl(data)
		test_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss
	test_loss /= len(test_loader) # loss function already averages over batch size
	print('\nModel 2  --  Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def save_model():
	print('Saving model...')
	ckpt = {'model1_state': model_1.state_dict(), 'model2_state': model_2.state_dict()}
	torch.save(ckpt, 'mnist_oltl.pt')

for epoch in range(1, args.epochs + 1):
	train(epoch)
	test(epoch)

save_model()
