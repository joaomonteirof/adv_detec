import argparse
import torch
from models import vgg, resnet, densenet

parser = argparse.ArgumentParser(description='Tiny Imagenet Classification')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='vgg')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG16')
elif args.model == 'resnet':
	model = resnet.ResNet18()
elif args.model == 'densenet':
	model = densenet.densenet_timgnet()

a = torch.rand(10,3,64,64)

b = model.forward(a)

print(b.size())
