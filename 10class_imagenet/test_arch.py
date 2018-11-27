import argparse
import torch
import torchvision
from models import vgg, resnet


# Training settings
parser = argparse.ArgumentParser(description='test arch')
parser.add_argument('--model', choices=['vgg', 'resnet'], default='vgg', help='Model')
args = parser.parse_args()

if args.model == 'vgg':
	features_model = torchvision.models.vgg19_bn(pretrained=False)
	model = vgg(features_model)
elif args.model == 'resnet':
	features_model = torchvision.models.resnet50(pretrained=False)
	model = resnet(features_model)

data = torch.rand(10,3,224,224)

print(data.size())

out = model(data)

print(out.size())

count = 0

for param in model.parameters():
	count+=param.numel()

print(count)
