import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet(nn.Module):
	def __init__(self, feat_model):
		super(resnet, self).__init__()

		self.conv1 = feat_model.conv1
		self.bn1 = feat_model.bn1
		self.relu = feat_model.relu
		self.maxpool = feat_model.maxpool

		self.layer1 = feat_model.layer1
		self.layer2 = feat_model.layer2
		self.layer3 = feat_model.layer3
		self.layer4 = feat_model.layer4

		self.avgpool = feat_model.avgpool

		self.fc = nn.Linear(2048, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

class vgg(nn.Module):
	def __init__(self, feat_model):
		super(vgg, self).__init__()

		self.features = feat_model.features

		self.fc = nn.Sequential( nn.Linear(512*7*7, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 10) )
			

	def forward(self, x):
		x = self.features(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
