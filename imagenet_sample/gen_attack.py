from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
import foolbox
from foolbox.models import PyTorchModel
from foolbox.attacks import FGSM, IterativeGradientSignAttack, DeepFoolAttack, SaliencyMapAttack, GaussianBlurAttack, SaltAndPepperNoiseAttack, AdditiveGaussianNoiseAttack
from PIL import Image

# Training settings
parser = argparse.ArgumentParser(description='Single ImageNet attack')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--sample-path', type=str, default='./sample.jpg', metavar='Path', help='Path for sample')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
resize = transforms.Resize(224)

im = Image.open(args.sample_path)
sample = to_tensor(resize(im))

model = models.vgg16(pretrained=True)
model.eval()

fool_model = PyTorchModel(model, bounds=(0,1), num_classes=1000, cuda=False)
attack = FGSM(fool_model)

clean_sample = Variable(sample.unsqueeze(0))
output = model(clean_sample)

target = output.data.max(1)[1][0]
confidence = output.data.max(1)[0][0]

attack_sample = attack(image=sample.numpy(), label=target)

attack_sample_var = Variable(torch.from_numpy(attack_sample).unsqueeze(0))
output_attack = model(attack_sample_var)
target_attack = output_attack.data.max(1)[1][0]
confidence_attack = output.data.max(1)[0][0]

print(target, target_attack)
print(confidence, confidence_attack)

mask = attack_sample_var.data - clean_sample.data
mask = (mask - mask.min()) / (mask.max()-mask.min())

clean_pil = to_pil(clean_sample.data[0].cpu())
attack_pil = to_pil(attack_sample_var.data[0].cpu())
mask_pil = to_pil(mask[0].cpu())

clean_pil.save('clean.bmp')
attack_pil.save('attack.bmp')
mask_pil.save('mask.bmp')
