import argparse
import os
import shutil

def read_classes(path):
	with open(path, 'r') as file:
		classes = file.readlines()
	return [class_[:-1] for class_ in classes]

def read_valid_info(path):
	with open(path, 'r') as file:
		classes = file.readlines()
	return [class_.split('\t')[0] for class_ in classes], [class_.split('\t')[1] for class_ in classes]

parser = argparse.ArgumentParser(description='Prepare Tiny Imagenet')
parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
parser.add_argument('--classes-file-path', type=str, default='./data/', metavar='Path', help='Path to data')
parser.add_argument('--valid-file-path', type=str, default='./data/', metavar='Path', help='Path to data')
args = parser.parse_args()

classes = read_classes(	args.classes_file_path + 'wnids.txt')
classes.sort()

files_list, classes_list = read_valid_info(args.valid_file_path + 'val_annotations.txt')

if len(classes) > 0 and len(classes_list) > 0 and len(files_list) > 0 and len(classes_list) == len(files_list):

	for i in range(len(classes)):
		if not os.path.isdir(args.data_path + str(i)):
			os.makedirs(args.data_path + str(i))

	for class_, file_ in zip(classes_list, files_list):

		class_idx = str(classes.index(class_))

		os.rename(args.data_path + 'images/' + file_, args.data_path + class_idx + '/' + file_)


	shutil.rmtree(args.data_path + 'images')

else:

	print('No dir found!')
	exit(1)
