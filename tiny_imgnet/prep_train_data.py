import argparse
import os
import shutil
import glob

def read_classes(path):
	with open(path, 'r') as file:
		classes = file.readlines()
	return [class_[:-1] for class_ in classes]

parser = argparse.ArgumentParser(description='Prepare Tiny Imagenet')
parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
parser.add_argument('--classes-file-path', type=str, default='./data/', metavar='Path', help='Path to data')
args = parser.parse_args()

classes = read_classes(	args.classes_file_path + 'wnids.txt')
classes.sort()

dirs = [d for d in os.listdir(args.data_path)]

if len(dirs) > 0:

	for data_folder in dirs:

		class_idx = str(classes.index(data_folder))

		print(args.data_path + data_folder, args.data_path + class_idx)

		os.makedirs(args.data_path + class_idx)

		files_list = glob.glob(args.data_path + data_folder + '/images/*.JPEG')

		if len(files_list) > 0:

			for file_ in files_list:
					file_name = file_.split('/')[-1]
					os.rename(args.data_path + data_folder + '/images/' + file_name, args.data_path + class_idx + '/' + file_name)
	
			shutil.rmtree(args.data_path + data_folder)

		else:

			print('No file found at {} '.format(args.data_path + data_folder + '/images/'))
			exit(1)

else:

	print('No dir found!')
	exit(1)
