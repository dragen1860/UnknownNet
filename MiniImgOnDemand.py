from torchvision.transforms import transforms
from PIL import Image
import csv
import collections, os
import  numpy as np
import random


class MiniImgOnDemand:
	"""
	put mini-imagenet files as :
	root :
		|- images/*.jpg includes all imgeas
		|- train.csv
		|- test.csv
		|- val.csv
	"""

	def __init__(self, root, type='train'):
		self.root = root
		self.resize = 84
		self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
		                                     lambda x: x.resize((self.resize, self.resize)),
		                                     transforms.ToTensor(),
		                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
		                                     ])

		def loadSplit(splitFile):
			"""
			return a dict saving the information of csv
			:param splitFile: csv file name
			:return: {label:[file1, file2 ...]}
			"""
			dictLabels = {}
			with open(splitFile) as csvfile:
				csvreader = csv.reader(csvfile, delimiter=',')
				next(csvreader, None)  # skip (filename, label)
				for i, row in enumerate(csvreader):
					filename = row[0]
					label = row[1]
					# append filename to current label
					if label in dictLabels.keys():
						dictLabels[label].append(filename)
					else:
						dictLabels[label] = [filename]
			return dictLabels

		# requiredFiles = ['train','val','test']
		self.ImagesDir = os.path.join(root, 'images')  # image path
		self.data = loadSplit(splitFile=os.path.join(root, 'train' + '.csv'))  # csv path
		self.data = collections.OrderedDict(sorted(self.data.items()))  # sort dict by !key! not value.
		# as the label of imagenet is not from 0, here re-label it
		self.classes_dict = {list(self.data.keys())[i]: i for i in
		                     range(len(self.data.keys()))}  # key1:0, key2:1, key3:2 ...
		tmp = {}
		for k, v in self.data.items():
			tmp[self.classes_dict[k]]=v
		self.data = tmp

		print(self.data.keys())

		self.cur_label = -1
		self.buff = []

	def get(self, label):
		if label != self.cur_label:
			self.batch(label)
		num = int( len(self.buff) * 0.9 )
		sample = random.sample(self.buff[:num], 1)
		sample = self.transform(sample[0])
		# size: [3, 224, 224]
		return sample

	def get_test(self, label):
		if label != self.cur_label:
			self.batch(label)
		num = int( len(self.buff) * 0.9 )
		sample = random.sample(self.buff[num:], 1)
		sample = self.transform(sample[0])
		# size: [3, 224, 224]
		return sample

	def batch(self, label):
		"""
		read all data into memory.
		:param label:
		:return:
		"""
		imgs = self.data[label]
		random.shuffle(imgs)
		imgs = list(map(lambda  x: os.path.join(self.root, 'images', x), imgs))
		self.buff =  imgs



if __name__ == '__main__':
	from matplotlib import pyplot as plt
	db = MiniImgOnDemand('../mini-imagenet/', type='train')

	label = random.randint(0, 63)
	for i in range(16):
		plt.subplot(4,4,i+1)
		plt.imshow(db.get(label).transpose(1,2).transpose(0,2).numpy())
	plt.show()