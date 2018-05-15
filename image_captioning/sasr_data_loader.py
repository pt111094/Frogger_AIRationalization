from sasr import sasr
from xlrd import open_workbook
import torch
import re
import pickle
from math import floor
from collections import Counter
from build_vocab import Vocabulary
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import nltk
from PIL import ImageChops
from PIL import Image, ImageDraw
from random import shuffle
from sasr_dataset import combined_dataset

class SASR_Data_Loader():
	def __init__(self,vocab = None,transform = None):
		self.vocab = vocab
		self.transform = transform
		# self.transform = transforms.Compose([ 
		# 	transforms.RandomCrop(224),
		# 	transforms.RandomHorizontalFlip(), 
		# 	transforms.ToTensor(), 
		# 	transforms.Normalize((0.485, 0.456, 0.406), 
		# 						(0.229, 0.224, 0.225))])
		self.image_size = 224
		self.current_image_dir = './data/Frogger_Turk/Currrent_State/'
		self.next_image_dir = './data/Frogger_Turk/Next_State/'
		self.subtracted_testing_images_dir = './data/subtractedTurkNodiff/Testing/'
		self.subtracted_training_images_dir = './data/subtractedTurkNodiff/Training/'
		self.concatenated_images_dir = './data/concatenatedTurk/'
		self.output_dir = './data/FroggerDatasetTurk/'
		self.training_output_dir = './data/FroggerDatasetTurkTrainingNodiff/1/'
		self.training_output_dir_read = './data/FroggerDatasetTurkTrainingNodiff/'
		self.testing_output_dir = './data/FroggerDatasetTurkTestingNodiff/1/'
		self.training_rationalizations = None
		self.testing_rationalizations = None
		self.training_images = []
		self.testing_images = []
	#def get_data(dir,batch_size,transform):
	def crop_image(self,im,next_im):
		isize = im.size
		frog =  Image.open('./png/frog.png')
		fsize = frog.size
		x0, y0 = fsize [0] // 2, fsize [1] // 2
		pixel = frog.getpixel((x0 + 10, y0 + 10))[:-1]
		best = (100000, 0, 0)
		for x in range (isize[0]):
			for y in range (isize[1]):
				ipixel = im.getpixel ((x, y))
				d = diff (ipixel, pixel)
				if d < best[0]: best = (d, x, y)

		x, y = best [1:]
		rect = (150,150)
		im2 = im.crop((x - x0 + 10 - rect[0]/2, y - y0 + 5 - rect[1]/2, x + 2*x0 + rect[0]/2, y + 2*y0 + rect[1]/2))
		im2_array = np.asarray(im2)
		im2_array.setflags(write=1)
		im2_array[np.where((im2_array==[0,0,0,0]).all(axis=2))] = [0,0,0,255]
		im3 = next_im.crop((x - x0 + 10 - rect[0]/2, y - y0 + 5 - rect[1]/2, x + 2*x0 + rect[0]/2, y + 2*y0 + rect[1]/2))
		im3_array = np.asarray(im3)
		im3_array.setflags(write=1)
		im3_array[np.where((im3_array==[0,0,0,0]).all(axis=2))] = [0,0,0,255]
		# for i in range(im2_array.shape[0]):
		# 	for j in range(im2_array.shape[1]):
		# 		if im2_array[i][j][0]==0 and im2_array[i][j][1]==0 and im2_array[i][j][2]==0 and im2_array[i][j][3]==0:
		# 			im2_array[i][j][0]=0
		# 			im2_array[i][j][1]=0
		# 			im2_array[i][j][2]=0
		# 			im2_array[i][j][3]=255
		im2 = Image.fromarray(im2_array,'RGBA')
		im3 = Image.fromarray(im3_array,'RGBA')
		return im2,im3
	def subtract_and_concatenate_images(self,current_image_dir, next_image_dir, output_dir,tr_indices,te_indices,good_ids):
		"""Resize the images in 'image_dir' and save into 'output_dir'."""
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		if not os.path.exists(self.subtracted_training_images_dir):
			os.makedirs(self.subtracted_training_images_dir)
		if not os.path.exists(self.subtracted_testing_images_dir):
			os.makedirs(self.subtracted_testing_images_dir)

		current_images = os.listdir(current_image_dir)
		current_images.sort(key=lambda f: int(filter(str.isdigit, f)))
		next_images = os.listdir(next_image_dir)
		next_images.sort(key=lambda f: int(filter(str.isdigit, f)))
		num_images = len(current_images)
		imgs = []
		combined_images = []
		# print(len(current_images))
		# print(current_images[0])
		for i, image in enumerate(current_images):
			if os.path.isdir(current_image_dir + image):
				continue
			# print(image)
			if i in good_ids:
				if good_ids[tr_indices[0]]<=i and i<=good_ids[tr_indices[1]]:
					# imgs.append(self.crop_image(Image.open(current_image_dir + current_images[i])))
					# imgs.append(self.crop_image(Image.open(next_image_dir + next_images[i])))
					imgs.append(Image.open(current_image_dir + current_images[i]))
					imgs.append(Image.open(next_image_dir + next_images[i]))
					# print(image)
					# diff = ImageChops.subtract(imgs[1],imgs[0])
					# diff = self.crop_image(diff)
					# print("after")
					# imgs[1] = diff
					imgs[0],imgs[1] = self.crop_image(imgs[0],imgs[1])
					# imgs[1] = self.crop_image(imgs[1])
					# imgs[0] = self.crop_image(imgs[0])
					imgs_comb = np.hstack(( np.asarray(j) for j in imgs))
					imgs_comb = Image.fromarray(imgs_comb)
					# combined_images.append(imgs_comb)
					# self.training_images.append(imgs_comb)
					imgs_comb.save(self.subtracted_training_images_dir + 'Frogger_State_' + str(i) + '.jpg' )
					imgs = []
				else:
					imgs.append(Image.open(current_image_dir + current_images[i]))
					imgs.append(Image.open(next_image_dir + next_images[i]))
					# print(image)
					# diff = ImageChops.subtract(imgs[1],imgs[0])
					# diff = self.crop_image(diff)
					# imgs[1] = diff
					imgs[0],imgs[1] = self.crop_image(imgs[0],imgs[1])
					# imgs[1] = self.crop_image(imgs[1])
					# imgs[0] = self.crop_image(imgs[0])
					imgs_comb = np.hstack(( np.asarray(j) for j in imgs))
					imgs_comb = Image.fromarray(imgs_comb)
					# combined_images.append(imgs_comb)
					# self.testing_images.append(imgs_comb)
					imgs_comb.save(self.subtracted_testing_images_dir + 'Frogger_State_' + str(i) + '.jpg' )
					imgs = []

			# print(i)
		return combined_images

	def resize_image(self,image, size):
		"""Resize an image to the given size."""
		return image.resize(size, Image.ANTIALIAS)
	def resize_images(self,image_dir, output_dir, size, output_data):
		"""Resize the images in 'image_dir' and save into 'output_dir'."""
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		images = os.listdir(image_dir)
		num_images = len(images)
		all_images = []
		for i, image in enumerate(images):
			with open(os.path.join(image_dir, image), 'r+b') as f:
				with Image.open(f) as img:
					img = self.resize_image(img, size)
					all_images.append(img)
					# output_data.append(img)
					img.save(os.path.join(output_dir, image), img.format)
	
	def load_data(self,data_file,flag):
		wb = open_workbook(data_file)
		vocab = self.vocab
		for sheet in wb.sheets():
			number_of_rows = sheet.nrows
			number_of_columns = sheet.ncols
			rationalizations = []
			items = []
			rows = []
			lengths = []
			max_length = 0

			bad_worker_ids = ['A2CNSIECB9UP05','A23782O23HSPLA','A2F9ZBSR6AXXND','A3GI86L18Z71XY','AIXTI8PKSX1D2','A2QWHXMFQI18GQ','A3SB7QYI84HYJT',
'A2Q2A7AB6MMFLI','A2P1KI42CJVNIA','A1IJXPKZTJV809','A2WZ0RZMKQ2WGJ','A3EKETMVGU2PM9','A1OCEC1TBE3CWA','AE1RYK54MH11G','A2ADEPVGNNXNPA',
'A15QGLWS8CNJFU','A18O3DEA5Z4MJD','AAAL4RENVAPML','A3TZBZ92CQKQLG','ABO9F0JD9NN54','A8F6JFG0WSELT','ARN9ET3E608LJ','A2TCYNRAZWK8CC',
'A32BK0E1IPDUAF','ANNV3E6CIVCW4']
			good_ids = []
			good_rationalizations = []
			counter = Counter()
			for row in range(1, number_of_rows):
				values = []
				# print(row)
				worker_id = sheet.cell(row,0).value
				if worker_id not in bad_worker_ids:
					good_ids.append(row-1)
					line = sheet.cell(row,4).value
					tokens = nltk.tokenize.word_tokenize(line.lower())
					# counter.update(tokens)
					line = line.lower()
					good_rationalizations.append(line)
					line = re.sub('[^a-z\ ]+', " ", line)
					
					words = line.split()
					length = len(tokens)
					lengths.append(length)
					# print(tokens)
					if length>max_length:
						max_length = length
					for index,word in enumerate(tokens): 
						tokens[index] = vocab.word2idx[word]
					rationalizations.append(words)
			rationalizations=[np.array(xi) for xi in rationalizations]
			# words = [word for word, cnt in counter.items() if cnt >= threshold]
		split = int(floor((90.0/100)*len(rationalizations)))
		
		tr = slice(0,split)
		tr_indices = [0,split-1]
		te_indices = [split,len(rationalizations)-1]
		te = slice(split,len(rationalizations))
		self.training_rationalizations = good_rationalizations[tr]
		self.testing_rationalizations = good_rationalizations[te]
		training_rationalizations_text = good_rationalizations[tr]
		testing_rationalizations_text = good_rationalizations[te]
		
		current_image_dir = self.current_image_dir
		next_image_dir = self.next_image_dir
		output_dir = self.output_dir
		concatenated_images_dir = self.concatenated_images_dir
		subtracted_images_dir = self.subtracted_training_images_dir
		image_size = [self.image_size, self.image_size]
		# concatenate_images(image_dir, concatenated_images_dir)
		if not flag:
			subtracted_images = self.subtract_and_concatenate_images(current_image_dir, next_image_dir, subtracted_images_dir,tr_indices,te_indices,good_ids)
			# exit(0)
			# exit(0)
			self.resize_images(self.subtracted_training_images_dir, self.training_output_dir, image_size, self.training_images)
			self.resize_images(self.subtracted_testing_images_dir, self.testing_output_dir, image_size, self.testing_images)
		

	def data_loader(self,batch_size = 5,transform = None,shuffle = False, num_workers = 2):
		# image_dataset = self.get_images(self.training_output_dir_read,1,self.transform)
		current_images = os.listdir(self.training_output_dir)
		current_images.sort(key=lambda f: int(filter(str.isdigit, f)))
		for i, image in enumerate(current_images):
			im = Image.open(self.training_output_dir + current_images[i])
			self.training_images.append(im.copy())
			# print(len(self.training_images))
			im.close()
		self.sasr_dataset = []
		j=0
		# print(len(self.training_images))
		# print(len(self.training_rationalizations))
		# while j<2:
		# self.training_images[300].save('image.png')
		# print(self.training_rationalizations[300])
		# exit(0)
		for i,image in enumerate(self.training_images):
			current_sasr = sasr()
			current_sasr.subtracted_images = image
			current_sasr.rationalization = self.training_rationalizations[i]
			self.sasr_dataset.append(current_sasr)
			# j+=1
		comb_dataset = combined_dataset(self.sasr_dataset,self.vocab,self.transform)
		data_load = torch.utils.data.DataLoader(dataset=comb_dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
		return data_load
		# shuffle(self.sasr_dataset)
		# final_dataset = []
		# rats = []
		# for i,sasr in enumerate(self.sasr_dataset):
		# 	if i%batch_size==0 or i==len(sasr_dataset)-1:
		# 		max_rat = max(rats,key=len)
		# 		max_length = len(max_rat)
		# 		rats.sort(key = lambda s: len(s))
		# 		rats.reverse()
		# 		captions = []
		# 		for index,r in enumerate(rats):
		# 			# print(max_length)
		# 			r = np.lib.pad(r,(0,max_length - len(r)),'constant')
		# 			captions.append(r)
		# 		lengths = [len(x) for x in rats]
		# 		lengths.sort()
		# 		lengths.reverse()
		# 		current_batch = torch.cat(current_batch_images,0)
		# 		final_dataset.append((current_batch,captions,lengths))
		# 		captions = []
		# 		lengths = []
		# 		current_batch_images = []
		# 		rats = []
		# 	else:
		# 		#create a list of the current batch of image tenors
		# 		#cat all of them at the end 
		# 		current_batch_images.append(sasr.subtracted_images)
		# 		rats.append(sasr.rationalization)


		# for i,images in enumerate(image_dataset):
		# 	j=0
		# 	for image in images: 
		# 		current_sasr = sasr()
		# 		current_sasr.subtracted_images = image
		# 		current_sasr.rationalizations = self.rationalizations[i*batch_size + j]
		# 		current_batch.append()

	def get_images(self,image_dir,batch_size,transform):
		trainset = torchvision.datasets.ImageFolder(root=image_dir, 
			transform = transform)
		image_dataset = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
			shuffle=False)
		return image_dataset
def diff (a, b):
    return sum ( (a - b) ** 2 for a, b in zip(a, b) )
def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).	
	We should build custom collate_fn rather than using default collate_fn, 
	because merging caption (including padding) is not supported in default.
	Args:
		data: list of tuple (image, caption). 
			image: torch tensor of shape (3, 256, 256).
			caption: torch tensor of shape (?); variable length.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]        
	return images, targets, lengths

# s = SASR_Data_Loader()
# s.load_data("filler")
# dat = s.data_loader("filler")
# for i,(im,captions,lengths) in enumerate(dat):
# 	print(im)
