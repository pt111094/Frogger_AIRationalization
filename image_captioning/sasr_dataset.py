import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import torchvision
import nltk
from PIL import Image
import numpy as np
from build_vocab import Vocabulary
class combined_dataset(data.Dataset):
	def __init__(self,sasr_dataset,vocab,transform):
		self.sasr_dataset = sasr_dataset
		self.vocab = vocab
		self.transform = transform
	def __getitem__(self,index):
		caption = []
		caption.append(self.vocab('<start>'))
		tokens = nltk.tokenize.word_tokenize(str(self.sasr_dataset[index].rationalization).lower())
		caption.extend([self.vocab(token) for token in tokens])
		caption.append(self.vocab('<end>'))
		target = torch.Tensor(caption)
		image = self.sasr_dataset[index].subtracted_images
		image = image.convert('RGB')
		if self.transform is not None:
			image = self.transform(image) 

		return image,target
	def __len__(self):
		return len(self.sasr_dataset)