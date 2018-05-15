#convert sasr into a pytorch dataset. 
#create a new data class, which takes an input as a list of sasr objects. Extension of data.Dataset class. 

import torch.utils.data as data
class sasr():
	def __init__(self):
		self.subtracted_image = None
		self.rationalization = None
