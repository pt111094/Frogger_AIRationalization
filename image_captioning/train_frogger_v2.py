import argparse
import torch
import torch.nn as nn
import re
import numpy as np
import os
import random
from PIL import Image

import pickle
# from data_loader import get_loader
# from data_loader import get_images
# from sasr_data_loader import data_loader
# from sasr_data_loader import load_data
from sasr_data_loader_v2 import SASR_Data_Loader
from build_vocab import Vocabulary
from model_v2 import EncoderCNN, AttnDecoderRNN, ResidualBlock, AttnEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as FF

MAX_LENGTH = 30
END_TOKEN = 2
START_TOKEN = 1
teacher_forcing_ratio = 0.5

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    # transform = transforms.Compose([ 
    #     transforms.RandomCrop(args.crop_size),
    #     transforms.RandomHorizontalFlip(), 
    #     transforms.ToTensor(), 
    #     transforms.Normalize((0.485, 0.456, 0.406), 
    #                          (0.229, 0.224, 0.225))])

    # fiveCropTransform = transforms.Compose([
    #     transforms.FiveCrop(10), # this is a list of PIL Images
    #     transforms.Lambda(lambda crops: torch.stack([(crop) for crop in crops])) # returns a 4D tensor
    # ])

    # transform = transforms.Compose([ 
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(), 
    #     transforms.Normalize((0.485, 0.456, 0.406), 
    #                          (0.229, 0.224, 0.225))])
    transform = transforms.Compose([
        transforms.ColorJitter(contrast = 0.3,saturation = 0.3),
        # transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
        transforms.RandomAffine(0,translate = (0.1,0.1)),
        transforms.ToTensor(), 
        # transforms.Normalize((0.8, 0.7, 0.8), 
        #                     (1, 1, 1))
        ])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
    #                          transform, args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers)

    #initialize data loader object and begin data preprocessing 
    sasr_data_loader = SASR_Data_Loader(vocab,transform)
    sasr_data_loader.load_data(args.data_file,args.init_flag)
    frogger_data_loader = sasr_data_loader.data_loader(args.batch_size, 
                             transform,
                             shuffle=True, num_workers=args.num_workers) 
    # initialize encoder and decoder models
    encoder = AttnEncoder(ResidualBlock,[3,3,3], args.embed_size)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, 
    #                      len(vocab), args.num_layers)
    decoder = AttnDecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    stransform = transforms.ToPILImage()
    total_step = len(frogger_data_loader)
    for epoch in range(args.num_epochs):
        for i,(images,captions,lengths) in enumerate(frogger_data_loader):
            ##### Uncomment section to see transformed images
            # images = images.squeeze()
            # print(images.size())
            # c = stransform(images)
            # c.save('save_image.png')
            # exit(0)
            ###### Ends here
            images = to_var(images)
            captions = to_var(captions)
            
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            captions = captions.view(-1)
            outputs = outputs.view(-1,len(vocab))
            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            if (epoch%5==0):
                if (i+1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'decoder_v2-%d-%d.pkl' %(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'encoder_v2-%d-%d.pkl' %(epoch+1, i+1)))                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=256 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab_frogger.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--data_file', type=str, default='Turk_Master_File.xlsx',
                        help='name of the excel file')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=700,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--init_flag', type=bool , default=False ,
                        help='Whether or not data has been initialized')
    
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)