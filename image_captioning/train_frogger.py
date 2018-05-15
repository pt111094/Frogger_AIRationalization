import argparse
import torch
import torch.nn as nn
import re
import numpy as np
import os
import pickle
# from data_loader import get_loader
# from data_loader import get_images
# from sasr_data_loader import data_loader
# from sasr_data_loader import load_data
from sasr_data_loader import SASR_Data_Loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

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

    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
    #                          transform, args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers) 
    sasr_data_loader = SASR_Data_Loader(vocab,transform)
    sasr_data_loader.load_data(args.data_file,args.init_flag)
    frogger_data_loader = sasr_data_loader.data_loader(args.batch_size, 
                             transform,
                             shuffle=True, num_workers=args.num_workers) 
    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    
    total_step = len(frogger_data_loader)
    for epoch in range(args.num_epochs):
        for i,(images,captions,lengths) in enumerate(frogger_data_loader):
            images = to_var(images, volatile=True)
            if (list(images.size())[0]!=1):
                captions = to_var(captions)
                # print(list(images.size())[0])
                # print(captions)
                # exit(0)
                
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                decoder.zero_grad()
                encoder.zero_grad()
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                          %(epoch, args.num_epochs, i, total_step, 
                            loss.data[0], np.exp(loss.data[0]))) 
                    
                # Save the models
                if (i+1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), 
                               os.path.join(args.model_path, 
                                            'encoder-%d-%d.pkl' %(epoch+1, i+1)))


    # exit(0)
    # total_step = len(data_loader)
    # for epoch in range(args.num_epochs):
    #     for i, (images, captions, lengths) in enumerate(data_loader):
    #         # print(captions)
    #         # print(images)
    #         # print(lengths)
    #         # print(captions)
    #         # # print(images)
    #         # exit(0)
    #         # Set mini-batch dataset
    #         images = to_var(images, volatile=True)
    #         print(captions)
    #         captions = to_var(captions)
    #         print(captions)
    #         print(lengths)
    #         targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
    #         # Forward, Backward and Optimize
    #         decoder.zero_grad()
    #         encoder.zero_grad()
    #         print(images)
    #         features = encoder(images)
    #         print(features)
    #         exit(0)
    #         outputs = decoder(features, captions, lengths)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #         # Print log info
    #         if i % args.log_step == 0:
    #             print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
    #                   %(epoch, args.num_epochs, i, total_step, 
    #                     loss.data[0], np.exp(loss.data[0]))) 
                
    #         # Save the models
    #         if (i+1) % args.save_step == 0:
    #             torch.save(decoder.state_dict(), 
    #                        os.path.join(args.model_path, 
    #                                     'decoder-%d-%d.pkl' %(epoch+1, i+1)))
    #             torch.save(encoder.state_dict(), 
    #                        os.path.join(args.model_path, 
    #                                     'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
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
    parser.add_argument('--save_step', type=int , default=500,
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