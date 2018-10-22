
# Importing the libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

from data_loader import get_loader
from model import ParRanker
from config import get_args
from utils import accuracy
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = get_loader(args.data_path, args.train_path, args.batch_size, args.data_shuffle)
val_loader = get_loader(args.data_path, args.val_path, args.batch_size, args.data_shuffle)
test_loader = get_loader(args.data_path, args.test_path, args.batch_size, args.data_shuffle)


# Creating the architecture of the Neural Network
model = ParRanker(args.data_path, args.word_list_path, args.glove_path,
                  args.word_dim, args.hidden_dim, args.dropout)
if torch.cuda.is_available():
    model.cuda()

# Print out the network information
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

criterion = nn.BCEWithLogitsLoss() # combines a Sigmoid
optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

best_epoch = 0
best_loss  = 9999.


def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = 0.
        model.train()
        for s, (x,len_x,y,len_y,l) in enumerate(train_loader):
            x = Variable(torch.stack(x, 1)).to(device)
            y = Variable(torch.stack(y, 1)).to(device)
            len_x = Variable(len_x).to(device)
            len_y = Variable(len_y).to(device)
            l = Variable(l).to(device).float()

            logit = model(x,y,len_x,len_y)
            loss = criterion(logit, l)
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/s))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0.
            val_accs = 0.
            with torch.no_grad():
                for s, (x,len_x,y,len_y,l) in enumerate(val_loader):
                    x = Variable(torch.stack(x, 1)).to(device)
                    y = Variable(torch.stack(y, 1)).to(device)
                    len_x = Variable(len_x).to(device)
                    len_y = Variable(len_y).to(device)
                    l = Variable(l).to(device).float()

                    logit = model(x,y,len_x,len_y)
                    acc  = accuracy(logit.cpu(), l.cpu(), args.threshold)
                    loss = criterion(logit, l)
                    val_accs += acc
                    val_loss += loss.item()

            print('[val loss] : '+str(val_loss/s)+'[val accs] : '+str(val_accs/s))
            if best_loss > (val_loss/s):
                best_loss = (val_loss/s)
                best_epoch= epoch+1
                torch.save(model,
                       os.path.join(args.model_path,
                       'model-%d.pkl'%(epoch+1)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path,
                          'model-%d.pkl'%(best_epoch))).state_dict())
    model.eval()
    test_loss = 0.
    test_accs = 0.
    with torch.no_grad():
        for s, (x,len_x,y,len_y,l) in enumerate(test_loader):
                x = Variable(torch.stack(x, 1)).to(device)
                y = Variable(torch.stack(y, 1)).to(device)
                len_x = Variable(len_x).to(device)
                len_y = Variable(len_y).to(device)
                l = Variable(l).to(device).float()

                logit = model(x,y,len_x,len_y)
                acc  = accuracy(logit.cpu(), l.cpu(), args.threshold)
                loss = criterion(logit, l)
                test_accs += acc
                test_loss += loss.item()

    print('[test loss] : '+str(test_loss/s)+'[test accs] : '+str(test_accs/s))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
